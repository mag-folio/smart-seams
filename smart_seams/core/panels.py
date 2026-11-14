"""Panel-based seam solver (v2).

Artists don't place seams edge by edge. They look at the object as a set of
*panels* (the tabletop, each side of a leg, a hole wall, a bevel band) and
decide which panels to keep together on one UV shell. This solver works the
same way:

1. **Panels.** Faces are flood-filled across smooth edges (dihedral angle
   below ``panel_angle``). Every seam therefore runs along a complete
   feature line, so a seam can never stop half way round a hole or leave
   stray whiskers.
2. **Tube cuts.** A panel that wraps round (cylinder wall, hole wall, the
   sides of a leg when they are smooth) cannot lie flat. It gets exactly one
   cut, along an edge loop, aligned across stacked rings and placed where it
   is least visible.
3. **Splitting.** A panel that still cannot flatten without stretching (a
   dome, a strongly curved blob) is split in two along its dominant normal
   direction, recursively.
4. **Shells.** Panels are merged into UV shells, smallest and thinnest
   panels first (bevel bands get absorbed into the surface they trim), then
   across the most *visible* shared borders, so the seams that remain are
   the hidden ones. Every merge is test-flattened and refused if it would
   stretch past the limit or overlap itself. If the shell count is still
   above ``max_shells``, the stretch limit is relaxed step by step. Overlap
   is never accepted.
"""

import math

from .seams import (bevel_crease_edges, complete_feature_lines, loop_smoothed_costs,
                    tube_ring_cuts)


class PanelSettings:
    def __init__(self, *, panel_angle=40.0, max_shells=10,
                 stretch_steps=(0.08, 0.12, 0.2, 0.3), thin_ratio=0.05,
                 loop_smoothing=0.85, sharp_bonus=0.35, concave_bonus=0.6,
                 max_split_depth=6):
        self.panel_angle = panel_angle          # edges sharper than this separate panels
        self.max_shells = max_shells            # hard limit on UV shells
        self.stretch_steps = stretch_steps      # stretch limits tried in order
        self.thin_ratio = thin_ratio            # panel is 'thin' if area/(perimeter/2)^2 < this
        self.loop_smoothing = loop_smoothing
        self.sharp_bonus = sharp_bonus
        self.concave_bonus = concave_bonus
        self.max_split_depth = max_split_depth
        # names used by charts.seam_cost / loop_smoothed_costs
        self.angle_bucket = 10.0


def _angle(e):
    return math.degrees(e.calc_face_angle(0.0)) if len(e.link_faces) == 2 else 180.0


class PanelSolver:
    def __init__(self, bm, visibility, settings, validator):
        self.bm = bm
        self.vis = visibility
        self.s = settings
        self.validate = validator
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        self.area = [f.calc_area() for f in bm.faces]
        self.angle = [_angle(e) for e in bm.edges]
        self.cost = loop_smoothed_costs(bm, visibility, settings)
        self.glued = [False] * len(bm.edges)
        self.stats = {}
        self.log = []
        # faces of the panel at one end of each side strip: both caps
        # attach to it, so the shell is a strip with the caps at its foot
        self.anchors = []

    # ------------------------------------------------------------------ panels
    def build_panels(self):
        bm, s = self.bm, self.s
        panel_of = [None] * len(bm.faces)
        panels = []
        self.creases = bevel_crease_edges(bm, s.panel_angle, self.angle)
        self.stats["bevel_creases"] = len(self.creases)
        strong = {e.index for e in bm.edges
                  if len(e.link_faces) == 2 and self.angle[e.index] >= s.panel_angle} | self.creases
        ext = complete_feature_lines(bm, strong, self.angle, 0.5 * s.panel_angle)
        self.creases |= ext
        self.stats["line_completions"] = len(ext)
        for f in bm.faces:
            if panel_of[f.index] is not None:
                continue
            pid = len(panels)
            members = [f.index]
            panel_of[f.index] = pid
            stack = [f]
            while stack:
                g = stack.pop()
                for e in g.edges:
                    if (len(e.link_faces) != 2 or self.angle[e.index] >= s.panel_angle
                            or e.index in self.creases):
                        continue
                    for h in e.link_faces:
                        if panel_of[h.index] is None:
                            panel_of[h.index] = pid
                            members.append(h.index)
                            stack.append(h)
            panels.append(members)
        # glue everything inside a panel
        for e in bm.edges:
            lf = e.link_faces
            if len(lf) == 2 and panel_of[lf[0].index] == panel_of[lf[1].index]:
                self.glued[e.index] = True
        self.panels = panels
        self.stats["panels"] = len(panels)
        self.hole_edges = self._hole_loop_edges(panel_of)

    def _hole_loop_edges(self, panel_of):
        """Edges on the *inner* boundary loops of panels (hole rims).

        These are cut all the way round or glued all the way round, never
        partially: a seam that stops part way round a hole looks broken.
        """
        bm = self.bm
        by_panel = {}
        for e in bm.edges:
            lf = e.link_faces
            if len(lf) == 2 and panel_of[lf[0].index] == panel_of[lf[1].index]:
                continue
            for f in lf:
                by_panel.setdefault(panel_of[f.index], set()).add(e.index)
        holes = set()
        for pid, edges in by_panel.items():
            left = set(edges)
            comps = []
            while left:
                start = left.pop()
                comp = {start}
                stack = [start]
                while stack:
                    x = bm.edges[stack.pop()]
                    for v in x.verts:
                        for y in v.link_edges:
                            if y.index in left:
                                left.discard(y.index)
                                comp.add(y.index)
                                stack.append(y.index)
                comps.append(comp)
            if len(comps) < 2:
                continue
            comps.sort(key=lambda c: -sum(bm.edges[i].calc_length() for i in c))
            for comp in comps[1:]:
                holes |= comp
        return holes

    def _straight_runs(self, edge_ids, tol=1.02):
        """Split a border into maximal straight runs (box edges between corners)."""
        bm = self.bm
        ids = set(edge_ids)
        by_vert = {}
        for ei in ids:
            for v in bm.edges[ei].verts:
                by_vert.setdefault(v.index, []).append(ei)
        ends = [vi for vi, xs in by_vert.items() if len(xs) == 1]
        if any(len(xs) > 2 for xs in by_vert.values()):
            return []
        cur = ends[0] if ends else bm.edges[next(iter(ids))].verts[0].index
        order, verts = [], [cur]
        prev = None
        while len(order) < len(ids):
            nxt = [ei for ei in by_vert[cur] if ei != prev and ei not in order]
            if not nxt:
                break
            prev = nxt[0]
            order.append(prev)
            cur = bm.edges[prev].other_vert(bm.verts[cur]).index
            verts.append(cur)
        runs = []
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order):
                chord = (bm.verts[verts[j + 1]].co - bm.verts[verts[i]].co).length
                length = sum(bm.edges[k].calc_length() for k in order[i:j + 1])
                if length > chord * tol + 1e-12:
                    break
                j += 1
            runs.append(order[i:j])
            i = j
        runs.sort(key=lambda r: -sum(bm.edges[k].calc_length() for k in r))
        return runs

    def _shortest_cut(self, idx, src, dst):
        """Cheapest path of glued inner edges of face set ``idx`` from a vertex
        in ``src`` to one in ``dst`` (edge indices, or [] if none)."""
        import heapq
        fset = set(idx)
        adj = {}
        for fi in idx:
            for e in self.bm.faces[fi].edges:
                if self.glued[e.index] and all(g.index in fset for g in e.link_faces):
                    a, b = e.verts[0].index, e.verts[1].index
                    w = self.cost[e.index] + 1e-6
                    adj.setdefault(a, set()).add((b, w, e.index))
                    adj.setdefault(b, set()).add((a, w, e.index))
        dist = {v: 0.0 for v in src}
        heap = [(0.0, v) for v in src]
        heapq.heapify(heap)
        prev = {}
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist.get(v, 1e30):
                continue
            if v in dst:
                path = []
                while v in prev:
                    v, ei = prev[v]
                    path.append(ei)
                return path
            for w, c, ei in adj.get(v, ()):
                if d + c < dist.get(w, 1e30):
                    dist[w] = d + c
                    prev[w] = (v, ei)
                    heapq.heappush(heap, (d + c, w))
        return []

    def _ordered_path(self, edge_ids):
        """Edges of a simple path or loop in walking order (None if branched)."""
        bm = self.bm
        by_vert = {}
        for ei in edge_ids:
            for v in bm.edges[ei].verts:
                by_vert.setdefault(v.index, []).append(ei)
        if any(len(xs) > 2 for xs in by_vert.values()):
            return None
        ends = [vi for vi, xs in by_vert.items() if len(xs) == 1]
        cur = ends[0] if ends else bm.edges[edge_ids[0]].verts[0].index
        order, used = [], set()
        while True:
            nxt = [ei for ei in by_vert[cur] if ei not in used]
            if not nxt:
                break
            used.add(nxt[0])
            order.append(nxt[0])
            cur = bm.edges[nxt[0]].other_vert(bm.verts[cur]).index
        return order if len(order) == len(edge_ids) else None

    def _faces(self, idx):
        return [self.bm.faces[i] for i in idx]

    def _ok(self, idx, limit):
        return self.validate(self._faces(idx), self.glued, limit)

    # -------------------------------------------------------------- tube cuts
    def _boundary_loops(self, idx):
        """Vertex sets of the open-boundary loops of a face set (current glue)."""
        fset = set(idx)
        bedges = []
        for fi in idx:
            for e in self.bm.faces[fi].edges:
                inside = sum(1 for g in e.link_faces if g.index in fset)
                if inside == 1 or (inside == 2 and not self.glued[e.index]):
                    bedges.append(e)
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in bedges:
            a, b = (find(v.index) for v in e.verts)
            parent[a] = b
        groups = {}
        for e in bedges:
            for v in e.verts:
                groups.setdefault(find(v.index), set()).add(v.index)
        return list(groups.values())

    def _cut_path(self, idx, ring_cuts):
        """Cut a wrapping panel open: prefer the aligned ring cut edges inside
        it, else the cheapest path between two of its boundary loops."""
        import heapq
        fset = set(idx)
        inner = [e for fi in idx for e in self.bm.faces[fi].edges
                 if self.glued[e.index] and all(g.index in fset for g in e.link_faces)]
        inner_ids = {e.index for e in inner}
        planned = [ei for ei in ring_cuts if ei in inner_ids]
        if planned:
            return planned
        loops = self._boundary_loops(idx)
        if len(loops) < 2:
            return []
        loops.sort(key=len)
        src, dst = loops[0], loops[1]
        adj = {}
        for e in inner:
            a, b = e.verts[0].index, e.verts[1].index
            w = self.cost[e.index] + 1e-6
            adj.setdefault(a, []).append((b, w, e.index))
            adj.setdefault(b, []).append((a, w, e.index))
        dist = {v: 0.0 for v in src}
        prev = {}
        heap = [(0.0, v) for v in src]
        heapq.heapify(heap)
        end = None
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist.get(v, 1e30):
                continue
            if v in dst:
                end = v
                break
            for w, c, ei in adj.get(v, ()):
                nd = d + c
                if nd < dist.get(w, 1e30):
                    dist[w] = nd
                    prev[w] = (v, ei)
                    heapq.heappush(heap, (nd, w))
        if end is None:
            return []
        path = []
        v = end
        while v in prev:
            v, ei = prev[v]
            path.append(ei)
        return path

    def _split(self, idx, limit):
        """Split a face set that cannot flatten in two.

        Two candidate partitions are tried: a crease-following watershed
        between the faces at either end of the panel's normal spread, and the
        plain split by the sign of the normal along that spread. The one that
        leaves more of the area flattening within ``limit`` wins; on a tie, the
        one whose cut runs less over smooth edges. Returns ``(parts, oks)``
        with the flatten result of each part, so it is not validated twice.
        """
        from mathutils import Vector
        faces = self._faces(idx)
        n = [f.normal for f in faces]
        mean = sum(n, Vector()) / len(n)
        # principal direction of normal spread (power iteration on covariance)
        axis = Vector((1.0, 0.3, 0.1))
        for _ in range(20):
            acc = Vector()
            for v in n:
                d = v - mean
                acc += d * d.dot(axis)
            if acc.length < 1e-12:
                break
            axis = acc.normalized()
        proj = {f.index: (f.normal - mean).dot(axis) for f in faces}
        cands = [self._watershed(idx, max(proj, key=proj.get), min(proj, key=proj.get)),
                 ([fi for fi in idx if proj[fi] >= 0], [fi for fi in idx if proj[fi] < 0])]
        cands = [c for c in cands if c[0] and c[1]]
        if not cands:
            # fall back to a spatial split
            cs = [f.calc_center_median() for f in faces]
            c = sum(cs, Vector()) / len(cs)
            ext = [max(p[k] for p in cs) - min(p[k] for p in cs) for k in range(3)]
            k = ext.index(max(ext))
            cands = [([f.index for f, p in zip(faces, cs) if p[k] >= c[k]],
                      [f.index for f, p in zip(faces, cs) if p[k] < c[k]])]
        total = sum(self.area[i] for i in idx) or 1.0
        best = None
        for a, b in cands:
            parts = self._components(a) + self._components(b)
            owner = {fi: pi for pi, p in enumerate(parts) for fi in p}
            oks = [self._ok(p, limit) for p in parts]
            good = sum(self.area[fi] for p, ok in zip(parts, oks) if ok for fi in p) / total
            soft = 0.0
            for fi in idx:
                for e in self.bm.faces[fi].edges:
                    lf = e.link_faces
                    if (len(lf) == 2 and lf[0].index in owner and lf[1].index in owner
                            and owner[lf[0].index] != owner[lf[1].index]):
                        soft += e.calc_length() * max(0.0, 1.0 - self.angle[e.index] / self.s.panel_angle)
            key = (-round(good, 6), soft)
            if best is None or key < best[0]:
                best = (key, parts, oks)
        return best[1], best[2]

    def _watershed(self, idx, seed_a, seed_b):
        """Grow two regions from two seed faces at once; the face goes to
        whichever front reaches it first. Crossing an edge costs the distance
        between face centres, steeply more on creased edges, so each front
        fills its own smooth basin first and the two meet on a crease: the
        split follows the panel's feature lines instead of an arbitrary
        normal plane (as in the chart growing of Lévy et al., LSCM 2002)."""
        import heapq
        if seed_a == seed_b:
            return [], []
        fset = set(idx)
        ref = 0.25 * self.s.panel_angle
        centre = {fi: self.bm.faces[fi].calc_center_median() for fi in idx}
        label = {}
        heap = [(0.0, seed_a, 0), (0.0, seed_b, 1)]
        while heap:
            d, fi, lab = heapq.heappop(heap)
            if fi in label:
                continue
            label[fi] = lab
            for e in self.bm.faces[fi].edges:
                if not self.glued[e.index]:
                    continue
                for g in e.link_faces:
                    gi = g.index
                    if gi in fset and gi not in label:
                        w = (centre[fi] - centre[gi]).length * (1.0 + (self.angle[e.index] / ref) ** 4)
                        heapq.heappush(heap, (d + w, gi, lab))
        a = [fi for fi in idx if label.get(fi) == 0]
        b = [fi for fi in idx if label.get(fi, 1) == 1]
        return a, b

    def _components(self, idx):
        fset = set(idx)
        seen = set()
        comps = []
        for fi in idx:
            if fi in seen:
                continue
            comp = [fi]
            seen.add(fi)
            stack = [fi]
            while stack:
                g = self.bm.faces[stack.pop()]
                for e in g.edges:
                    if not self.glued[e.index]:
                        continue
                    for h in e.link_faces:
                        if h.index in fset and h.index not in seen:
                            seen.add(h.index)
                            comp.append(h.index)
                            stack.append(h.index)
            comps.append(comp)
        return comps

    def _unglue_between(self, groups):
        owner = {}
        for gi, g in enumerate(groups):
            for fi in g:
                owner[fi] = gi
        for g in groups:
            for fi in g:
                for e in self.bm.faces[fi].edges:
                    lf = e.link_faces
                    if len(lf) == 2:
                        a, b = owner.get(lf[0].index), owner.get(lf[1].index)
                        if a is not None and b is not None and a != b:
                            self.glued[e.index] = False

    def _cone_cuts(self, min_defect_deg=90.0):
        """Cut every strong cone point (cone or pyramid apex) open to its border.

        An interior vertex whose face angles add up to far less than 360
        degrees cannot lie flat in any unwrap; Blender's ABF fails on it and
        falls back to a poor solution. Each such vertex gets the cheapest
        (least visible) path to its panel's border as a seam, the cut an
        artist would make down the back of a cone.
        """
        import heapq
        bm = self.bm
        cuts = 0
        panel_of = {fi: pid for pid, idx in enumerate(self.panels) for fi in idx}
        for v in bm.verts:
            if not v.link_faces or any(len(e.link_faces) != 2 for e in v.link_edges):
                continue
            pids = {panel_of[f.index] for f in v.link_faces}
            if len(pids) != 1 or not all(self.glued[e.index] for e in v.link_edges):
                continue
            total = sum(math.degrees(l.calc_angle()) for l in v.link_loops)
            if 360.0 - total < min_defect_deg:
                continue
            fset = set(self.panels[pids.pop()])

            def is_border(x):
                return any(not self.glued[e.index] or len(e.link_faces) != 2
                           or any(g.index not in fset for g in e.link_faces)
                           for e in x.link_edges)
            dist, prev = {v.index: 0.0}, {}
            heap = [(0.0, v.index)]
            end = None
            while heap:
                d, vi = heapq.heappop(heap)
                if d > dist.get(vi, 1e30):
                    continue
                x = bm.verts[vi]
                if vi != v.index and is_border(x):
                    end = vi
                    break
                for e in x.link_edges:
                    if not self.glued[e.index]:
                        continue
                    w = e.other_vert(x).index
                    nd = d + self.cost[e.index] + 1e-6
                    if nd < dist.get(w, 1e30):
                        dist[w] = nd
                        prev[w] = (vi, e.index)
                        heapq.heappush(heap, (nd, w))
            if end is None:
                continue
            vi = end
            while vi in prev:
                vi, ei = prev[vi]
                self.glued[ei] = False
            cuts += 1
        self.stats["cone_cuts"] = cuts

    def make_panels_flat(self):
        """Ensure every panel flattens within the base stretch limit."""
        limit = self.s.stretch_steps[0]
        self._cone_cuts()
        ring_cuts = tube_ring_cuts(self.bm, self.cost)
        out = []
        work = [(p, 0, False) for p in self.panels]
        cuts = splits = 0
        while work:
            idx, depth, known_ok = work.pop()
            if known_ok or len(idx) <= 1 or self._ok(idx, limit):
                out.append(idx)
                continue
            path = self._cut_path(idx, ring_cuts)
            if path:
                for ei in path:
                    self.glued[ei] = False
                if self._ok(idx, limit):
                    cuts += 1
                    out.append(idx)
                    continue
                for ei in path:
                    self.glued[ei] = True
            if depth >= self.s.max_split_depth:
                out.append(idx)
                continue
            parts, oks = self._split(idx, limit)
            self._unglue_between(parts)
            splits += 1
            work.extend((p, depth + 1, ok) for p, ok in zip(parts, oks))
        self.panels = out
        self.stats.update(tube_cuts=cuts, splits=splits, flat_panels=len(out))

    # ------------------------------------------------------------- side bands
    def _panel_normal(self, idx):
        from mathutils import Vector
        n = sum((self.bm.faces[i].normal * self.area[i] for i in idx), Vector())
        return n.normalized() if n.length > 1e-12 else None

    def join_side_bands(self, axes=None, max_tilt=0.35, min_turn_deg=270.0):
        """Join a ring of side panels into one continuous strip.

        The four sides of a box (or the facets of a prism, bevel bands
        included) are treated like one tube panel: they are glued into a
        single strip and cut open once, on the least visible border between
        two of them. The caps are attached later by ``merge_shells``. This
        gives the artist's T / band-plus-caps layout, where a texture flows
        round the sides with a single vertical seam, instead of a zig-zag
        that leaves seams on several visible vertical edges.
        """
        from mathutils import Vector
        axes = axes or (Vector((0, 0, 1)), Vector((1, 0, 0)), Vector((0, 1, 0)))
        limit = self.s.stretch_steps[0]
        panel_of = {}
        for pid, idx in enumerate(self.panels):
            for fi in idx:
                panel_of[fi] = pid
        normals = [self._panel_normal(idx) for idx in self.panels]
        used = set()
        bands = 0
        for axis in axes:
            side = {pid for pid, n in enumerate(normals)
                    if pid not in used and n is not None and abs(n.dot(axis)) < max_tilt}
            # borders between side panels that run along the axis
            border = {}
            for e in self.bm.edges:
                lf = e.link_faces
                if len(lf) != 2 or self.glued[e.index]:
                    continue
                a, b = panel_of[lf[0].index], panel_of[lf[1].index]
                if a == b or a not in side or b not in side:
                    continue
                d = e.verts[1].co - e.verts[0].co
                if d.length < 1e-12 or abs(d.normalized().dot(axis)) < 0.7:
                    continue
                border.setdefault((min(a, b), max(a, b)), []).append(e.index)
            nbrs = {}
            for a, b in border:
                nbrs.setdefault(a, []).append(b)
                nbrs.setdefault(b, []).append(a)
            seen = set()
            for start in nbrs:
                if start in seen:
                    continue
                # connected component; a band is a simple cycle (degree 2)
                comp, stack = {start}, [start]
                while stack:
                    for q in nbrs[stack.pop()]:
                        if q not in comp:
                            comp.add(q)
                            stack.append(q)
                seen |= comp
                if len(comp) < 3 or any(len(nbrs[p]) != 2 for p in comp):
                    continue
                # walk the cycle and check it really wraps round the axis
                order = [start, nbrs[start][0]]
                while len(order) < len(comp):
                    a, b = nbrs[order[-1]]
                    order.append(a if a != order[-2] else b)
                turn = 0.0
                for i, p in enumerate(order):
                    q = order[(i + 1) % len(order)]
                    turn += math.degrees(normals[p].angle(normals[q], 0.0))
                if turn < min_turn_deg:
                    continue
                pairs = [(min(p, q), max(p, q)) for p, q in
                         ((order[i], order[(i + 1) % len(order)]) for i in range(len(order)))]
                if any(k not in border for k in pairs):
                    continue
                # cut on the least visible border, glue the others
                cut = min(pairs, key=lambda k: sum(self.cost[i] for i in border[k]))
                glue = [ei for k in pairs if k != cut for ei in border[k]]
                for ei in glue:
                    self.glued[ei] = True
                strip = [fi for p in order for fi in self.panels[p]]
                if not self._ok(strip, limit):
                    for ei in glue:
                        self.glued[ei] = False
                    continue
                used |= comp
                bands += 1
                # the strip's two end panels sit either side of the cut; the
                # caps go on the more visible one, so their remaining seams
                # (the other panels' rims) are the more hidden ones
                ends = [p for p in order if any(k == cut for k in
                        ((min(p, q), max(p, q)) for q in nbrs[p]))]
                def rim_cost(p):
                    fset = set(self.panels[p])
                    return sum(self.cost[e.index] for fi in self.panels[p]
                               for e in self.bm.faces[fi].edges
                               if e.index not in border.get(cut, ())
                               and any(g.index not in fset for g in e.link_faces)
                               and abs((e.verts[1].co - e.verts[0].co).normalized().dot(axis)) < 0.7)
                self.anchors.append(set(self.panels[max(ends, key=rim_cost)]))
                self.log.append(f"side band: {len(order)} panels round {tuple(axis)}")
        if used:
            # one panel per band (glue-connected faces of the joined panels)
            joined = self._components([fi for p in sorted(used) for fi in self.panels[p]])
            self.panels = [idx for p, idx in enumerate(self.panels) if p not in used] + joined
        self.stats["side_bands"] = bands

    # ------------------------------------------------------------------ shells
    def _thinness(self, idx):
        area = sum(self.area[i] for i in idx)
        fset = set(idx)
        per = 0.0
        for fi in idx:
            for e in self.bm.faces[fi].edges:
                inside = sum(1 for g in e.link_faces if g.index in fset)
                if inside == 1 or not self.glued[e.index]:
                    per += e.calc_length()
        return area / ((per / 2.0) ** 2 + 1e-12)

    def merge_shells(self):
        s = self.s
        shells = {i: list(p) for i, p in enumerate(self.panels)}
        shell_of = {}
        for sid, idx in shells.items():
            for fi in idx:
                shell_of[fi] = sid
        total = sum(self.area) or 1.0

        def borders(sid):
            """{other shell: [edge idx]} along unglued shared edges."""
            out = {}
            for fi in shells[sid]:
                for e in self.bm.faces[fi].edges:
                    lf = e.link_faces
                    if len(lf) != 2 or self.glued[e.index]:
                        continue
                    a, b = shell_of[lf[0].index], shell_of[lf[1].index]
                    if a == b:
                        continue
                    other = b if a == sid else a
                    out.setdefault(other, []).append(e.index)
            return out

        def components(edge_ids):
            ids = set(edge_ids)
            comps = []
            while ids:
                start = ids.pop()
                comp = [start]
                stack = [start]
                while stack:
                    e = self.bm.edges[stack.pop()]
                    for v in e.verts:
                        for y in v.link_edges:
                            if y.index in ids:
                                ids.discard(y.index)
                                comp.append(y.index)
                                stack.append(y.index)
                comps.append(comp)
            comps.sort(key=lambda c: -sum(self.bm.edges[i].calc_length() for i in c))
            return comps

        # (area, perimeter, thinness) per shell; a shell's entry is dropped
        # when it takes part in a merge (the only time its faces or inner
        # glue change), so priorities are not recomputed on every pass
        stats_cache = {}

        def shell_stats(sid):
            st = stats_cache.get(sid)
            if st is None:
                fset = set(shells[sid])
                per = 0.0
                for fi in shells[sid]:
                    for e in self.bm.faces[fi].edges:
                        inside = sum(1 for g in e.link_faces if g.index in fset)
                        if inside == 1 or not self.glued[e.index]:
                            per += e.calc_length()
                area = sum(self.area[i] for i in shells[sid])
                st = stats_cache[sid] = (area, per, area / ((per / 2.0) ** 2 + 1e-12))
            return st

        def perimeter(sid):
            return shell_stats(sid)[1]

        def hinges(comps):
            """Ever shorter contiguous pieces of each border component, centred
            on its most visible edge: halves, quarters, ... down to one edge.
            Used only to get under Max Shells: a short hinge joins two shells
            that cannot flatten glued along their whole border."""
            out = []
            for comp in comps:
                if any(ei in self.hole_edges for ei in comp):
                    continue
                order = self._ordered_path(comp)
                if not order:
                    continue
                n = len(order)
                centre = max(range(n), key=lambda i: self.cost[order[i]])
                size = n // 2
                while size >= 1:
                    lo = min(max(0, centre - size // 2), n - size)
                    piece = order[lo:lo + size]
                    if piece not in out:
                        out.append(piece)
                    size //= 2
            return out

        def relief_cuts(a, b, comps):
            """A cap that sits in a ring's hole (a pupil in an eyeball, a
            patch enclosed by its neighbour) cannot join the ring on a partial
            hinge: the free borders would cross. Cut the ring once instead,
            on the cheapest path from the shared border to its other boundary
            loop, the way a side band is opened for its caps."""
            out = []
            for comp in comps:
                cverts = {v.index for ei in comp for v in self.bm.edges[ei].verts}
                for x in (a, b):
                    loops = self._boundary_loops(shells[x])
                    others = [lp for lp in loops if not (lp & cverts)]
                    if not others:
                        continue
                    path = self._shortest_cut(shells[x], cverts, set().union(*others))
                    if path:
                        out.append(path)
            return out

        def darts(a, b, edge_ids, limit, max_darts=3):
            """Glue the whole border and relieve the stretch with darts: cuts
            from the shell's border to the inner vertex where most curvature
            is concentrated (largest angle defect), up to ``max_darts``."""
            union = shells[a] + shells[b]
            fset = set(union)
            for ei in edge_ids:
                self.glued[ei] = True
            made = []
            for _ in range(max_darts):
                border, inner = set(), []
                for fi in union:
                    for v in self.bm.faces[fi].verts:
                        if all(self.glued[e.index] and len(e.link_faces) == 2
                               and all(g.index in fset for g in e.link_faces)
                               for e in v.link_edges):
                            inner.append(v)
                        else:
                            border.add(v.index)
                if not inner or not border:
                    break
                v = max(set(inner), key=lambda x: abs(
                    360.0 - sum(math.degrees(l.calc_angle()) for l in x.link_loops)))
                path = self._shortest_cut(union, {v.index}, border)
                if not path:
                    break
                for ei in path:
                    self.glued[ei] = False
                made.extend(path)
                if attempt(a, b, [[]], limit):
                    return True
            for ei in made:
                self.glued[ei] = True
            for ei in edge_ids:
                self.glued[ei] = False
            return False

        def try_merge(a, b, edge_ids, limit, hinge=False):
            comps = components(edge_ids)
            if hinge:
                for cut in relief_cuts(a, b, comps):
                    for ei in cut:
                        self.glued[ei] = False
                    if attempt(a, b, [edge_ids] + comps, limit):
                        return True
                    for ei in cut:
                        self.glued[ei] = True
                return attempt(a, b, hinges(comps), limit) or darts(a, b, edge_ids, limit)
            options = [edge_ids] if len(comps) > 1 else []
            options += comps
            # unfold along one straight edge run (a box face folding open),
            # never along part of a hole rim, and only if the run is a real
            # share of the smaller shell's border (no one-edge flaps)
            # A cap attached this way keeps its other borders as seams, so try
            # the most visible run first: the seams that remain are the
            # hidden ones (a lid hinges on its front edge, not its back).
            min_len = 0.2 * min(perimeter(a), perimeter(b))
            runs = []
            for comp in comps:
                if any(ei in self.hole_edges for ei in comp):
                    continue
                for run in self._straight_runs(comp):
                    if len(run) == len(comp):
                        continue
                    if sum(self.bm.edges[k].calc_length() for k in run) >= min_len:
                        runs.append(run)
            # a cap goes on the strip's end panel first (both caps on the
            # same panel), then on the most visible run
            def on_anchor(run):
                return any(g.index in anc for anc in self.anchors
                           for k in run for g in self.bm.edges[k].link_faces)
            runs.sort(key=lambda r: (not on_anchor(r), -sum(self.cost[k] for k in r)))
            options += runs
            return attempt(a, b, options, limit)

        def attempt(a, b, options, limit):
            for glue in options:
                for ei in glue:
                    self.glued[ei] = True
                if self._ok(shells[a] + shells[b], limit):
                    if len(shells[a]) < len(shells[b]):
                        a, b = b, a
                    for fi in shells[b]:
                        shell_of[fi] = a
                    shells[a].extend(shells.pop(b))
                    stats_cache.pop(a, None)
                    stats_cache.pop(b, None)
                    return True
                for ei in glue:
                    self.glued[ei] = False
            return False

        def priority(a, b, edge_ids):
            """Lower = merge first.

            Tiny or thin shells (bevel bands, small trims) go first, onto the
            neighbour they share the *longest* border with, so they lie flat
            along an edge instead of sticking out like whiskers. Then the
            rest, across the most visible borders first, so the seams that
            survive are the hidden ones.
            """
            area_a, area_b = shell_stats(a)[0], shell_stats(b)[0]
            small = a if area_a <= area_b else b
            area = min(area_a, area_b)
            thin = shell_stats(small)[2]
            shared = sum(self.bm.edges[i].calc_length() for i in edge_ids)
            if area / total < 0.01 or thin < s.thin_ratio:
                return (0, -shared / (perimeter(small) + 1e-12))
            vis_cost = sum(self.cost[i] for i in edge_ids)
            return (1, -vis_cost / (area ** 0.5 + 1e-9))

        refused = set()
        for li, limit in enumerate(s.stretch_steps):
            # first step: merge freely (fewer shells is better);
            # later steps: only while above the hard limit
            while True:
                if li > 0 and len(shells) <= s.max_shells:
                    break
                cands = []
                for a in list(shells):
                    for b, eids in borders(a).items():
                        if a < b:
                            key = (a, b, len(shells[a]), len(shells[b]), limit)
                            if key not in refused:
                                cands.append((priority(a, b, eids), a, b, eids, key))
                if not cands:
                    break
                cands.sort(key=lambda c: c[0])
                merged = False
                for _p, a, b, eids, key in cands:
                    if try_merge(a, b, eids, limit):
                        merged = True
                        break
                    refused.add(key)
                if not merged:
                    break
            self.stats[f"shells_at_{limit}"] = len(shells)

        # Max Shells is a hard limit. If merges glued along whole borders
        # still fail at the stretch ceiling, join neighbours on a shorter
        # hinge (part of their border, down to one edge) instead. Every such
        # merge is still validated at the ceiling, so overlap, flips and
        # stretch past the ceiling are never traded for it; the price is
        # longer seams. Only if no hinge passes for any pair do the shells
        # stay above the limit (reported as ``max_shells_unmet``).
        ceiling = s.stretch_steps[-1]
        hinged = 0
        while len(shells) > s.max_shells:
            cands = []
            for a in list(shells):
                for b, eids in borders(a).items():
                    if a < b:
                        key = (a, b, len(shells[a]), len(shells[b]), "hinge")
                        if key not in refused:
                            cands.append((priority(a, b, eids), a, b, eids, key))
            cands.sort(key=lambda c: c[0])
            merged = False
            for _p, a, b, eids, key in cands:
                if try_merge(a, b, eids, ceiling, hinge=True):
                    merged = True
                    hinged += 1
                    break
                refused.add(key)
            if not merged:
                break
        self.stats["hinge_merges"] = hinged
        if len(shells) > s.max_shells:
            self.stats["max_shells_unmet"] = True
        self.shells = list(shells.values())
        self.stats["shells"] = len(self.shells)

    # ------------------------------------------------------------------ result
    def result(self):
        chart_of_face = [0] * len(self.bm.faces)
        for sid, idx in enumerate(self.shells):
            for fi in idx:
                chart_of_face[fi] = sid
        seams = [e.index for e in self.bm.edges if len(e.link_faces) >= 2 and not self.glued[e.index]]
        return chart_of_face, seams


def numpy_validator():
    from .flatten import flatten, quality

    def validate(faces, glued, limit):
        res = flatten(faces, glued)
        if res is None:
            return False
        dist, flipped, crossings = quality(*res)
        return dist <= limit and flipped <= 0.001 and crossings == 0
    return validate


LAST_STATS = {}
LAST_ANCHORS = []


def compute_shells(bm, visibility, settings=None, validator=None):
    """Return ``(chart_of_face, seam_edge_indices)``."""
    settings = settings or PanelSettings()
    bm.verts.index_update(); bm.edges.index_update(); bm.faces.index_update()
    solver = PanelSolver(bm, visibility, settings, validator or numpy_validator())
    solver.build_panels()
    solver.make_panels_flat()
    solver.join_side_bands()
    solver.merge_shells()
    LAST_STATS.clear()
    LAST_STATS.update(solver.stats)
    LAST_ANCHORS[:] = [sorted(a) for a in solver.anchors]
    return solver.result()
