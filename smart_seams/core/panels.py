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

from .seams import loop_smoothed_costs, tube_ring_cuts


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
                    if len(e.link_faces) != 2 or self.angle[e.index] >= s.panel_angle:
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

    def _split(self, idx):
        """Split a face set in two along its dominant normal direction."""
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
        a = [f.index for f in faces if (f.normal - mean).dot(axis) >= 0]
        b = [f.index for f in faces if (f.normal - mean).dot(axis) < 0]
        if not a or not b:
            # fall back to a spatial split
            cs = [f.calc_center_median() for f in faces]
            c = sum(cs, Vector()) / len(cs)
            ext = [max(p[k] for p in cs) - min(p[k] for p in cs) for k in range(3)]
            k = ext.index(max(ext))
            a = [f.index for f, p in zip(faces, cs) if p[k] >= c[k]]
            b = [f.index for f, p in zip(faces, cs) if p[k] < c[k]]
        out = []
        for part in (a, b):
            out.extend(self._components(part))
        return out

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

    def make_panels_flat(self):
        """Ensure every panel flattens within the base stretch limit."""
        limit = self.s.stretch_steps[0]
        ring_cuts = tube_ring_cuts(self.bm, self.cost)
        out = []
        work = [(p, 0) for p in self.panels]
        cuts = splits = 0
        while work:
            idx, depth = work.pop()
            if len(idx) <= 1 or self._ok(idx, limit):
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
            parts = self._split(idx)
            self._unglue_between(parts)
            splits += 1
            work.extend((p, depth + 1) for p in parts)
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

        def perimeter(sid):
            fset = set(shells[sid])
            per = 0.0
            for fi in shells[sid]:
                for e in self.bm.faces[fi].edges:
                    inside = sum(1 for g in e.link_faces if g.index in fset)
                    if inside == 1 or not self.glued[e.index]:
                        per += e.calc_length()
            return per

        def try_merge(a, b, edge_ids, limit):
            comps = components(edge_ids)
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
            for glue in options:
                for ei in glue:
                    self.glued[ei] = True
                if self._ok(shells[a] + shells[b], limit):
                    if len(shells[a]) < len(shells[b]):
                        a, b = b, a
                    for fi in shells[b]:
                        shell_of[fi] = a
                    shells[a].extend(shells.pop(b))
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
            area_a = sum(self.area[i] for i in shells[a])
            area_b = sum(self.area[i] for i in shells[b])
            small = a if area_a <= area_b else b
            area = min(area_a, area_b)
            thin = self._thinness(shells[small])
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
