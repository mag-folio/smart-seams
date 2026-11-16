"""Seam-cost helpers shared by the solver.

* ``seam_cost``: how noticeable a seam on an edge would be (visibility x
  length, cheaper on sharp or concave edges where shading hides it).
* ``edge_loops``: group edges into edge loops (straight through valence-4
  vertices).
* ``loop_smoothed_costs``: blend each edge's cost with its loop's mean, so
  stacked rings agree on one cut column.
* ``tube_ring_cuts``: one aligned, least-visible cut per tube-like quad ring.
"""

import math


def _edge_angle(e):
    return math.degrees(e.calc_face_angle(0.0))


def _is_concave(e):
    try:
        return _edge_angle(e) > 1.0 and not e.is_convex
    except (AttributeError, ValueError):
        return False


def seam_cost(e, visibility, settings):
    """How noticeable a seam on this edge would be (higher = worse)."""
    faces = e.link_faces
    vis = max(visibility[f.index] for f in faces) if faces else 0.0
    cost = e.calc_length() * (0.05 + vis)
    if _edge_angle(e) >= 45.0:
        cost *= settings.sharp_bonus
    if _is_concave(e):
        cost *= settings.concave_bonus
    return cost


def edge_loops(bm):
    """Group edges into edge loops (walk straight through valence-4 vertices).

    Returns a list ``loop_id`` per edge index.
    """
    parent = list(range(len(bm.edges)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for v in bm.verts:
        if len(v.link_edges) != 4 or len(v.link_faces) != 4 or not v.is_manifold:
            continue
        edges = list(v.link_edges)
        for e in edges:
            fe = set(f.index for f in e.link_faces)
            opp = [x for x in edges if x is not e and not (fe & set(f.index for f in x.link_faces))]
            if len(opp) == 1:
                a, b = find(e.index), find(opp[0].index)
                parent[a] = b
    return [find(i) for i in range(len(bm.edges))]


def tube_ring_cuts(bm, cost, min_turn_deg=270.0):
    """Choose one cut per tube-like quad ring, aligned across stacked rings.

    A face ring is a closed strip of quads walked across opposite edges (a
    cylinder wall, the four sides of a leg, a bevel band). If the faces'
    normals turn by >= ``min_turn_deg`` going round, it is a tube and needs
    exactly one cut. The edges crossing a ring ("rungs") of stacked rings lie
    on shared edge loops, so cutting along the cheapest *edge loop* cuts all
    stacked rings in the same column - a single straight seam instead of a
    staircase.

    Returns the set of edge indices that must stay seams.
    """
    bm.faces.ensure_lookup_table()
    loop_of = edge_loops(bm)
    rings = []
    seen_pairs = set()
    for f in bm.faces:
        if len(f.verts) != 4:
            continue
        for start in (0, 1):
            key = (f.index, start)
            if key in seen_pairs:
                continue
            # walk across edge `start` and its opposite
            rung_edges, faces = [], []
            loop = f.loops[start]
            cur_f = f
            ok = True
            for _ in range(len(bm.faces) + 1):
                faces.append(cur_f)
                e = loop.edge
                rung_edges.append(e)
                if len(e.link_faces) != 2:
                    ok = False
                    break
                nxt = loop.link_loop_radial_next
                nf = nxt.face
                if len(nf.verts) != 4:
                    ok = False
                    break
                loop = nxt.link_loop_next.link_loop_next
                cur_f = nf
                if nf is f:
                    break
            else:
                ok = False
            if not ok or cur_f is not f or len(faces) < 3:
                continue
            for g, e in zip(faces, rung_edges):
                idx = [l.edge for l in g.loops].index(e)
                seen_pairs.add((g.index, idx % 2))
            turn = sum(math.degrees(e.calc_face_angle(0.0)) for e in rung_edges)
            if turn >= min_turn_deg:
                rings.append(rung_edges)

    if not rings:
        return set()
    # score each edge loop by mean seam cost per length of its rungs
    score = {}
    for ring in rings:
        for e in ring:
            lid = loop_of[e.index]
            s, n = score.get(lid, (0.0, 0))
            score[lid] = (s + cost[e.index] / (e.calc_length() or 1e-9), n + 1)
    ring_loops = [{loop_of[e.index]: e for e in ring} for ring in rings]
    return _pick_ring_cuts(ring_loops, score)


def _pick_ring_cuts(ring_loops, score):
    """Greedy: cut along the edge loop with the lowest mean cost divided by the
    number of still-uncut rings it crosses, until every ring is cut.

    ``hits`` (uncut rings per loop) is kept up to date as rings are cut, and
    the cheapest loop comes from a heap, so a dense mesh with thousands of
    rings takes moments. (Recounting ``hits`` for every candidate on every
    pass made a 27k-face mesh take many minutes.) Same choices as
    ``_pick_ring_cuts_reference``.
    """
    import heapq
    rings_of, first = {}, {}
    for ri, rl in enumerate(ring_loops):
        for pos, lid in enumerate(rl):
            rings_of.setdefault(lid, []).append(ri)
            first.setdefault(lid, (ri, pos))
    hits = {lid: len(rs) for lid, rs in rings_of.items()}

    def key(lid):
        s, n = score[lid]
        return (s / n) / hits[lid]

    heap = [(key(lid), first[lid], lid) for lid in hits]
    heapq.heapify(heap)
    cut = [False] * len(ring_loops)
    left = len(ring_loops)
    forced = set()
    def valid(entry):
        k, _, lid = entry
        return hits[lid] > 0 and k == key(lid)

    def first_uncut(lid):
        """Where the original scan meets ``lid`` first: (ring, position)."""
        for ri in rings_of[lid]:
            if not cut[ri]:
                return ri, list(ring_loops[ri]).index(lid)
        return (len(ring_loops), 0)

    while left and heap:
        top = heapq.heappop(heap)
        if not valid(top):
            continue                      # stale entry
        # exact ties: the original takes the one it meets first
        tied = [top]
        while heap and heap[0][0] == top[0]:
            e = heapq.heappop(heap)
            if valid(e) and all(e[2] != t[2] for t in tied):
                tied.append(e)
        if len(tied) > 1:
            tied.sort(key=lambda e: first_uncut(e[2]))
            for e in tied[1:]:
                heapq.heappush(heap, e)
        lid = tied[0][2]
        for ri in rings_of[lid]:
            if cut[ri]:
                continue
            cut[ri] = True
            left -= 1
            forced.add(ring_loops[ri][lid].index)
            for other in ring_loops[ri]:
                hits[other] -= 1
                if hits[other] > 0:
                    heapq.heappush(heap, (key(other), first[other], other))
    return forced


def _pick_ring_cuts_reference(ring_loops, score):
    """The original greedy (recounts every candidate each pass), for tests."""
    forced = set()
    uncut = list(range(len(ring_loops)))
    while uncut:
        # cheapest loop among the rungs of any uncut ring; prefer loops that
        # cut many rings at once (stacks)
        best = None
        for ri in uncut:
            for lid in ring_loops[ri]:
                s, n = score[lid]
                hits = sum(1 for rj in uncut if lid in ring_loops[rj])
                key = (s / n) / hits
                if best is None or key < best[0]:
                    best = (key, lid)
        lid = best[1]
        still = []
        for ri in uncut:
            if lid in ring_loops[ri]:
                forced.add(ring_loops[ri][lid].index)
            else:
                still.append(ri)
        uncut = still
    return forced


def _opposite(e, f):
    for l in f.loops:
        if l.edge is e:
            return l.link_loop_next.link_loop_next.edge
    return None


def _strip_width(e, f):
    """Width of quad ``f`` measured across ``e`` (distance to the opposite edge)."""
    o = _opposite(e, f)
    mean = 0.5 * (e.calc_length() + o.calc_length())
    return f.calc_area() / mean if mean > 1e-12 else 0.0


def bevel_crease_edges(bm, panel_angle, angles=None, min_angle=None, ratio=3.0, max_faces=8):
    """Corner lines of rounded (bevelled) edges.

    A bevel turns a sharp corner into a narrow strip of faces whose edges each
    bend by only part of the corner angle, so no single edge reaches
    ``panel_angle`` and panels would flood straight round the corner. Here a
    *bevel band* is found from a creased seed edge (>= ``min_angle``) by
    walking across narrow quads on both sides until a face at least
    ``ratio`` times wider (or a non-quad / open border); the crossed edges
    form the band, which counts if its total turn reaches ``panel_angle``. Closed rings (cylinder walls, hole walls) are never
    bands. Each band contributes one edge, its sharpest, as the corner line;
    the choice is made per edge loop so a long bevel gets one straight line
    instead of switching between its parallel edges.
    """
    if angles is None:
        angles = [math.degrees(e.calc_face_angle(0.0)) if len(e.link_faces) == 2 else 180.0
                  for e in bm.edges]
    if min_angle is None:
        min_angle = 0.25 * panel_angle

    def creased(e):
        return len(e.link_faces) == 2 and min_angle <= angles[e.index] < panel_angle

    def side(seed, f, ref):
        """Walk across narrow quads away from ``seed`` until a much wider
        face (the flank). Returns the crossed edges, or None if the strip
        closes into a ring or never reaches a flank."""
        edges, widths = [], []
        cur, face = seed, f
        for _ in range(max_faces):
            if len(face.verts) != 4:
                return edges                      # junction: accept as flank
            w = _strip_width(cur, face)
            if w > ratio * max(widths + [ref]):
                return edges                      # wide flank face
            nxt = _opposite(cur, face)
            if nxt is seed:
                return None                       # closed ring: a tube, not a bevel
            if len(nxt.link_faces) != 2:
                return edges                      # open border
            edges.append(nxt)
            widths.append(w)
            cur = nxt
            face = nxt.link_faces[0] if nxt.link_faces[1] is face else nxt.link_faces[1]
        return None

    bands = []
    seen = set()
    for e in bm.edges:
        if e.index in seen or not creased(e):
            continue
        f0, f1 = e.link_faces
        if len(f0.verts) != 4 and len(f1.verts) != 4:
            continue
        ws = [_strip_width(e, f) for f in (f0, f1) if len(f.verts) == 4]
        ref = min(ws)
        if ref <= 1e-12:
            continue
        a = side(e, f0, ref)
        b = side(e, f1, ref)
        if a is None or b is None:
            continue
        band = list(reversed(a)) + [e] + b
        if len(band) < 2:
            continue
        seen.update(x.index for x in band if creased(x))
        # an edge of the band that is already sharp enough is the corner line
        if max(angles[x.index] for x in band) >= panel_angle:
            continue
        if sum(angles[x.index] for x in band) < panel_angle:
            continue
        bands.append([x.index for x in band])

    if not bands:
        return set()
    loop_of = edge_loops(bm)
    tot, cnt = {}, {}
    for band in bands:
        for ei in band:
            lid = loop_of[ei]
            tot[lid] = tot.get(lid, 0.0) + angles[ei]
            cnt[lid] = cnt.get(lid, 0) + 1
    out = set()
    for band in bands:
        mid = 0.5 * (len(band) - 1)
        best = max(range(len(band)), key=lambda i: (
            round(tot[loop_of[band[i]]] / cnt[loop_of[band[i]]], 1),
            angles[band[i]], -abs(i - mid)))
        out.add(band[best])
    return out


def complete_feature_lines(bm, strong, angles, low, min_dot=0.85, max_gap=4):
    """Close short gaps in feature lines (hysteresis, as in Canny).

    A corner line is often a little softer at its ends (a foot, where two
    bevels meet) and dips under the panel angle for an edge or two. Such a
    gap lets the panel flood leak round the corner, so the whole line stops
    being a border. From each loose end of a feature line, walk on along the
    straightest next edge while it still bends by at least ``low`` degrees,
    for at most ``max_gap`` edges. The walk is kept only if it reaches
    another feature line or an open border, i.e. it actually closes a gap;
    a line that just fades out on a smooth surface is left alone.
    Returns the added edge indices.
    """
    def on_border(v, skip):
        return any(x.index != skip and (x.index in strong or len(x.link_faces) != 2)
                   for x in v.link_edges)

    def straightest(e, v):
        d = v.co - e.other_vert(v).co
        if d.length < 1e-12:
            return None
        d.normalize()
        best, best_dot = None, min_dot
        for x in v.link_edges:
            if x is e:
                continue
            dx = x.other_vert(v).co - v.co
            if dx.length < 1e-12:
                continue
            dot = d.dot(dx.normalized())
            if dot > best_dot:
                best, best_dot = x, dot
        return best

    added = set()
    for ei in strong:
        e = bm.edges[ei]
        for v in e.verts:
            # loose end: no other feature edge or border at this vertex
            if on_border(v, ei):
                continue
            chain, cur, at = [], e, v
            for _ in range(max_gap):
                nxt = straightest(cur, at)
                if (nxt is None or nxt.index in strong or len(nxt.link_faces) != 2
                        or angles[nxt.index] < low):
                    chain = None
                    break
                chain.append(nxt.index)
                at = nxt.other_vert(at)
                cur = nxt
                if on_border(at, nxt.index):
                    break
            else:
                chain = None
            if chain:
                added.update(chain)
    return added


def loop_smoothed_costs(bm, visibility, settings):
    """Seam cost per edge, blended with the mean cost of its edge loop.

    Without this, each ring of a stack (a hole wall with bevels, a leg with
    several segments) picks its own cheapest cut edge and the seam
    staircases between columns. Sharing the loop's cost makes the rings agree
    on one column, so the cut follows a single straight edge loop.
    """
    raw = [seam_cost(e, visibility, settings) for e in bm.edges]
    lengths = [e.calc_length() or 1e-9 for e in bm.edges]
    loop_of = edge_loops(bm)
    tot, ln = {}, {}
    for i, lid in enumerate(loop_of):
        tot[lid] = tot.get(lid, 0.0) + raw[i] / lengths[i]
        ln[lid] = ln.get(lid, 0) + 1
    w = settings.loop_smoothing
    return [(1.0 - w) * raw[i] + w * lengths[i] * tot[loop_of[i]] / ln[loop_of[i]]
            for i in range(len(raw))]
