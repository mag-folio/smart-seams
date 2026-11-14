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
    forced = set()
    uncut = list(range(len(rings)))
    ring_loops = [{loop_of[e.index]: e for e in ring} for ring in rings]
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
