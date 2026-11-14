"""UV quality metrics, so 'good' vs 'bad' unwraps can be compared with numbers.

All functions take a BMesh with an active UV layer (``analyze`` also accepts
``None`` and then reports an empty result). Face normals must be up to date
(they are for meshes read with ``from_mesh`` / ``from_edit_mesh``).

Definitions (see ``tests/test_metrics.py`` for known-answer checks):

- *area distortion* of a face is |log(r)| with r = (UV area / 3D area) divided
  by the same ratio for the whole shell. A uniformly scaled shell therefore has
  zero distortion (texel density between shells is a layout choice, not
  stretch); a shell stretched uniformly in one direction has zero *area*
  distortion too and shows up only in the angle error.
- *flipped faces* are faces whose UV winding is opposite to the majority of
  their shell (folds). A whole shell with reversed winding is not a fold; it is
  counted separately in ``mirrored_islands``.
- *overlap* is rasterised (``uv_overlap_pct``, % of covered texels covered by
  more than one face, so self-overlap inside one shell counts) and checked
  exactly (``uv_boundary_crossings``, number of properly crossing UV boundary
  segment pairs, which also catches overlaps thinner than a texel).
"""

import math

import numpy as np

# |log| distortion given to a face/shell whose UV area has collapsed to zero
COLLAPSED = abs(math.log(1e-6))


def _loop_pair_uvs(a, b, uv_layer):
    """UVs of edge loop ``a`` and of the opposite loop ``b``, in the same vertex order."""
    a1, a2 = a[uv_layer].uv, a.link_loop_next[uv_layer].uv
    if b.vert is a.vert:  # same direction (flipped normals)
        b1, b2 = b[uv_layer].uv, b.link_loop_next[uv_layer].uv
    else:
        b1, b2 = b.link_loop_next[uv_layer].uv, b[uv_layer].uv
    return a1, a2, b1, b2


def uv_islands(bm, uv_layer, tol=1e-5):
    """Faces grouped into UV islands (connected through shared UV coordinates)."""
    bm.faces.ensure_lookup_table()
    seen = set()
    islands = []
    for f in bm.faces:
        if f.index in seen:
            continue
        island = []
        stack = [f]
        seen.add(f.index)
        while stack:
            g = stack.pop()
            island.append(g)
            for loop in g.loops:
                for other in loop.edge.link_loops:
                    h = other.face
                    if h.index in seen or h is g:
                        continue
                    a1, a2, b1, b2 = _loop_pair_uvs(loop, other, uv_layer)
                    if (a1 - b1).length < tol and (a2 - b2).length < tol:
                        seen.add(h.index)
                        stack.append(h)
        islands.append(island)
    return islands


def _uv_area(f, uv_layer):
    uvs = [l[uv_layer].uv for l in f.loops]
    a = 0.0
    for i in range(len(uvs)):
        x1, y1 = uvs[i]
        x2, y2 = uvs[(i + 1) % len(uvs)]
        a += x1 * y2 - x2 * y1
    return a * 0.5  # signed


def _uv_corner_angles(f, uv_layer):
    loops = list(f.loops)
    out = []
    n = len(loops)
    for i in range(n):
        p = loops[i][uv_layer].uv
        a = loops[i - 1][uv_layer].uv - p
        b = loops[(i + 1) % n][uv_layer].uv - p
        if a.length < 1e-12 or b.length < 1e-12:
            out.append(0.0)
        else:
            out.append(a.angle(b, 0.0))
    return out


def seam_components(seam_edges):
    """How many separate seam networks there are (1-2 per island is ideal)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in seam_edges:
        a, b = (find(v.index) for v in e.verts)
        parent[a] = b
    return len({find(v.index) for e in seam_edges for v in e.verts})


def seam_dead_ends(seam_edges, border_verts=()):
    """Seam vertices with only one seam edge that are not on the mesh border.

    A seam that runs into an open border is not dangling, so border vertices
    are ignored. On a closed surface the cut of one shell is a tree, and a tree
    always has at least two leaves: a single straight cut reports 2, and each
    extra branch adds one. So this measures how branched the cuts are; it is
    not a count of mistakes.
    """
    deg = {}
    for e in seam_edges:
        for v in e.verts:
            deg[v.index] = deg.get(v.index, 0) + 1
    border = set(border_verts)
    return sum(1 for vi, d in deg.items() if d == 1 and vi not in border)


def _boundary_loops(bm, uv_layer, tol=1e-5):
    """Loops whose edge is a UV boundary (mesh border, non-manifold or UV seam)."""
    out = []
    for e in bm.edges:
        ll = list(e.link_loops)
        if len(ll) != 2:
            out.extend(ll)
            continue
        a, b = ll
        a1, a2, b1, b2 = _loop_pair_uvs(a, b, uv_layer)
        if (a1 - b1).length > tol or (a2 - b2).length > tol:
            out.extend(ll)
    return out


def _raster_grid(bm, uv_layer, res):
    """Per-texel count of faces covering it.

    Scanline rasterisation of each UV polygon (even-odd rule, so concave n-gons
    are exact). Texel centres are sampled with half-open rules and crossings
    are computed from canonically ordered edge endpoints, so a texel centre
    lying exactly on an edge shared by two faces is given to exactly one of
    them. The grid covers the unit square at ``res`` texels per UV unit; UVs
    outside it widen the grid, and very large UV ranges coarsen it so the grid
    never exceeds ``4 * res`` texels a side.
    """
    polys = []
    lo_x = lo_y = 0.0
    hi_x = hi_y = 1.0
    for f in bm.faces:
        pts = [(l[uv_layer].uv.x, l[uv_layer].uv.y) for l in f.loops]
        polys.append(pts)
        for x, y in pts:
            lo_x, hi_x = min(lo_x, x), max(hi_x, x)
            lo_y, hi_y = min(lo_y, y), max(hi_y, y)
    if not polys:
        return None
    span = max(hi_x - lo_x, hi_y - lo_y)
    scale = float(res)
    if span * scale > 4 * res:
        scale = 4 * res / span
    w = int(math.ceil((hi_x - lo_x) * scale)) + 1
    h = int(math.ceil((hi_y - lo_y) * scale)) + 1
    grid = np.zeros((h, w), dtype=np.int32)
    for pts in polys:
        P = [((x - lo_x) * scale, (y - lo_y) * scale) for x, y in pts]
        edges = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            if a[1] == b[1]:
                continue  # horizontal edges never cross a scanline (half-open rule)
            if (a[1], a[0]) > (b[1], b[0]):
                a, b = b, a
            edges.append((a[1], b[1], a[0], (b[0] - a[0]) / (b[1] - a[1])))
        if not edges:
            continue
        y0 = max(0, int(math.ceil(min(e[0] for e in edges) - 0.5)))
        y1 = min(h, int(math.ceil(max(e[1] for e in edges) - 0.5)))
        for row in range(y0, y1):
            yc = row + 0.5
            xs = sorted(ax + (yc - ay) * k for ay, by, ax, k in edges if ay <= yc < by)
            for i in range(0, len(xs) - 1, 2):
                c0 = max(0, int(math.ceil(xs[i] - 0.5)))
                c1 = min(w, int(math.ceil(xs[i + 1] - 0.5)))
                if c1 > c0:
                    grid[row, c0:c1] += 1
    return grid


def overlap_pct(bm, uv_layer, res=512):
    """% of covered UV texels that are covered more than once (0 = no overlap).

    Overlaps between faces of the same shell count too. Any overlapping texel
    makes the result at least 0.001, so a real overlap never rounds to 0.
    Overlaps thinner than a texel can be missed: use ``boundary_crossings``.
    """
    grid = _raster_grid(bm, uv_layer, res)
    if grid is None:
        return 0.0
    covered = int(np.count_nonzero(grid))
    if covered == 0:
        return 0.0
    multi = int(np.count_nonzero(grid > 1))
    if multi == 0:
        return 0.0
    return max(round(100.0 * multi / covered, 3), 0.001)


def boundary_crossings(bm, uv_layer):
    """Exact count of UV boundary segment pairs that properly cross each other.

    Boundary segments are taken from every shell (seams, mesh borders), so this
    sees overlaps between shells and inside one shell, however thin. Segments
    that only touch (shared end points, T-junctions, collinear contact) do not
    count. A shell lying entirely inside another without crossing is not
    detected here (the raster ``overlap_pct`` catches that).
    """
    loops = _boundary_loops(bm, uv_layer)
    if len(loops) < 2:
        return 0
    P = np.array([(l[uv_layer].uv.x, l[uv_layer].uv.y) for l in loops], dtype=float)
    Q = np.array([(l.link_loop_next[uv_layer].uv.x, l.link_loop_next[uv_layer].uv.y) for l in loops],
                 dtype=float)
    D = Q - P
    L = np.hypot(D[:, 0], D[:, 1])
    keep = L > 0
    P, Q, D, L = P[keep], Q[keep], D[keep], L[keep]
    if len(P) < 2:
        return 0
    allp = np.vstack([P, Q])
    span = float(np.ptp(allp, axis=0).max()) or 1.0
    eps = span * 1e-6  # distance tolerance (UVs are stored as float32)
    xmin = np.minimum(P[:, 0], Q[:, 0])
    xmax = np.maximum(P[:, 0], Q[:, 0])
    order = np.argsort(xmin, kind="stable")
    xs_sorted = xmin[order]
    count = 0
    for pos, i in enumerate(order):
        end = int(np.searchsorted(xs_sorted, xmax[i] + eps, side="right"))
        if end <= pos + 1:
            continue
        j = order[pos + 1:end]
        # side of segment j's end points relative to line i, and vice versa (as distances)
        o1 = (D[i, 0] * (P[j, 1] - P[i, 1]) - D[i, 1] * (P[j, 0] - P[i, 0])) / L[i]
        o2 = (D[i, 0] * (Q[j, 1] - P[i, 1]) - D[i, 1] * (Q[j, 0] - P[i, 0])) / L[i]
        o3 = (D[j, 0] * (P[i, 1] - P[j, 1]) - D[j, 1] * (P[i, 0] - P[j, 0])) / L[j]
        o4 = (D[j, 0] * (Q[i, 1] - P[j, 1]) - D[j, 1] * (Q[i, 0] - P[j, 0])) / L[j]
        straddle_i = ((o1 > eps) & (o2 < -eps)) | ((o1 < -eps) & (o2 > eps))
        straddle_j = ((o3 > eps) & (o4 < -eps)) | ((o3 < -eps) & (o4 > eps))
        count += int(np.count_nonzero(straddle_i & straddle_j))
    return count


def _empty_result():
    return {
        "islands": 0,
        "seam_components": 0,
        "seam_dead_ends": 0,
        "uv_overlap_pct": 0.0,
        "uv_boundary_crossings": 0,
        "worst_island_area_distortion": 0.0,
        "seam_edges": 0,
        "seam_length": 0.0,
        "seam_length_pct": 0.0,
        "seam_on_hard_edges_pct": 0.0,
        "seam_visibility": None,
        "area_distortion": 0.0,
        "angle_distortion_deg": 0.0,
        "flipped_faces": 0,
        "mirrored_islands": 0,
        "uv_coverage_pct": 0.0,
    }


def analyze(bm, uv_layer, visibility=None, matrix_world=None, hard_angle=30.0):
    """Return a dict of metrics (all keys present even with no UV layer / no faces)."""
    if uv_layer is None or len(bm.faces) == 0:
        return _empty_result()
    bm.faces.ensure_lookup_table()
    islands = uv_islands(bm, uv_layer)

    # a seam is any edge whose two sides have different UVs (island borders
    # *and* cuts inside an island)
    seam_edges = []
    for e in bm.edges:
        ll = e.link_loops
        if len(ll) != 2:
            continue
        a1, a2, b1, b2 = _loop_pair_uvs(ll[0], ll[1], uv_layer)
        if (a1 - b1).length > 1e-5 or (a2 - b2).length > 1e-5:
            seam_edges.append(e)
    border_verts = {v.index for e in bm.edges if len(e.link_faces) < 2 for v in e.verts}
    seam_len = sum(e.calc_length() for e in seam_edges)
    total_edge_len = sum(e.calc_length() for e in bm.edges) or 1.0
    on_hard = sum(e.calc_length() for e in seam_edges
                  if math.degrees(e.calc_face_angle(0.0)) >= hard_angle)

    vis_cost = None
    if visibility is not None:
        vis_cost = sum(e.calc_length() * max(visibility[f.index] for f in e.link_faces) for e in seam_edges)
        vis_cost /= (seam_len or 1.0)

    # distortion per island: area ratio spread and angle error
    area_err = 0.0
    ang_err = 0.0
    total_area = 0.0
    flipped = 0
    mirrored = 0
    uv_total = 0.0
    worst = 0.0
    for isl in islands:
        a3f = [f.calc_area() for f in isl]
        signs = [_uv_area(f, uv_layer) for f in isl]
        a3 = sum(a3f)
        a2 = sum(abs(s) for s in signs)
        uv_total += a2
        if a3 <= 0:
            continue  # nothing to measure on a zero-area shell
        collapsed = a2 <= 1e-12 * a3
        k = a2 / a3
        signed_sum = sum(signs)
        major = 1.0 if signed_sum >= 0 else -1.0
        if not collapsed and signed_sum < 0:
            mirrored += 1
        isl_err = 0.0
        for f, s, fa in zip(isl, signs, a3f):
            if fa <= 0:
                continue
            if not collapsed and s * major < 0:
                flipped += 1
            if collapsed:
                e = COLLAPSED
            else:
                r = abs(s) / (fa * k)
                e = abs(math.log(max(r, 1e-6)))
            isl_err += fa * e
            ang3 = [l.calc_angle() for l in f.loops]
            ang2 = _uv_corner_angles(f, uv_layer)
            ang_err += fa * sum(abs(x - y) for x, y in zip(ang3, ang2)) / len(ang3)
            total_area += fa
        area_err += isl_err
        # worst island area distortion (stretch is about the worst island, not the mean)
        worst = max(worst, isl_err / a3)

    return {
        "islands": len(islands),
        "seam_components": seam_components(seam_edges),
        "seam_dead_ends": seam_dead_ends(seam_edges, border_verts),
        "uv_overlap_pct": overlap_pct(bm, uv_layer),
        "uv_boundary_crossings": boundary_crossings(bm, uv_layer),
        "worst_island_area_distortion": round(worst, 4),
        "seam_edges": len(seam_edges),
        "seam_length": round(seam_len, 4),
        "seam_length_pct": round(100.0 * seam_len / total_edge_len, 2),
        "seam_on_hard_edges_pct": round(100.0 * on_hard / (seam_len or 1.0), 1),
        "seam_visibility": None if vis_cost is None else round(vis_cost, 3),
        "area_distortion": round(area_err / (total_area or 1.0), 4),
        "angle_distortion_deg": round(math.degrees(ang_err / (total_area or 1.0)), 2),
        "flipped_faces": flipped,
        "mirrored_islands": mirrored,
        # sum of shell UV areas as % of the 0-1 square (overlapping area counts twice)
        "uv_coverage_pct": round(100.0 * uv_total, 1),
    }
