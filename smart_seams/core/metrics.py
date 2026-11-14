"""UV quality metrics, so 'good' vs 'bad' unwraps can be compared with numbers.

All functions take a BMesh with an active UV layer.
"""

import math


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
                    a1, a2 = loop[uv_layer].uv, loop.link_loop_next[uv_layer].uv
                    if other.vert is loop.vert:  # same direction (flipped normals)
                        b1, b2 = other[uv_layer].uv, other.link_loop_next[uv_layer].uv
                    else:
                        b1, b2 = other.link_loop_next[uv_layer].uv, other[uv_layer].uv
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


def seam_dead_ends(seam_edges):
    """Seam vertices with only one seam edge (relief cuts / dangling seams)."""
    deg = {}
    for e in seam_edges:
        for v in e.verts:
            deg[v.index] = deg.get(v.index, 0) + 1
    return sum(1 for d in deg.values() if d == 1)


def overlap_pct(bm, uv_layer, res=512):
    """% of covered UV texels that are covered more than once (0 = no overlap)."""
    hits = {}
    for f in bm.faces:
        uvs = [l[uv_layer].uv for l in f.loops]
        for i in range(1, len(uvs) - 1):
            tri = (uvs[0], uvs[i], uvs[i + 1])
            xs = [p.x * res for p in tri]
            ys = [p.y * res for p in tri]
            x0, x1 = int(math.floor(min(xs))), int(math.ceil(max(xs)))
            y0, y1 = int(math.floor(min(ys))), int(math.ceil(max(ys)))
            (ax, ay), (bx, by), (cx, cy) = zip(xs, ys)
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            for y in range(y0, y1):
                py = y + 0.5
                for x in range(x0, x1):
                    px = x + 0.5
                    w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
                    w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
                    if w1 >= 0 and w2 >= 0 and w1 + w2 <= 1:
                        key = (x, y)
                        s = hits.setdefault(key, set())
                        s.add(f.index)
    if not hits:
        return 0.0
    multi = sum(1 for s in hits.values() if len(s) > 1)
    return round(100.0 * multi / len(hits), 3)


def analyze(bm, uv_layer, visibility=None, matrix_world=None):
    """Return a dict of metrics."""
    islands = uv_islands(bm, uv_layer)
    island_of = {}
    for i, isl in enumerate(islands):
        for f in isl:
            island_of[f.index] = i

    # a seam is any edge whose two sides have different UVs (island borders
    # *and* cuts inside an island)
    seam_edges = []
    for e in bm.edges:
        ll = e.link_loops
        if len(ll) != 2:
            continue
        a, b = ll
        a1, a2 = a[uv_layer].uv, a.link_loop_next[uv_layer].uv
        if b.vert is a.vert:
            b1, b2 = b[uv_layer].uv, b.link_loop_next[uv_layer].uv
        else:
            b1, b2 = b.link_loop_next[uv_layer].uv, b[uv_layer].uv
        if (a1 - b1).length > 1e-5 or (a2 - b2).length > 1e-5:
            seam_edges.append(e)
    seam_len = sum(e.calc_length() for e in seam_edges)
    total_edge_len = sum(e.calc_length() for e in bm.edges) or 1.0
    on_hard = sum(e.calc_length() for e in seam_edges if math.degrees(e.calc_face_angle(0.0)) >= 30.0)

    vis_cost = None
    if visibility is not None:
        vis_cost = sum(e.calc_length() * max(visibility[f.index] for f in e.link_faces) for e in seam_edges)
        vis_cost /= (seam_len or 1.0)

    # distortion per island: area ratio spread and angle error
    area_err = 0.0
    ang_err = 0.0
    total_area = 0.0
    flipped = 0
    uv_total = 0.0
    for isl in islands:
        a3 = sum(f.calc_area() for f in isl)
        a2 = sum(abs(_uv_area(f, uv_layer)) for f in isl)
        uv_total += a2
        if a3 <= 0 or a2 <= 0:
            continue
        k = a2 / a3
        signs = [_uv_area(f, uv_layer) for f in isl]
        major = 1.0 if sum(signs) >= 0 else -1.0
        for f, s in zip(isl, signs):
            fa = f.calc_area()
            if fa <= 0:
                continue
            if s * major < 0:
                flipped += 1
            r = abs(s) / (fa * k)
            area_err += fa * abs(math.log(max(r, 1e-6)))
            ang3 = [l.calc_angle() for l in f.loops]
            ang2 = _uv_corner_angles(f, uv_layer)
            ang_err += fa * sum(abs(x - y) for x, y in zip(ang3, ang2)) / len(ang3)
            total_area += fa

    # worst island area distortion (stretch is about the worst island, not the mean)
    worst = 0.0
    for isl in islands:
        a3 = sum(f.calc_area() for f in isl)
        a2 = sum(abs(_uv_area(f, uv_layer)) for f in isl)
        if a3 <= 0 or a2 <= 0:
            continue
        k = a2 / a3
        err = sum(f.calc_area() * abs(math.log(max(abs(_uv_area(f, uv_layer)) / (f.calc_area() * k), 1e-6)))
                  for f in isl if f.calc_area() > 0) / a3
        worst = max(worst, err)

    return {
        "islands": len(islands),
        "seam_components": seam_components(seam_edges),
        "seam_dead_ends": seam_dead_ends(seam_edges),
        "uv_overlap_pct": overlap_pct(bm, uv_layer),
        "worst_island_area_distortion": round(worst, 4),
        "seam_edges": len(seam_edges),
        "seam_length": round(seam_len, 4),
        "seam_length_pct": round(100.0 * seam_len / total_edge_len, 2),
        "seam_on_hard_edges_pct": round(100.0 * on_hard / (seam_len or 1.0), 1),
        "seam_visibility": None if vis_cost is None else round(vis_cost, 3),
        "area_distortion": round(area_err / (total_area or 1.0), 4),
        "angle_distortion_deg": round(math.degrees(ang_err / (total_area or 1.0)), 2),
        "flipped_faces": flipped,
        "uv_coverage_pct": round(100.0 * uv_total, 1),
    }
