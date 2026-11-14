"""Fast trial flattening used to *validate* a chart before accepting it.

This is a compact Least Squares Conformal Maps (Lévy et al. 2002) solver in
numpy. It is not used for the final UVs (Blender's own unwrapper does that);
it only answers: "if these faces were one island, would it stretch or fold
over itself?" so the seam solver can refuse bad merges.
"""

import math

import numpy as np


def _wedges(faces, glued):
    """Map (vertex, face) corners to UV-vertex ids, splitting at unglued edges."""
    face_set = {f.index for f in faces}
    corner_id = {}
    next_id = 0
    for f in faces:
        for loop in f.loops:
            key = (loop.vert.index, f.index)
            if key in corner_id:
                continue
            # flood around the vertex through glued edges inside the chart
            stack = [loop]
            corner_id[key] = next_id
            while stack:
                l = stack.pop()
                v = l.vert
                for e in (l.edge, l.link_loop_prev.edge):
                    if not glued[e.index]:
                        continue
                    for l2 in e.link_loops:
                        g = l2.face
                        if g.index not in face_set or g is l.face:
                            continue
                        # the loop of g that sits on v
                        for l3 in g.loops:
                            if l3.vert is v:
                                k2 = (v.index, g.index)
                                if k2 not in corner_id:
                                    corner_id[k2] = next_id
                                    stack.append(l3)
                                break
            next_id += 1
    return corner_id, next_id


def face_triangles(f):
    """Corner-index triangles of a face, valid for concave n-gons.

    A plain fan from corner 0 is only correct for convex faces: for a concave
    quad or n-gon some fan triangles lie outside the face with reversed
    orientation, which LSCM then tries to flip, so a perfectly flat chart is
    reported as distorted and folded.
    """
    n = len(f.loops)
    if n == 3:
        return [(0, 1, 2)]
    from mathutils.geometry import tessellate_polygon
    co = [l.vert.co.copy() for l in f.loops]
    tris = tessellate_polygon([co])
    if len(tris) != n - 2:  # tessellation failed (degenerate polygon): fall back to a fan
        return [(0, i, i + 1) for i in range(1, n - 1)]
    out = []
    nrm = f.normal if f.normal.length > 0 else None
    for a, b, c in tris:
        # tessellate_polygon does not promise the face's winding: match it
        if nrm is not None and (co[b] - co[a]).cross(co[c] - co[a]).dot(nrm) < 0:
            b, c = c, b
        out.append((a, b, c))
    return out


def flatten(faces, glued):
    """Return (uv array [n,2], triangles [(ids), area3d]) or None on failure."""
    corner_id, n = _wedges(faces, glued)
    if n < 3:
        return None
    pos3 = np.zeros((n, 3))
    tris = []
    for f in faces:
        ids = [corner_id[(l.vert.index, f.index)] for l in f.loops]
        for l in f.loops:
            pos3[corner_id[(l.vert.index, f.index)]] = l.vert.co
        for a, b, c in face_triangles(f):
            tris.append((ids[a], ids[b], ids[c]))
    T = np.array(tris)
    p0, p1, p2 = pos3[T[:, 0]], pos3[T[:, 1]], pos3[T[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    cr = np.cross(e1, e2)
    area2 = np.linalg.norm(cr, axis=1)
    ok = area2 > 1e-14
    T, e1, e2, area2 = T[ok], e1[ok], e2[ok], area2[ok]
    if len(T) == 0:
        return None
    l1 = np.linalg.norm(e1, axis=1)
    x1 = l1
    x2 = np.einsum("ij,ij->i", e1, e2) / l1
    y2 = area2 / l1
    # local coords: q0=(0,0) q1=(x1,0) q2=(x2,y2)
    # W_j = q_{j+2} - q_{j+1} (complex), scaled by 1/sqrt(area2)
    s = 1.0 / np.sqrt(area2)
    W = np.stack([
        (x2 - x1) + 1j * y2,          # vertex 0: q2 - q1
        (0 - x2) + 1j * (0 - y2),      # vertex 1: q0 - q2
        (x1 - 0) + 0j,                 # vertex 2: q1 - q0
    ], axis=1) * s[:, None]

    m = len(T)
    A = np.zeros((2 * m, 2 * n))
    rows = np.arange(m)
    for j in range(3):
        col = T[:, j]
        wr, wi = W[:, j].real, W[:, j].imag
        # real part: wr*u - wi*v ; imag part: wi*u + wr*v
        np.add.at(A, (rows, col), wr)
        np.add.at(A, (rows, n + col), -wi)
        np.add.at(A, (m + rows, col), wi)
        np.add.at(A, (m + rows, n + col), wr)

    # pins: two far-apart vertices
    a = 0
    b = int(np.argmax(np.linalg.norm(pos3 - pos3[a], axis=1)))
    a = int(np.argmax(np.linalg.norm(pos3 - pos3[b], axis=1)))
    d = float(np.linalg.norm(pos3[a] - pos3[b])) or 1.0
    pinned = {a: (0.0, 0.0), b: (d, 0.0)}
    pin_cols = [a, b, n + a, n + b]
    pin_vals = np.array([0.0, d, 0.0, 0.0])
    free = np.ones(2 * n, bool)
    free[pin_cols] = False
    Af = A[:, free]
    rhs = -A[:, pin_cols] @ pin_vals
    try:
        x, *_ = np.linalg.lstsq(Af, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    sol = np.zeros(2 * n)
    sol[free] = x
    for k, (u, v) in pinned.items():
        sol[k], sol[n + k] = u, v
    uv = np.stack([sol[:n], sol[n:]], axis=1)
    return uv, T, area2 * 0.5


def quality(uv, T, area3):
    """(area_distortion, flipped_ratio, overlap_ratio) for a flattened chart."""
    a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
    area2 = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    tot2 = np.abs(area2).sum()
    tot3 = area3.sum()
    if tot2 <= 1e-14 or tot3 <= 0:
        return math.inf, 1.0, 1.0
    sign = 1.0 if area2.sum() >= 0 else -1.0
    flipped = float(area3[(area2 * sign) <= 0].sum() / tot3)
    k = tot2 / tot3
    r = np.abs(area2) / (area3 * k)
    dist = float((area3 * np.abs(np.log(np.clip(r, 1e-6, None)))).sum() / tot3)

    return dist, flipped, boundary_crossings(uv, T)


def boundary_crossings(uv, T):
    """Number of properly crossing, non-adjacent boundary segment pairs.

    With no flipped triangles, an island overlaps itself iff its UV boundary
    crosses itself, so this is an exact and fast overlap test.
    """
    und = {}
    for a, b, c in T:
        for x, y in ((a, b), (b, c), (c, a)):
            k = (x, y) if x < y else (y, x)
            und[k] = und.get(k, 0) + 1
    seg = np.array([k for k, n in und.items() if n == 1])
    if len(seg) < 4:
        return 0
    p, q = uv[seg[:, 0]], uv[seg[:, 1]]
    span = float(np.ptp(uv, axis=0).max()) or 1.0
    eps = span * 1e-9
    # pairwise orientation tests (vectorised)
    d = q - p                                   # (S,2)
    def orient(a_idx_p, a_d, pts):
        # cross(d_i, pts_j - p_i) for all i, j
        rel = pts[None, :, :] - a_idx_p[:, None, :]
        return a_d[:, None, 0] * rel[:, :, 1] - a_d[:, None, 1] * rel[:, :, 0]
    # side of p_j / q_j w.r.t. segment i, as a signed distance (dividing by
    # |d_i| makes the tolerance a distance; a fixed span^2 tolerance on the raw
    # cross product let short boundary segments cross unnoticed)
    ln = np.hypot(d[:, 0], d[:, 1])
    ln[ln == 0] = np.inf  # zero-length segments never straddle anything
    o1 = orient(p, d, p) / ln[:, None]
    o2 = orient(p, d, q) / ln[:, None]
    tol = span * 1e-6
    # segment j strictly straddles line i (touching / collinear doesn't count)
    cross_i = ((o1 > tol) & (o2 < -tol)) | ((o1 < -tol) & (o2 > tol))
    cross = cross_i & cross_i.T
    # ignore segments sharing an endpoint
    s0, s1 = seg[:, 0], seg[:, 1]
    share = ((s0[:, None] == s0[None, :]) | (s0[:, None] == s1[None, :]) |
             (s1[:, None] == s0[None, :]) | (s1[:, None] == s1[None, :]))
    cross &= ~share
    return int(np.triu(cross, 1).sum())


def chart_ok(faces, glued, max_distortion=0.08, max_overlap=0):
    res = flatten(faces, glued)
    if res is None:
        return False
    dist, flipped, crossings = quality(*res)
    return dist <= max_distortion and flipped <= 0.001 and crossings <= max_overlap
