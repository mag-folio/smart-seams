"""Turn seams into a finished UV layout.

Steps (all on the active edit-mode mesh):

1. Mark the solved seams on the mesh (so the artist can see and tweak them).
2. Unwrap with Blender's own solver (Angle Based by default).
3. Straighten all-quad strips with *Follow Active Quads* - but only keep the
   result when it is actually less distorted than the unwrap.
4. Normalise texel density, then shrink islands that are rarely seen
   (bottoms, backs, faces pressed against the ground) by visibility.
5. Pack into 0-1 with axis-aligned rotation.
"""

import math
import random

import bmesh
import bpy

from . import blender_validator, keepalive, metrics


def _island_stretch(faces, uv_layer):
    """Area-weighted angle + area error of one island (lower is better)."""
    a3 = sum(f.calc_area() for f in faces)
    a2 = sum(abs(metrics._uv_area(f, uv_layer)) for f in faces)
    if a3 <= 0 or a2 <= 1e-12:
        return float("inf")
    k = a2 / a3
    err = 0.0
    for f in faces:
        fa = f.calc_area()
        if fa <= 0:
            continue
        s = metrics._uv_area(f, uv_layer)
        r = abs(s) / (fa * k)
        ang3 = [l.calc_angle() for l in f.loops]
        ang2 = metrics._uv_corner_angles(f, uv_layer)
        err += fa * (abs(math.log(max(r, 1e-6))) +
                     sum(abs(x - y) for x, y in zip(ang3, ang2)) / len(ang3))
    return err / a3


def _self_overlaps(faces, uv_layer):
    """True if the island's UV boundary crosses itself or any face flipped."""
    import numpy as np

    from .flatten import boundary_crossings
    key_id, pts, tris, signs = {}, [], [], []
    for f in faces:
        ids = []
        for l in f.loops:
            uv = l[uv_layer].uv
            k = (l.vert.index, round(uv.x, 7), round(uv.y, 7))
            if k not in key_id:
                key_id[k] = len(pts)
                pts.append((uv.x, uv.y))
            ids.append(key_id[k])
        for i in range(1, len(ids) - 1):
            tris.append((ids[0], ids[i], ids[i + 1]))
        signs.append(metrics._uv_area(f, uv_layer))
    pos = sum(1 for s in signs if s > 1e-14)
    neg = sum(1 for s in signs if s < -1e-14)
    if pos and neg:
        return True
    return boundary_crossings(np.array(pts), np.array(tris)) > 0


def _tris_overlap(a, b, eps):
    """Separating-axis test for two 2D triangles (touching is not overlap)."""
    for tri in (a, b):
        for i in range(3):
            (px, py), (qx, qy) = tri[i], tri[(i + 1) % 3]
            nx, ny = qy - py, px - qx
            ln = math.hypot(nx, ny)
            if ln < 1e-15:
                continue
            nx /= ln
            ny /= ln
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) <= min(pb) + eps or max(pb) <= min(pa) + eps:
                return False
    return True


def _faces_overlap(faces, uv_layer):
    """True if any two faces of the island cover the same UV area.

    Catches folds that keep every face's orientation and a clean outline
    (Follow Active Quads round a pole vertex can lay one quad on another).
    """
    tris = []
    for f in faces:
        uv = [(l[uv_layer].uv.x, l[uv_layer].uv.y) for l in f.loops]
        for i in range(1, len(uv) - 1):
            tris.append((f.index, (uv[0], uv[i], uv[i + 1])))
    if not tris:
        return False
    xs = [p[0] for _, t in tris for p in t]
    ys = [p[1] for _, t in tris for p in t]
    size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-12)
    eps = size * 1e-6
    cell = size / max(1.0, math.sqrt(len(tris)))
    grid = {}
    for k, (_, t) in enumerate(tris):
        x0 = int(min(p[0] for p in t) // cell)
        x1 = int(max(p[0] for p in t) // cell)
        y0 = int(min(p[1] for p in t) // cell)
        y1 = int(max(p[1] for p in t) // cell)
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                grid.setdefault((gx, gy), []).append(k)
    tested = set()
    for members in grid.values():
        keepalive.pulse()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if tris[a][0] == tris[b][0] or (a, b) in tested:
                    continue
                tested.add((a, b))
                if _tris_overlap(tris[a][1], tris[b][1], eps):
                    return True
    return False


def _snapshot(faces, uv_layer):
    return {(f.index, i): l[uv_layer].uv.copy() for f in faces for i, l in enumerate(f.loops)}


def _restore(faces, uv_layer, snap):
    for f in faces:
        for i, l in enumerate(f.loops):
            l[uv_layer].uv = snap[(f.index, i)]


def _square_active(face, uv_layer):
    """Give a quad a rectangular UV matching its 3D edge lengths."""
    loops = list(face.loops)
    w = (loops[0].edge.calc_length() + loops[2].edge.calc_length()) * 0.5
    h = (loops[1].edge.calc_length() + loops[3].edge.calc_length()) * 0.5
    c = sum((l[uv_layer].uv for l in loops), loops[0][uv_layer].uv * 0) / 4
    # keep the current scale of the island
    cur = abs(metrics._uv_area(face, uv_layer))
    s = math.sqrt(cur / (w * h)) if w * h > 0 and cur > 0 else 1.0
    w *= s
    h *= s
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    for l, (x, y) in zip(loops, corners):
        l[uv_layer].uv = (c.x + x, c.y + y)


def _area_and_angle_error(faces, uv_layer):
    """(area stretch, mean angle error in radians), both area weighted.

    The area term is the same measure as the metrics' worst shell stretch.
    """
    a3 = sum(f.calc_area() for f in faces)
    a2 = sum(abs(metrics._uv_area(f, uv_layer)) for f in faces)
    if a3 <= 0 or a2 <= 1e-12:
        return float("inf"), float("inf")
    k = a2 / a3
    area = angle = 0.0
    for f in faces:
        fa = f.calc_area()
        if fa <= 0:
            continue
        r = abs(metrics._uv_area(f, uv_layer)) / (fa * k)
        area += fa * abs(math.log(max(r, 1e-6)))
        ang3 = [l.calc_angle() for l in f.loops]
        ang2 = metrics._uv_corner_angles(f, uv_layer)
        angle += fa * sum(abs(x - y) for x, y in zip(ang3, ang2)) / len(ang3)
    return area / a3, angle / a3


def _active_candidates(faces, count):
    """Faces to try as the active quad: the largest ones plus an even spread."""
    by_area = sorted(faces, key=lambda f: -f.calc_area())
    out = by_area[:max(1, count // 2)]
    step = max(1, len(faces) // max(1, count - len(out)))
    for f in faces[::step]:
        if len(out) >= count:
            break
        if f not in out:
            out.append(f)
    return out


def straighten_quad_islands(obj, charts, max_stretch=0.12, max_area=0.1,
                            max_angle=math.radians(10.5), tries=12, time_limit=None):
    """Follow-Active-Quads on all-quad charts (rims, bands, leg strips).

    Straight strips pack far tighter than arcs and are much easier to paint
    on. A straightened island is kept when it neither overlaps nor flips and

    * its combined stretch stays under ``max_stretch`` (or within 15 % of
      the unwrap), or
    * its area stretch stays under ``max_area`` and its mean angle error
      under ``max_angle``: a gently tapered band (a lamp's conical side) may
      trade a little angle error for a straight strip, but never more area
      stretch than the strict limit.

    Follow Active Quads depends on which quad it starts from, so several
    active faces are tried and the least distorted acceptable result wins.

    Returns ``(count, relaxed)``: the number of straightened charts, and for
    those kept only under the relaxed (angle-trading) rule, a map of chart
    index -> UV snapshot of the unwrap, so the caller can undo them when
    the long strip they make costs more texture space than it is worth.

    ``time_limit`` (seconds): charts not reached in time keep their unwrap.
    """
    import time
    me = obj.data
    straightened = 0
    relaxed = {}
    end = (time.perf_counter() + time_limit) if time_limit else None
    for ci, faces_idx in enumerate(charts):
        if end is not None and time.perf_counter() > end:
            break
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active
        faces = [bm.faces[i] for i in faces_idx]
        if len(faces) < 2 or any(len(f.verts) != 4 or f.hide for f in faces):
            continue
        before = _island_stretch(faces, uv_layer)
        area0, angle0 = _area_and_angle_error(faces, uv_layer)
        snap = _snapshot(faces, uv_layer)
        best = None
        for active_idx in [f.index for f in _active_candidates(faces, tries)]:
            keepalive.pulse()
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            uv_layer = bm.loops.layers.uv.active
            faces = [bm.faces[i] for i in faces_idx]
            _restore(faces, uv_layer, snap)
            # Follow Active Quads needs the active face selected and visible
            # (else "Active face not selected"): select only this chart, with
            # verts and edges, and flush so the selection is consistent
            for f in bm.faces:
                f.select_set(False)
            for f in faces:
                f.select_set(True)
            bm.select_flush(True)
            active = bm.faces[active_idx]
            bm.faces.active = active
            _square_active(active, uv_layer)
            bmesh.update_edit_mesh(me)
            try:
                blender_validator.call_op("uv.follow_active_quads", mode='LENGTH_AVERAGE')
            except RuntimeError:
                continue
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            uv_layer = bm.loops.layers.uv.active
            faces = [bm.faces[i] for i in faces_idx]
            after = _island_stretch(faces, uv_layer)
            area, angle = _area_and_angle_error(faces, uv_layer)
            strict = after <= max(before * 1.15, max_stretch)
            ok = strict or (area <= max(area0 * 1.15, max_area) and
                            angle <= max(angle0 * 1.15, max_angle))
            if (ok and (best is None or (not strict, after) < best[:2])
                    and not _self_overlaps(faces, uv_layer)
                    and not _faces_overlap(faces, uv_layer)):
                best = (not strict, after, _snapshot(faces, uv_layer))
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active
        faces = [bm.faces[i] for i in faces_idx]
        if best is None:
            _restore(faces, uv_layer, snap)
        else:
            _restore(faces, uv_layer, best[2])
            straightened += 1
            if best[0]:
                relaxed[ci] = snap
        bmesh.update_edit_mesh(me)
    return straightened, relaxed


def _grid_rotation(loops, uv_layer, min_agreement, tol=math.radians(4.0)):
    """Rotation that puts the shell's dominant edge grid on the UV axes.

    Edge directions are taken modulo 90°. The dominant direction is the one
    with the most edge length within ``tol`` of it (a bevelled corner or a
    splayed foot does not drag it off, as it would a plain mean). Returns
    None unless at least ``min_agreement`` of the edge length lies on that
    grid.
    """
    dirs, total = [], 0.0
    for l in loops:
        d = l.link_loop_next[uv_layer].uv - l[uv_layer].uv
        w = d.length
        if w < 1e-12:
            continue
        dirs.append((math.atan2(d.y, d.x) % (math.pi / 2), w))
        total += w
    if total <= 0:
        return None

    def support(a):
        return sum(w for b, w in dirs
                   if abs(math.remainder(b - a, math.pi / 2)) <= tol)

    # histogram in 0.5° bins, circular window of +-tol
    nb = 180
    hist = [0.0] * nb
    for b, w in dirs:
        hist[int(b / (math.pi / 2) * nb) % nb] += w
    half = max(1, int(round(tol / (math.pi / 2) * nb)))
    window = [sum(hist[(i + k) % nb] for k in range(-half, half + 1)) for i in range(nb)]
    best = (max(range(nb), key=window.__getitem__) + 0.5) * (math.pi / 2) / nb
    # refine: length-weighted mean of the inliers around the peak
    sx = sy = 0.0
    for b, w in dirs:
        if abs(math.remainder(b - best, math.pi / 2)) <= tol:
            sx += w * math.cos(4.0 * b)
            sy += w * math.sin(4.0 * b)
    peak = math.atan2(sy, sx) / 4.0
    if support(peak) < min_agreement * total:
        return None
    return -peak


def align_to_grid(obj, charts, min_agreement=0.5, anchors=()):
    """Turn grid-shaped shells so their edges line up with the UV grid.

    The rotation is the dominant edge direction (modulo 90°). It is applied
    when at least ``min_agreement`` of the shell's edge length lies on one
    grid (box sides, straight strips, a leg strip with a splayed foot);
    irregular shells are left for the packer to rotate.
    Shells with vertical walls are then turned in 90° steps so the object's
    up direction points up in UV.

    A shell holding a side strip (``anchors``: the faces of the strip's end
    panel, which carries both caps) is turned a further quarter so the strip
    runs vertically, with the end panel and its two caps at the bottom
    (the upside-down T).

    Returns the indices of the aligned charts (pin them while packing).
    """
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    aligned = []
    for ci, faces_idx in enumerate(charts):
        faces = [bm.faces[i] for i in faces_idx]
        loops = [l for f in faces for l in f.loops]
        rot = _grid_rotation(loops, uv_layer, min_agreement)
        if rot is None:
            continue
        aligned.append(ci)
        # which way does 3D "up" run in UV after that rotation?
        ux = uy = 0.0
        c, s = math.cos(rot), math.sin(rot)
        for l in loops:
            d = l.link_loop_next[uv_layer].uv - l[uv_layer].uv
            dz = l.link_loop_next.vert.co.z - l.vert.co.z
            ux += (c * d.x - s * d.y) * dz
            uy += (s * d.x + c * d.y) * dz
        if ux * ux + uy * uy > 1e-12:
            quarter = round(math.atan2(uy, ux) / (math.pi / 2))  # 1 = already up
            rot += (1 - quarter) * math.pi / 2
        anchor = next((set(a) for a in anchors if set(a) & set(faces_idx)), None)
        if anchor:
            rot += math.pi / 2
            # anchor panel at the bottom of the strip
            c, s = math.cos(rot), math.sin(rot)
            def v_after(ls):
                return sum(s * l[uv_layer].uv.x + c * l[uv_layer].uv.y for l in ls) / len(ls)
            if v_after([l for f in faces if f.index in anchor for l in f.loops]) > v_after(loops):
                rot += math.pi
        if abs(math.remainder(rot, 2 * math.pi)) < 1e-6:
            continue
        cx = sum(l[uv_layer].uv.x for l in loops) / len(loops)
        cy = sum(l[uv_layer].uv.y for l in loops) / len(loops)
        c, s = math.cos(rot), math.sin(rot)
        for l in loops:
            x, y = l[uv_layer].uv.x - cx, l[uv_layer].uv.y - cy
            l[uv_layer].uv = (cx + c * x - s * y, cy + s * x + c * y)
    bmesh.update_edit_mesh(me)
    return aligned


def set_pinned(obj, charts, pinned):
    """Pin or unpin every UV of the given charts."""
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    for faces_idx in charts:
        for fi in faces_idx:
            for l in bm.faces[fi].loops:
                l[uv_layer].pin_uv = pinned
    bmesh.update_edit_mesh(me)


PACK_VARIANTS = (
    ('CONCAVE', 'ANY'),
    ('CONCAVE', 'AXIS_ALIGNED'),
)


def _spread_charts(bm, uv_layer, charts):
    """Lay the shells out in a row with gaps between them.

    The final unwrap packs shells edge to edge, so a vertex on a seam can end
    up at exactly the same UV spot in both shells. Blender's packer then sees
    the two shells as one island and never pulls them apart, which left
    shells stacked on top of each other. Separated shells can't be joined.
    """
    bm.faces.ensure_lookup_table()
    boxes = []
    for c in charts:
        xs = [l[uv_layer].uv.x for fi in c for l in bm.faces[fi].loops]
        ys = [l[uv_layer].uv.y for fi in c for l in bm.faces[fi].loops]
        boxes.append((min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0))
    gap = 0.1 * max([max(b[2] - b[0], b[3] - b[1]) for b in boxes] + [1e-3])
    x = 0.0
    for c, (x0, y0, x1, y1) in zip(charts, boxes):
        dx, dy = x - x0, -y0
        for fi in c:
            for l in bm.faces[fi].loops:
                l[uv_layer].uv = (l[uv_layer].uv.x + dx, l[uv_layer].uv.y + dy)
        x += (x1 - x0) + gap


def _uv_area_total(bm, uv_layer):
    return sum(abs(metrics._uv_area(f, uv_layer)) for f in bm.faces)


def rotate_chart(faces, uv_layer, angle):
    """Turn one island about its centroid."""
    loops = [l for f in faces for l in f.loops]
    cx = sum(l[uv_layer].uv.x for l in loops) / len(loops)
    cy = sum(l[uv_layer].uv.y for l in loops) / len(loops)
    c, s = math.cos(angle), math.sin(angle)
    for l in loops:
        x, y = l[uv_layer].uv.x - cx, l[uv_layer].uv.y - cy
        l[uv_layer].uv = (cx + c * x - s * y, cy + s * x + c * y)


def _rasterize(tris, pad):
    """Conservative mask of 2D triangles (``tris``: N x 3 x 2, already in
    pixel units, min corner at 0): every pixel the triangles touch is set.

    Pixel centres inside a triangle are set, plus every pixel an edge passes
    through (edges sampled at quarter-pixel steps), then grown by one pixel:
    any point of a triangle lies within one pixel of a set pixel.

    Same tests as ``_rasterize_reference`` (the exact same centre test, and
    every edge sample point it uses plus one between each pair), so the mask
    contains everything the original marked. It is computed for all
    triangles at once in numpy (grouped by box size and edge length), which
    is what keeps dense meshes fast.
    """
    import numpy as np
    tris = np.asarray(tris, dtype=float)
    w = int(math.ceil(tris[..., 0].max())) + 2 * pad + 2
    h = int(math.ceil(tris[..., 1].max())) + 2 * pad + 2
    grid = np.zeros((h, w), dtype=bool)
    t = tris + pad
    ax, ay = t[:, 0, 0], t[:, 0, 1]
    bx, by = t[:, 1, 0], t[:, 1, 1]
    cx, cy = t[:, 2, 0], t[:, 2, 1]

    # interiors: pixel-centre test over each box, boxes grouped by size
    x0 = np.minimum(np.minimum(ax, bx), cx).astype(int)
    x1 = np.ceil(np.maximum(np.maximum(ax, bx), cx)).astype(int)
    y0 = np.minimum(np.minimum(ay, by), cy).astype(int)
    y1 = np.ceil(np.maximum(np.maximum(ay, by), cy)).astype(int)
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    live = np.abs(den) > 1e-12
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    kw = 1 << np.ceil(np.log2(np.maximum(bw, 1))).astype(int)
    kh = 1 << np.ceil(np.log2(np.maximum(bh, 1))).astype(int)
    for gw, gh in set(zip(kw[live].tolist(), kh[live].tolist())):
        idx = np.nonzero(live & (kw == gw) & (kh == gh))[0]
        ox, oy = np.meshgrid(np.arange(gw), np.arange(gh))
        ox, oy = ox.ravel()[None, :], oy.ravel()[None, :]
        chunk = max(1, 4_000_000 // (gw * gh))
        for s0 in range(0, len(idx), chunk):
            k = idx[s0:s0 + chunk]
            px = x0[k][:, None] + ox
            py = y0[k][:, None] + oy
            fx, fy = px + 0.5, py + 0.5
            Ax, Ay, Bx, By, Cx, Cy = (v[k][:, None] for v in (ax, ay, bx, by, cx, cy))
            D = den[k][:, None]
            w1 = ((By - Cy) * (fx - Cx) + (Cx - Bx) * (fy - Cy)) / D
            w2 = ((Cy - Ay) * (fx - Cx) + (Ax - Cx) * (fy - Cy)) / D
            m = ((w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
                 & (ox < bw[k][:, None]) & (oy < bh[k][:, None]))
            grid[py[m], px[m]] = True

    # edges: the original's sample points (n per edge) and one between each pair
    for (sx, sy), (ex, ey) in (((ax, ay), (bx, by)), ((bx, by), (cx, cy)), ((cx, cy), (ax, ay))):
        n = (np.hypot(ex - sx, ey - sy) * 2).astype(int) + 2
        for nv in np.unique(n):
            k = np.nonzero(n == nv)[0]
            st = np.linspace(0.0, 1.0, 2 * (nv - 1) + 1)[None, :]
            chunk = max(1, 4_000_000 // st.shape[1])
            for s0 in range(0, len(k), chunk):
                kk = k[s0:s0 + chunk]
                X = (sx[kk][:, None] + (ex[kk] - sx[kk])[:, None] * st).astype(int)
                Y = (sy[kk][:, None] + (ey[kk] - sy[kk])[:, None] * st).astype(int)
                grid[Y.ravel(), X.ravel()] = True
    return _dilate(grid, 1)


def _rasterize_reference(tris, pad):
    """The original per-triangle rasteriser, kept to test ``_rasterize`` against."""
    import numpy as np
    w = int(math.ceil(tris[..., 0].max())) + 2 * pad + 2
    h = int(math.ceil(tris[..., 1].max())) + 2 * pad + 2
    grid = np.zeros((h, w), dtype=bool)
    t = tris + pad
    for (ax, ay), (bx, by), (cx, cy) in t:
        x0, x1 = int(min(ax, bx, cx)), int(math.ceil(max(ax, bx, cx)))
        y0, y1 = int(min(ay, by, cy)), int(math.ceil(max(ay, by, cy)))
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) > 1e-12:
            px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
            w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
            grid[y0:y1 + 1, x0:x1 + 1] |= (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        for (sx, sy), (ex, ey) in (((ax, ay), (bx, by)), ((bx, by), (cx, cy)), ((cx, cy), (ax, ay))):
            n = int(math.hypot(ex - sx, ey - sy) * 2) + 2
            s = np.linspace(0.0, 1.0, n)
            grid[(sy + (ey - sy) * s).astype(int), (sx + (ex - sx) * s).astype(int)] = True
    return _dilate(grid, 1)


def _dilate(grid, k):
    """Grow a mask by ``k`` pixels (square neighbourhood)."""
    import numpy as np
    out = grid.copy()
    for _ in range(k):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g2 = g.copy()
        g2[:, 1:] |= g[:, :-1]
        g2[:, :-1] |= g[:, 1:]
        out = g2
    return out


def _min_rect_angle(points):
    """Rotation that turns a point set to its minimal-area bounding box."""
    import numpy as np
    pts = np.unique(np.round(points, 9), axis=0)
    if len(pts) < 3:
        return 0.0
    # convex hull (monotone chain)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) -
                                     (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    best = (float("inf"), 0.0)
    for i in range(len(hull)):
        d = hull[(i + 1) % len(hull)] - hull[i]
        a = -math.atan2(d[1], d[0])
        c, s = math.cos(a), math.sin(a)
        x = hull[:, 0] * c - hull[:, 1] * s
        y = hull[:, 0] * s + hull[:, 1] * c
        area = (x.max() - x.min()) * (y.max() - y.min())
        if area < best[0]:
            best = (area, a)
    return best[1]


def _fft_size(n):
    """Smallest length >= n with no prime factors above 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def raster_pack(islands, margin, coverage_floor, res=300, steps=6, n_angles=16,
                strategies=(('area', 'top'), ('bbox', 'corner')), time_limit=None):
    """Bottom-left raster packer with free rotation for irregular islands.

    ``islands``: list of ``(tris, pinned)`` with ``tris`` an N x 3 x 2 numpy
    array in UV units (all islands at their final relative scale, total
    area about 1). Pinned islands keep their rotation; the others try
    ``n_angles`` turns starting from their minimal bounding box. Each
    strategy (island order: by area or by bounding box; spot: lowest top
    edge, or smallest square from the corner) is tried in turn at every
    texture size until one fits everything.

    Masks are conservative rasterisations grown by half the margin, so two
    placed islands are always at least ``margin`` apart (a pixel more, at
    worst). The texture size in pixels is binary searched between the
    coverage Blender's packer reached (``coverage_floor``) and 40 % more.

    Returns ``(coverage, [(angle, scale, tx, ty) per island])`` mapping a UV
    point p to ``rot(angle) @ p * scale + (tx, ty)``, or None if it found
    nothing better than ``coverage_floor``. With ``time_limit`` (seconds) it
    stops searching when the time is up and returns the best fit found.
    """
    import time
    import numpy as np
    end = (time.perf_counter() + time_limit) if time_limit else None

    def late():
        return end is not None and time.perf_counter() > end
    areas = [float(np.abs(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])).sum()) * 0.5
             for t, _ in islands]
    area = sum(areas)
    if area <= 0:
        return None
    order = sorted(range(len(islands)), key=lambda i: -areas[i])

    # coverage = area * res^2 / size^2
    def size_for(cov):
        return int(math.ceil(res * math.sqrt(area / cov)))

    lo_size, hi_size = size_for(coverage_floor * 1.4), size_for(coverage_floor)
    # room round each raw mask for the margin it grows by later
    pad = int(math.ceil(margin * 0.5 * hi_size)) + 2
    # candidate rotations and their raw masks (pixel size 1 / res UV units)
    cands = []
    for tris, pinned in islands:
        pts = tris.reshape(-1, 2)
        if pinned:
            angles = [0.0]
        else:
            a0 = _min_rect_angle(pts)
            angles = [a0 + k * 2 * math.pi / n_angles for k in range(n_angles)]
        cs = []
        for a in angles:
            c, s = math.cos(a), math.sin(a)
            rot = np.array([[c, -s], [s, c]])
            t = tris @ rot.T
            lo = t.reshape(-1, 2).min(axis=0)
            cs.append((a, lo, _rasterize((t - lo) * res, pad)))
        cands.append(cs)

    def attempt(size, order, score):
        """Place everything in a size x size pixel texture, or None."""
        k = int(math.ceil(margin * 0.5 * size))  # half margin, in pixels
        occ = np.zeros((size, size), dtype=float)
        placed = [None] * len(islands)
        # FFTs at a 2-3-5 smooth length are several times faster than at an
        # arbitrary (often prime) size. Zero padding changes nothing in the
        # region read below: there the mask lies wholly inside the texture.
        n = _fft_size(size)
        for i in order:
            keepalive.pulse()
            if late():
                return None
            occ_f = np.fft.rfft2(occ, s=(n, n))
            best = None
            for a, lo, raw in cands[i]:
                keepalive.pulse()
                m = _dilate(raw, k)
                ys, xs = np.nonzero(m)
                j0, j1, i0, i1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                m = m[j0:j1, i0:i1]
                h, w = m.shape
                if h > size or w > size:
                    continue
                mp = np.zeros((size, size))
                mp[:h, :w] = m
                hit = np.fft.irfft2(occ_f * np.conj(np.fft.rfft2(mp, s=(n, n))), s=(n, n))
                free = hit[:size - h + 1, :size - w + 1] < 0.5
                if not free.any():
                    continue
                if score == 'top':
                    # lowest top edge, then leftmost
                    rows = np.nonzero(free.any(axis=1))[0]
                    y = int(rows[0])
                    x = int(np.nonzero(free[y])[0][0])
                    key = (y + h, x)
                else:
                    # 'corner': smallest square from the origin that holds it
                    yy, xx = np.nonzero(free)
                    ext = np.maximum(yy + h, xx + w)
                    j = int(np.lexsort((xx + yy, ext))[0])
                    y, x = int(yy[j]), int(xx[j])
                    key = (int(ext[j]), x + y)
                if best is None or key < best[0]:
                    best = (key, a, lo, m, x, y, i0, j0)
            if best is None:
                return None
            _, a, lo, m, x, y, i0, j0 = best
            occ[y:y + m.shape[0], x:x + m.shape[1]] += m
            placed[i] = (a, lo, x - i0, y - j0)
        return placed

    bbox = []
    for cs in cands:
        _, _, raw = cs[0]
        ys, xs = np.nonzero(raw)
        bbox.append(float((ys.ptp() + 1) * (xs.ptp() + 1)))
    orders = {
        'area': order,
        'bbox': sorted(range(len(islands)), key=lambda i: -bbox[i]),
    }

    def attempt_any(size):
        for o, sc in strategies:
            got = attempt(size, orders[o], sc)
            if got is not None:
                return got
        return None

    best = None
    for _ in range(steps):
        if hi_size - lo_size <= 1 or late():
            break
        mid = (lo_size + hi_size) // 2
        got = attempt_any(mid)
        if got is None:
            lo_size = mid
        else:
            hi_size, best = mid, (mid, got)
    if best is None:
        return None
    size, placed = best
    out = []
    for (tris, _), (a, lo, ox, oy) in zip(islands, placed):
        # pixel q = (rot(a) p - lo) * res + pad; texture px = q + (ox, oy)
        sc = res / size
        out.append((a, sc, (-lo[0] * res + pad + ox) / size, (-lo[1] * res + pad + oy) / size))
    return area * (res / size) ** 2, out


def pack(obj, charts, pinned_charts=(), margin=0.01, variants=PACK_VARIANTS, starts=4,
         use_raster=True, raster_time=None):
    """Pack all islands into 0-1 as tightly as Blender's packer allows.

    * Islands keep their relative scale (equal texel density, then the
      visibility scaling), so packing only scales the whole layout.
    * ``pinned_charts`` (grid-aligned shells) keep their rotation; the
      packer may turn the others freely.
    * ``margin`` is the gap between two islands, as a fraction of the
      texture (0.01 = about 10 px at 1K, enough for a few mip levels); each
      island gets half of it on every side, so the gap to the texture border
      is ``margin / 2``. (Blender's own margin is padding *per island*, so
      passing it straight through made every gap twice the setting.) It is
      given to the packer as an absolute (ADD) margin, calibrated from the
      scale the layout ends up at.
    * Several shape / rotation settings are tried, and with free rotation
      also ``starts - 1`` seeded random turns of the free shells (Blender's
      packer only searches near each island's current angle); the best
      coverage wins. The seed is fixed, so results are repeatable.

    Returns ``(coverage, shape_method, rotate_method)`` of the kept layout.
    """
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    for f in bm.faces:
        f.select_set(True)
    # normalise: total UV area 1 (uniform, so texel ratios are unchanged)
    area = _uv_area_total(bm, uv_layer)
    if area > 1e-12:
        s = 1.0 / math.sqrt(area)
        for f in bm.faces:
            for l in f.loops:
                l[uv_layer].uv = l[uv_layer].uv * s
    _spread_charts(bm, uv_layer, charts)
    snap = _snapshot(bm.faces, uv_layer)
    bmesh.update_edit_mesh(me)
    pinned_faces = {fi for c in pinned_charts for fi in c}
    free = [c for c in charts if not set(c) & pinned_faces]
    rng = random.Random(1)

    def start_layout(k):
        """UVs to pack from: as unwrapped, or with the free shells turned
        (Blender's packer only tries a few angles around the current one)."""
        if k == 0:
            return snap
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uv = bm.loops.layers.uv.active
        _restore(bm.faces, uv, snap)
        for c in free:
            rotate_chart([bm.faces[i] for i in c], uv, rng.uniform(0.0, 2.0 * math.pi))
        return _snapshot(bm.faces, uv)

    def run(start, shape, rot, m):
        keepalive.pulse()
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        _restore(bm.faces, bm.loops.layers.uv.active, start)
        bmesh.update_edit_mesh(me)
        set_pinned(obj, pinned_charts, True)
        try:
            bpy.ops.uv.pack_islands(rotate=True, rotate_method=rot, scale=True,
                                    margin_method='ADD', margin=m, shape_method=shape,
                                    pin=True, pin_method='ROTATION')
        finally:
            set_pinned(obj, pinned_charts, False)
        bm = bmesh.from_edit_mesh(me)
        return _uv_area_total(bm, bm.loops.layers.uv.active)

    margin = margin * 0.5  # per island side
    best = None
    candidates = []
    # final linear scale = sqrt(coverage); the margin scales with it
    m = margin / math.sqrt(0.5)
    for k in range(starts if free else 1):
        start = start_layout(k)
        for shape, rot in variants:
            if k and rot != 'ANY':
                continue  # turning the start only matters to free rotation
            for _ in range(3):
                cov = run(start, shape, rot, m)
                m_new = margin / math.sqrt(max(cov, 1e-6))
                if abs(m_new - m) <= 0.03 * m:
                    break
                m = m_new
            # keep it only if the gap is at least the requested one
            if m * math.sqrt(cov) < margin * 0.97:
                m = m_new
                cov = run(start, shape, rot, m)
                if m * math.sqrt(cov) < margin * 0.97:
                    continue
            bm = bmesh.from_edit_mesh(me)
            candidates.append((cov, shape, rot, _snapshot(bm.faces, bm.loops.layers.uv.active)))
    # Best coverage that really has no overlapping shells (Blender's packer
    # can't separate shells it thinks are joined, and reports that as a
    # near perfect fit).
    for cand in sorted(candidates, key=lambda c: -c[0]):
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uv = bm.loops.layers.uv.active
        _restore(bm.faces, uv, cand[3])
        inside = all(-1e-4 <= l[uv].uv.x <= 1 + 1e-4 and -1e-4 <= l[uv].uv.y <= 1 + 1e-4
                     for f in bm.faces for l in f.loops)
        if inside and not _faces_overlap(list(bm.faces), uv):
            best = cand
            break
    if best is None:  # calibration never settled: plain run, margin as given
        cov = run(snap, 'CONCAVE', 'AXIS_ALIGNED', margin / math.sqrt(0.5))
        bm = bmesh.from_edit_mesh(me)
        if _faces_overlap(list(bm.faces), bm.loops.layers.uv.active):
            cov = 0.05  # overlapping: any clean raster layout beats it
        best = (cov, 'CONCAVE', 'AXIS_ALIGNED', _snapshot(bm.faces, bm.loops.layers.uv.active))

    # second opinion: raster packer with free rotation (nests concave shells)
    if use_raster and len(charts) > 1:
        import numpy as np
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uv = bm.loops.layers.uv.active
        islands = []
        for c in charts:
            tris = []
            for fi in c:
                pts = [tuple(snap[(fi, j)]) for j in range(len(bm.faces[fi].loops))]
                tris += [(pts[0], pts[j], pts[j + 1]) for j in range(1, len(pts) - 1)]
            islands.append((np.array(tris, dtype=float), bool(set(c) & pinned_faces)))
        got = raster_pack(islands, margin * 2.0, best[0], time_limit=raster_time)
        if got is not None:
            for c, (a, sc, tx, ty) in zip(charts, got[1]):
                ca, sa = math.cos(a) * sc, math.sin(a) * sc
                for fi in c:
                    for j, l in enumerate(bm.faces[fi].loops):
                        x, y = snap[(fi, j)]
                        l[uv].uv = (ca * x - sa * y + tx, sa * x + ca * y + ty)
            cov = _uv_area_total(bm, uv)
            inside = all(-1e-6 <= l[uv].uv.x <= 1 + 1e-6 and -1e-6 <= l[uv].uv.y <= 1 + 1e-6
                         for f in bm.faces for l in f.loops)
            if cov > best[0] and inside and not _faces_overlap(list(bm.faces), uv):
                best = (cov, 'RASTER', 'ANY', _snapshot(bm.faces, uv))
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    _restore(bm.faces, bm.loops.layers.uv.active, best[3])
    bmesh.update_edit_mesh(me)
    return best[0], best[1], best[2]


def scale_by_visibility(obj, charts, visibility, min_scale=0.5, matrix_world=None):
    """Uniform texel density, then shrink rarely-seen islands.

    The linear scale of each island is ``lerp(min_scale, 1, visibility)``
    where visibility is the area-weighted mean of its faces.
    """
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    mw = matrix_world or obj.matrix_world
    world_scale = abs(mw.to_3x3().determinant()) ** (1.0 / 3.0) or 1.0
    factors = []
    for faces_idx in charts:
        faces = [bm.faces[i] for i in faces_idx]
        a3 = sum(f.calc_area() for f in faces) * world_scale ** 2
        a2 = sum(abs(metrics._uv_area(f, uv_layer)) for f in faces)
        if a3 <= 0 or a2 <= 1e-12:
            factors.append(1.0)
            continue
        vis = sum(visibility[f.index] * f.calc_area() for f in faces) / (a3 / world_scale ** 2)
        lin = min_scale + (1.0 - min_scale) * max(0.0, min(1.0, vis))
        density = math.sqrt(a3 / a2) * lin
        loops = [l for f in faces for l in f.loops]
        cx = sum(l[uv_layer].uv.x for l in loops) / len(loops)
        cy = sum(l[uv_layer].uv.y for l in loops) / len(loops)
        for l in loops:
            uv = l[uv_layer].uv
            l[uv_layer].uv = (cx + (uv.x - cx) * density, cy + (uv.y - cy) * density)
        factors.append(lin)
    bmesh.update_edit_mesh(me)
    return factors
