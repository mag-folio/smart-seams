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

import bmesh
import bpy

from . import metrics


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


def straighten_quad_islands(obj, charts, max_stretch=0.12):
    """Follow-Active-Quads on all-quad charts (rims, bands, leg strips).

    Kept when the straightened island stays under ``max_stretch`` (or is no
    worse than the original unwrap): straight strips pack tighter and are
    much easier to paint on.
    """
    me = obj.data
    straightened = 0
    for faces_idx in charts:
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active
        faces = [bm.faces[i] for i in faces_idx]
        if len(faces) < 2 or any(len(f.verts) != 4 for f in faces):
            continue
        before = _island_stretch(faces, uv_layer)
        snap = _snapshot(faces, uv_layer)
        # active = largest face (most representative of the strip)
        active = max(faces, key=lambda f: f.calc_area())
        for f in bm.faces:
            f.select = False
        for f in faces:
            f.select = True
        bm.faces.active = active
        _square_active(active, uv_layer)
        bmesh.update_edit_mesh(me)
        try:
            bpy.ops.uv.follow_active_quads(mode='LENGTH_AVERAGE')
        except RuntimeError:
            bm = bmesh.from_edit_mesh(me)
            uv_layer = bm.loops.layers.uv.active
            bm.faces.ensure_lookup_table()
            _restore([bm.faces[i] for i in faces_idx], uv_layer, snap)
            bmesh.update_edit_mesh(me)
            continue
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active
        faces = [bm.faces[i] for i in faces_idx]
        after = _island_stretch(faces, uv_layer)
        if after <= max(before * 1.15, max_stretch) and not _self_overlaps(faces, uv_layer):
            straightened += 1
        else:
            _restore(faces, uv_layer, snap)
        bmesh.update_edit_mesh(me)
    return straightened


def align_to_grid(obj, charts, min_agreement=0.9):
    """Turn grid-shaped shells so their edges line up with the UV grid.

    The rotation is the length-weighted mean edge direction (modulo 90°).
    It is applied only when the edges clearly agree on one grid (box sides,
    straight strips); irregular shells are left for the packer to rotate.
    Shells with vertical walls are then turned in 90° steps so the object's
    up direction points up in UV, so a side band lies as a horizontal strip
    with its caps above and below it (the T layout).

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
        sx = sy = total = 0.0
        for l in loops:
            d = l.link_loop_next[uv_layer].uv - l[uv_layer].uv
            w = d.length
            if w < 1e-12:
                continue
            a = 4.0 * math.atan2(d.y, d.x)
            sx += w * math.cos(a)
            sy += w * math.sin(a)
            total += w
        if total <= 0 or math.hypot(sx, sy) < min_agreement * total:
            continue
        aligned.append(ci)
        rot = -math.atan2(sy, sx) / 4.0
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
