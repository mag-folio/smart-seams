"""Per-face visibility estimation.

A face's *visibility* (0..1) estimates how likely it is to be seen by a camera
in normal use. It drives two decisions:

* where seams go (seams are pushed toward low-visibility edges), and
* how much UV space an island gets (hidden islands are shrunk).

Three cues are combined:

1. **Openness** - ambient-occlusion style ray casting against the object
   itself (and an optional ground plane at the object's lowest point). Faces
   between table legs, inside cavities or touching the floor score low.
2. **Up/down facing** - faces pointing at the ground are rarely seen.
3. **Front direction** (optional) - faces pointing away from a chosen front
   axis or the scene camera are de-prioritised.
"""

import math

from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import keepalive


def _hemisphere_dirs(count):
    """Deterministic, roughly uniform directions on the unit sphere (Fibonacci)."""
    dirs = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        z = 1.0 - (2.0 * i + 1.0) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        dirs.append(Vector((r * math.cos(theta), r * math.sin(theta), z)))
    return dirs


def face_visibility(bm, matrix_world, *, samples=48, use_ground=True,
                    front=None, back_weight=0.6, down_weight=0.15,
                    occlusion_strength=1.0):
    """Return a list of visibility values indexed by ``face.index``.

    ``bm`` must be in local space; ``matrix_world`` transforms it to world space
    so 'down' and 'front' mean the same thing as in the viewport.
    ``front`` is a world-space direction vector or ``None``.

    Everything happens in world space: rays are cast against the transformed
    mesh and normals use the inverse-transpose of ``matrix_world``, so
    unapplied rotation and (non-uniform or negative) scale give the same
    result as applying the transform first. Sampling is a fixed Fibonacci
    pattern, so the result is deterministic between runs.

    Cost is one BVH ray cast per (face, upper-hemisphere direction) that the
    ground plane doesn't already block; ~90k faces at 32 samples take about
    3 s. The rest of the solver dominates the run time.
    """
    if not bm.faces:
        return []
    bm.verts.index_update()
    bm.faces.index_update()
    bm.faces.ensure_lookup_table()
    normal_mat = matrix_world.to_3x3().inverted_safe().transposed()

    world_co = [matrix_world @ v.co for v in bm.verts]
    polys = [[v.index for v in f.verts] for f in bm.faces]
    tree = BVHTree.FromPolygons(world_co, polys, all_triangles=False)

    lo = Vector((min(c.x for c in world_co), min(c.y for c in world_co), min(c.z for c in world_co)))
    hi = Vector((max(c.x for c in world_co), max(c.y for c in world_co), max(c.z for c in world_co)))
    diag = max((hi - lo).length, 1e-6)
    ground_z = lo.z
    max_dist = diag * 0.75
    eps = diag * 1e-4

    dirs = _hemisphere_dirs(max(1, int(samples)) * 2)
    up = Vector((0.0, 0.0, 1.0))
    result = [1.0] * len(bm.faces)

    for f in bm.faces:
        if f.index % 128 == 0:
            keepalive.pulse()
        n = (normal_mat @ f.normal).normalized()
        c = matrix_world @ f.calc_center_median()
        origin = c + n * eps

        # --- openness (cosine weighted) --------------------------------
        open_w = total_w = 0.0
        for d in dirs:
            cos_t = d.dot(n)
            if cos_t <= 0.0:
                continue
            total_w += cos_t
            blocked = False
            if use_ground and d.z < 0.0:
                # ray hits the ground plane before travelling max_dist?
                t = (ground_z - origin.z) / d.z
                if 0.0 <= t < max_dist:
                    blocked = True
            if not blocked:
                hit, _, _, _ = tree.ray_cast(origin, d, max_dist)
                blocked = hit is not None
            if not blocked:
                open_w += cos_t
        openness = open_w / total_w if total_w else 1.0
        # remap: a fully open face = 1, a face open only half way is still
        # quite visible, so soften with a power curve.
        openness = 1.0 - occlusion_strength * (1.0 - openness ** 0.75)

        # --- facing ------------------------------------------------------
        dz = n.dot(up)
        if dz < -0.2:
            # linearly fade from 1 at dz=-0.2 to down_weight at dz=-1
            t = (-dz - 0.2) / 0.8
            facing = 1.0 + (down_weight - 1.0) * t
        else:
            facing = 1.0
        if front is not None:
            fd = n.dot(front)
            if fd < 0.0:
                facing *= 1.0 + (back_weight - 1.0) * (-fd)

        result[f.index] = max(0.0, min(1.0, openness * facing))
    return result
