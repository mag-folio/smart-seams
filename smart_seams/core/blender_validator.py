"""Validate candidate UV shells with Blender's own unwrapper.

Using the same solver (ABF / LSCM in C) as the final unwrap makes validation
both fast and exact: what passes here is what the user gets.

A hidden temporary mesh object is kept in Edit Mode for the whole solve; for
each candidate its geometry is replaced by the candidate faces (with seams on
the unglued edges), unwrapped, and measured.

The solver often asks about the same candidate again (for example when the
stretch limit is relaxed and a refused merge is retried), so measurements are
cached per exact input (ordered faces + seams). The unwrapper is
deterministic, so a cached answer is identical to a fresh one.
"""

import math

import bmesh
import bpy
import numpy as np

from .flatten import boundary_crossings

TMP_NAME = "__smart_seams_validator__"

# Optional ``callable(calls)`` invoked after every trial unwrap. The add-on's
# operators set it to drive the progress indicator on slow objects.
on_call = None


def _is_tmp(idblock):
    try:
        return idblock.name.startswith(TMP_NAME)
    except ReferenceError:
        return False


def cleanup():
    """Remove any temporary validator objects/meshes (e.g. left by a crash).

    Returns the number of data-blocks removed.
    """
    removed = 0
    for ob in [o for o in bpy.data.objects if _is_tmp(o)]:
        bpy.data.objects.remove(ob, do_unlink=True)
        removed += 1
    for me in [m for m in bpy.data.meshes if _is_tmp(m) and m.users == 0]:
        bpy.data.meshes.remove(me)
        removed += 1
    return removed


class BlenderValidator:
    def __init__(self, method='ANGLE_BASED'):
        self.method = method
        self.calls = 0          # validation requests
        self.unwraps = 0        # actual trial unwraps (cache misses)
        self.obj = None
        self._cache = {}

    def __enter__(self):
        ctx = bpy.context
        self._prev_active = ctx.view_layer.objects.active
        self._prev_selected = list(ctx.selected_objects)
        if ctx.object is not None and ctx.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        cleanup()
        try:
            me = bpy.data.meshes.new(TMP_NAME)
            self.obj = bpy.data.objects.new(TMP_NAME, me)
            self.obj.hide_render = True
            ctx.scene.collection.objects.link(self.obj)
            me.uv_layers.new(name="UVMap")
            for o in ctx.view_layer.objects:
                o.select_set(False)
            self.obj.select_set(True)
            ctx.view_layer.objects.active = self.obj
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        ctx = bpy.context
        try:
            if self.obj is not None and self.obj.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except (RuntimeError, ReferenceError):
            pass
        if self.obj is not None:
            try:
                me = self.obj.data
                bpy.data.objects.remove(self.obj, do_unlink=True)
                if me.users == 0:
                    bpy.data.meshes.remove(me)
            except ReferenceError:
                pass
            self.obj = None
        cleanup()
        for o in self._prev_selected:
            try:
                o.select_set(True)
            except (ReferenceError, RuntimeError):
                pass
        try:
            ctx.view_layer.objects.active = self._prev_active
        except (ReferenceError, RuntimeError):
            pass
        self._cache.clear()
        return False

    def __call__(self, faces, glued, limit):
        self.calls += 1
        fset = {f.index for f in faces}
        seam_ids = []
        for f in faces:
            for e in f.edges:
                lf = e.link_faces
                if len(lf) == 2 and lf[0].index in fset and lf[1].index in fset and not glued[e.index]:
                    seam_ids.append(e.index)
        key = (tuple(f.index for f in faces), frozenset(seam_ids))
        cached = self._cache.get(key)
        if cached is not None:
            ok, dist, crossings_ok = cached
            if not ok or dist > limit:
                self._notify()
                return False
            if crossings_ok is not None:
                self._notify()
                return crossings_ok
        result = self._unwrap_and_measure(faces, set(seam_ids), limit, key)
        self._notify()
        return result

    def _notify(self):
        if on_call is not None:
            try:
                on_call(self.calls)
            except Exception:  # progress display must never break the solve
                pass

    def _unwrap_and_measure(self, faces, seam_ids, limit, key):
        self.unwraps += 1
        me = self.obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.clear()
        vmap = {}
        for f in faces:
            for v in f.verts:
                if v.index not in vmap:
                    vmap[v.index] = bm.verts.new(v.co)
        new_faces = []
        for f in faces:
            try:
                nf = bm.faces.new([vmap[v.index] for v in f.verts])
            except ValueError:
                self._cache[key] = (False, math.inf, None)
                return False  # duplicate face: cannot validate, treat as bad
            new_faces.append(nf)
        # seams = unglued edges between two faces of this set
        if seam_ids:
            edge_lookup = {}
            for ne in bm.edges:
                edge_lookup[frozenset(v for v in ne.verts)] = ne
            for f in faces:
                for e in f.edges:
                    if e.index in seam_ids:
                        ne = edge_lookup.get(frozenset((vmap[e.verts[0].index], vmap[e.verts[1].index])))
                        if ne is not None:
                            ne.seam = True
        for nf in new_faces:
            nf.select = True
        bmesh.update_edit_mesh(me)
        try:
            bpy.ops.uv.unwrap(method=self.method, fill_holes=True, correct_aspect=True, margin=0.0)
        except RuntimeError:
            return False
        bm = bmesh.from_edit_mesh(me)
        uvl = bm.loops.layers.uv.active
        ok, dist, uv, T = self._measure(bm, uvl)
        if not ok or dist > limit:
            self._cache[key] = (ok, dist, None)
            return False
        crossings_ok = boundary_crossings(uv, T) == 0
        self._cache[key] = (ok, dist, crossings_ok)
        return crossings_ok

    @staticmethod
    def _measure(bm, uvl):
        """Return ``(ok, stretch, uv, tris)``; ``ok`` is False for degenerate
        or flipped results."""
        # triangles with 3D area and UV coordinates; UV vertices keyed by
        # (vertex, rounded uv) so seams split them
        key_id = {}
        uv_pts = []
        tris = []
        area3 = []
        for f in bm.faces:
            loops = list(f.loops)
            ids = []
            for l in loops:
                uv = l[uvl].uv
                k = (l.vert.index, round(uv.x, 7), round(uv.y, 7))
                if k not in key_id:
                    key_id[k] = len(uv_pts)
                    uv_pts.append((uv.x, uv.y))
                ids.append(key_id[k])
            for i in range(1, len(loops) - 1):
                p0, p1, p2 = loops[0].vert.co, loops[i].vert.co, loops[i + 1].vert.co
                a = (p1 - p0).cross(p2 - p0).length * 0.5
                if a <= 1e-14:
                    continue
                tris.append((ids[0], ids[i], ids[i + 1]))
                area3.append(a)
        if not tris:
            return False, math.inf, None, None
        uv = np.array(uv_pts)
        T = np.array(tris)
        A3 = np.array(area3)
        a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
        A2 = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        tot2, tot3 = np.abs(A2).sum(), A3.sum()
        if tot2 <= 1e-14:
            return False, math.inf, None, None
        sign = 1.0 if A2.sum() >= 0 else -1.0
        if A3[(A2 * sign) <= 0].sum() / tot3 > 0.001:
            return False, math.inf, None, None
        r = np.abs(A2) / (A3 * tot2 / tot3)
        dist = float((A3 * np.abs(np.log(np.clip(r, 1e-6, None)))).sum() / tot3)
        if not math.isfinite(dist):
            return False, math.inf, None, None
        return True, dist, uv, T
