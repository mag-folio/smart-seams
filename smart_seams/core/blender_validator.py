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

from . import keepalive
from .flatten import boundary_crossings, face_triangles

TMP_NAME = "__smart_seams_validator__"


# ``bpy.ops`` refreshes the whole view layer before and after every operator
# call. The trial unwraps only touch the hidden temporary mesh in Edit Mode
# (its BMesh is updated by hand), so that refresh is pure overhead, and in a
# big scene it costs more than the unwrap itself (965 objects: 14 ms a call,
# two calls per trial). Call the operator directly when Blender allows it.
try:
    from _bpy import ops as _bpy_ops
    _op_call = _bpy_ops.call
except (ImportError, AttributeError):
    _op_call = None


def call_op(idname, **kw):
    """``bpy.ops.<idname>(**kw)`` without the view layer refresh (see above).

    Only for operators that work on the edit mesh and need no scene update:
    the trial unwraps and the strip straightening.
    """
    global _op_call
    if _op_call is not None:
        try:
            return _op_call(idname, kw)
        except TypeError:  # signature changed in this Blender: use the normal path
            _op_call = None
    mod, name = idname.split(".")
    return getattr(getattr(bpy.ops, mod), name)(**kw)


def _trial_unwrap(**kw):
    """``bpy.ops.uv.unwrap(**kw)`` without the view layer refresh."""
    return call_op("uv.unwrap", **kw)


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
    if removed:
        try:
            bpy.context.view_layer.update()  # drop stale (None) object entries
        except (RuntimeError, AttributeError):
            pass
    return removed


def _triangles(f, loops):
    """Corner triangles of a face for measuring.

    Convex faces keep the plain fan from corner 0 (the established
    measurement). A fan over a concave n-gon has triangles outside the face
    with reversed orientation, which would look folded and make every merge
    with that face fail, so those faces use a proper triangulation.

    Quads keep the fan even when concave: Blender's unwrapper splits a quad
    along its own diagonal, and measuring a concave quad with the 3D
    triangulation instead refuses merges the fan accepts (the test table,
    which has 25 strongly concave quads, goes from 10 to 13 shells).
    """
    n = len(loops)
    fan = [(0, i, i + 1) for i in range(1, n - 1)]
    if n > 4:
        f.normal_update()  # faces of the temporary mesh have no normal yet
        nrm = f.normal
        p0 = loops[0].vert.co
        for _a, b, c in fan:
            if (loops[b].vert.co - p0).cross(loops[c].vert.co - p0).dot(nrm) < -1e-12:
                return face_triangles(f)
    return fan


class BlenderValidator:
    def __init__(self, method='ANGLE_BASED'):
        self.method = method
        self.calls = 0          # validation requests
        self.unwraps = 0        # actual trial unwraps (cache misses)
        self.obj = None
        self._cache = {}
        self._local = {}        # local_check results per exact input
        self.local_unwraps = 0  # small patch unwraps done by local_check
        self.shell_stretch = {}  # faces that passed -> their stretch (merge headroom)

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
        # removing objects through bpy.data can leave stale (None) entries in
        # view_layer.objects until the layer is refreshed
        try:
            ctx.view_layer.update()
        except RuntimeError:
            pass
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
        self._local.clear()
        self.shell_stretch.clear()
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
                if crossings_ok:
                    self.shell_stretch[frozenset(fset)] = dist
                self._notify()
                return crossings_ok
        result = self._unwrap_and_measure(faces, set(seam_ids), limit, key)
        if result:
            self.shell_stretch[frozenset(fset)] = self._cache[key][1]
        self._notify()
        return result

    def _notify(self):
        keepalive.pulse()
        if on_call is not None:
            try:
                on_call(self.calls)
            except Exception:  # progress display must never break the solve
                pass

    def _unwrap_and_measure(self, faces, seam_ids, limit, key):
        self.unwraps += 1
        if self._trial_mesh(faces, seam_ids) is None:
            self._cache[key] = (False, math.inf, None)
            return False  # duplicate face: cannot validate, treat as bad
        try:
            _trial_unwrap(method=self.method, fill_holes=True, correct_aspect=True, margin=0.0)
        except RuntimeError:
            return False
        bm = bmesh.from_edit_mesh(self.obj.data)
        uvl = bm.loops.layers.uv.active
        ok, dist, uv, T = self._measure(bm, uvl)
        if not ok or dist > limit:
            self._cache[key] = (ok, dist, None)
            return False
        crossings_ok = boundary_crossings(uv, T) == 0
        self._cache[key] = (ok, dist, crossings_ok)
        return crossings_ok

    def _trial_mesh(self, faces, seam_ids):
        """Replace the temporary mesh by ``faces`` (seams on ``seam_ids``),
        all selected. Returns its BMesh, or None if a face is a duplicate."""
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
                return None
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
        return bm

    def local_check(self, faces, glued, piece, limit):
        """Pre-check for joining a small piece to a very big shell. It can
        only *reject* a join, never accept one.

        ``faces`` is the piece plus the part of the big shell around it (a
        few face rings), with the join glued. The patch is unwrapped like any
        trial; the join is refused if the patch folds over or overlaps
        itself, or if the piece's own faces (``piece``, a set of face
        indices) stretch past ``limit`` measured against the patch around
        them. Such a join would re-unwrap thousands of faces only to fail, or
        to pass by hiding the piece's stretch in the big shell's average.
        Returns False to refuse; True means "now run the full trial".
        """
        fset = {f.index for f in faces}
        seam_ids = set()
        for f in faces:
            for e in f.edges:
                lf = e.link_faces
                if len(lf) == 2 and lf[0].index in fset and lf[1].index in fset and not glued[e.index]:
                    seam_ids.add(e.index)
        key = (tuple(f.index for f in faces), frozenset(seam_ids), frozenset(piece))
        cached = self._local.get(key)
        if cached is None:
            cached = self._local[key] = self._local_measure(faces, seam_ids, piece)
            self.local_unwraps += 1
            self._notify()
        ok, stretch = cached
        return ok and stretch <= limit

    def _local_measure(self, faces, seam_ids, piece):
        """``(no flips and no overlap, stretch of the piece's faces)``."""
        if self._trial_mesh(faces, seam_ids) is None:
            return False, math.inf
        try:
            _trial_unwrap(method=self.method, fill_holes=True, correct_aspect=True, margin=0.0)
        except RuntimeError:
            return False, math.inf
        bm = bmesh.from_edit_mesh(self.obj.data)
        uvl = bm.loops.layers.uv.active
        # the temporary mesh's faces are in the order of ``faces``
        mark = [f.index in piece for f in faces]
        ok, _dist, uv, T, part = self._measure(bm, uvl, mark)
        if not ok or boundary_crossings(uv, T) != 0:
            return False, math.inf
        return True, part

    @staticmethod
    def _measure(bm, uvl, mark=None):
        """Return ``(ok, stretch, uv, tris)``; ``ok`` is False for degenerate
        or flipped results.

        With ``mark`` (one bool per face, in face order) a fifth value is
        returned: the stretch of the marked faces alone, measured against the
        scale of the whole unwrap."""
        # triangles with 3D area and UV coordinates; UV vertices keyed by
        # (vertex, rounded uv) so seams split them
        key_id = {}
        uv_pts = []
        tris = []
        area3 = []
        marked = []
        bad = (False, math.inf, None, None) if mark is None else (False, math.inf, None, None, math.inf)
        for fi, f in enumerate(bm.faces):
            loops = list(f.loops)
            ids = []
            for l in loops:
                uv = l[uvl].uv
                k = (l.vert.index, round(uv.x, 7), round(uv.y, 7))
                if k not in key_id:
                    key_id[k] = len(uv_pts)
                    uv_pts.append((uv.x, uv.y))
                ids.append(key_id[k])
            for i0, i1, i2 in _triangles(f, loops):
                p0, p1, p2 = loops[i0].vert.co, loops[i1].vert.co, loops[i2].vert.co
                a = (p1 - p0).cross(p2 - p0).length * 0.5
                if a <= 1e-14:
                    continue
                tris.append((ids[i0], ids[i1], ids[i2]))
                area3.append(a)
                if mark is not None:
                    marked.append(mark[fi])
        if not tris:
            return bad
        uv = np.array(uv_pts)
        T = np.array(tris)
        A3 = np.array(area3)
        a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
        A2 = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        tot2, tot3 = np.abs(A2).sum(), A3.sum()
        if tot2 <= 1e-14:
            return bad
        sign = 1.0 if A2.sum() >= 0 else -1.0
        if A3[(A2 * sign) <= 0].sum() / tot3 > 0.001:
            return bad
        r = np.abs(A2) / (A3 * tot2 / tot3)
        logs = A3 * np.abs(np.log(np.clip(r, 1e-6, None)))
        dist = float(logs.sum() / tot3)
        if not math.isfinite(dist):
            return bad
        if mark is None:
            return True, dist, uv, T
        m = np.array(marked, bool)
        part = float(logs[m].sum() / A3[m].sum()) if m.any() else 0.0
        return True, dist, uv, T, part
