"""Robustness test: run the operators the way real users do, on awkward input.

    blender -b test_data/uv_test.blend --factory-startup --python scripts/test_robustness.py

Every case must either finish or fail with a clear message (never a Python
traceback), leave the user in the mode / selection they started with, keep
the component selection, and never leak the temporary validator object.
Exits with code 1 if any case fails.
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Euler, Vector  # noqa: E402

import smart_seams  # noqa: E402
from smart_seams.core import blender_validator  # noqa: E402

FAILS = []
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ helpers

def scene():
    return bpy.context.scene


def view_layer():
    return bpy.context.view_layer


def settings():
    return scene().smart_seams


def object_mode():
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def select(objs, active=None):
    object_mode()
    for o in view_layer().objects:
        o.select_set(o in objs)
    view_layer().objects.active = active


def copy_of(name, new_name, linked=False):
    src = bpy.data.objects[name]
    ob = src.copy()
    if not linked:
        ob.data = src.data.copy()
    ob.name = new_name
    scene().collection.objects.link(ob)
    return ob


def from_pydata(name, verts, edges=(), faces=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata([Vector(v) for v in verts], list(edges), [list(f) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    scene().collection.objects.link(ob)
    return ob


def grid_box(name, size=1.0, loc=(0, 0, 0)):
    """A subdivided box (24 quads) built with bmesh."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=bm.verts[:])
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    scene().collection.objects.link(ob)
    return ob


def remove(*objs):
    object_mode()
    for ob in objs:
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if isinstance(me, bpy.types.Mesh) and me.users == 0:
            bpy.data.meshes.remove(me)


def leaked():
    names = [o.name for o in bpy.data.objects if o.name.startswith(blender_validator.TMP_NAME)]
    names += [m.name for m in bpy.data.meshes if m.name.startswith(blender_validator.TMP_NAME)]
    return names


def run(op_name, expect=('FINISHED',), allow_error=False):
    """Call an operator; return (result set or None, error text or None)."""
    op = getattr(bpy.ops.uv, op_name)
    try:
        res = op()
    except RuntimeError as ex:
        text = str(ex)
        if "Traceback" in text or "Python:" in text:
            raise AssertionError(f"{op_name} raised a Python traceback:\n{text}")
        if not allow_error:
            raise AssertionError(f"{op_name} failed unexpectedly: {text}")
        return None, text
    assert res & set(expect), f"{op_name} returned {res}, expected {expect}"
    return res, None


def snapshot():
    vl = view_layer()
    act = vl.objects.active
    return {
        "active": act.name if act else None,
        "mode": act.mode if act else 'OBJECT',
        "selected": sorted(o.name for o in vl.objects if o.select_get()),
        "in_mode": sorted(o.name for o in vl.objects if o.mode != 'OBJECT'),
    }


def check_state(before, what=""):
    after = snapshot()
    assert after == before, f"state changed {what}: {before} -> {after}"
    assert not leaked(), f"temporary validator data leaked: {leaked()}"


def report():
    return json.loads(settings().last_report or "{}")


def message():
    return getattr(settings(), "last_message", "")


def uv_bounds_ok(ob):
    """All UVs of the active layer lie inside the 0..1 square."""
    me = ob.data
    uvl = me.uv_layers.active
    co = [0.0] * (len(me.loops) * 2)
    uvl.data.foreach_get("uv", co)
    return all(-1e-4 <= c <= 1.0 + 1e-4 for c in co)


def mesh_selection(me):
    return ([v.select for v in me.vertices], [e.select for e in me.edges], [p.select for p in me.polygons])


# -------------------------------------------------------------------- cases

@case
def object_mode_single():
    ob = bpy.data.objects["cube"]
    select([ob], ob)
    before = snapshot()
    run("smart_seams_unwrap")
    check_state(before)
    r = report()["cube"]
    assert r["islands"] == 1 and r["uv_overlap_pct"] == 0.0, r
    assert "cube" in message() and "shell" in message(), message()


@case
def edit_mode_keeps_component_selection():
    ob = copy_of("cylinder", "rb_cyl_edit")
    try:
        select([ob], ob)
        me = ob.data
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bm = bmesh.from_edit_mesh(me)
        for f in bm.faces:
            f.select_set(False)
        for f in bm.faces:
            if f.index % 3 == 0:
                f.select_set(True)
        bmesh.update_edit_mesh(me)
        bpy.ops.object.mode_set(mode='OBJECT')
        sel_before = mesh_selection(me)
        bpy.ops.object.mode_set(mode='EDIT')
        before = snapshot()
        assert before["mode"] == 'EDIT'
        run("smart_seams_unwrap")
        check_state(before)
        bpy.ops.object.mode_set(mode='OBJECT')
        assert mesh_selection(me) == sel_before, "edit-mode component selection was not restored"
        assert report()["rb_cyl_edit"]["islands"] == 3
    finally:
        remove(ob)


@case
def multi_object_edit_mode():
    a = grid_box("rb_multi_a", loc=(5, 0, 0))
    b = grid_box("rb_multi_b", loc=(8, 0, 0))
    try:
        select([a, b], a)
        bpy.ops.object.mode_set(mode='EDIT')
        before = snapshot()
        assert before["in_mode"] == ["rb_multi_a", "rb_multi_b"], before
        run("smart_seams_unwrap")
        check_state(before, "(multi-object edit)")
        assert set(report()) == {"rb_multi_a", "rb_multi_b"}, report()
    finally:
        remove(a, b)


@case
def multiple_selected_with_non_mesh():
    a = grid_box("rb_sel_a", loc=(5, 5, 0))
    b = grid_box("rb_sel_b", loc=(8, 5, 0))
    empty = bpy.data.objects.new("rb_empty", None)
    scene().collection.objects.link(empty)
    cam_data = bpy.data.cameras.new("rb_cam")
    cam = bpy.data.objects.new("rb_cam", cam_data)
    scene().collection.objects.link(cam)
    try:
        # non-mesh active, meshes and an empty selected
        select([a, b, empty, cam], cam)
        before = snapshot()
        run("smart_seams_unwrap")
        check_state(before, "(non-mesh active)")
        assert set(report()) == {"rb_sel_a", "rb_sel_b"}, report()
        # only non-mesh objects: operator is unavailable (poll), not a crash
        select([empty, cam], cam)
        before = snapshot()
        _res, err = run("smart_seams_unwrap", allow_error=True)
        assert err and "poll" in err, err
        check_state(before)
    finally:
        remove(a, b, empty, cam)
        bpy.data.cameras.remove(cam_data)


@case
def no_selection():
    select([], None)
    before = snapshot()
    _res, err = run("smart_seams_unwrap", allow_error=True)
    assert err and "poll" in err, err
    _res, err = run("smart_seams_mark", allow_error=True)
    assert err and "poll" in err, err
    check_state(before)


@case
def unapplied_transforms():
    ob = grid_box("rb_xform", loc=(0, 8, 0))
    ob.scale = (3.0, 0.4, 1.5)
    ob.rotation_euler = Euler((0.3, 0.5, 1.1))
    ob.location = (0, 8, 2)
    neg = grid_box("rb_mirror", loc=(0, 12, 0))
    neg.scale = (-1.0, 1.0, 2.0)
    try:
        for o in (ob, neg):
            select([o], o)
            before = snapshot()
            run("smart_seams_unwrap")
            check_state(before)
            r = report()[o.name]
            assert r["uv_overlap_pct"] == 0.0 and r["flipped_faces"] == 0, r
            assert uv_bounds_ok(o), "UVs outside 0..1"
    finally:
        remove(ob, neg)


@case
def modifiers_untouched():
    ob = copy_of("cylinder", "rb_mods")
    try:
        ob.modifiers.new("Bevel", 'BEVEL')
        ob.modifiers.new("Subsurf", 'SUBSURF')
        n_faces = len(ob.data.polygons)
        select([ob], ob)
        before = snapshot()
        run("smart_seams_unwrap")
        check_state(before)
        assert [m.type for m in ob.modifiers] == ['BEVEL', 'SUBSURF']
        assert len(ob.data.polygons) == n_faces, "modifiers were applied"
        assert report()["rb_mods"]["islands"] == 3
    finally:
        remove(ob)


@case
def linked_duplicates():
    a = copy_of("cube", "rb_link_a")
    b = copy_of("rb_link_a", "rb_link_b", linked=True)
    b.location.x += 4
    b.scale = (1, 1, 3)
    try:
        assert a.data == b.data
        select([a, b], a)
        before = snapshot()
        run("smart_seams_unwrap")
        check_state(before)
        assert list(report()) == ["rb_link_a"], report()
        assert "shares its mesh" in message(), message()
    finally:
        remove(b)
        remove(a)


@case
def mesh_without_faces():
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]
    wire = from_pydata("rb_wire", verts, edges=[(0, 1), (1, 2), (2, 3)])
    empty_mesh = from_pydata("rb_empty_mesh", [])
    try:
        for ob in (wire, empty_mesh):
            for op in ("smart_seams_unwrap", "smart_seams_mark", "smart_seams_analyze"):
                select([ob], ob)
                before = snapshot()
                _res, err = run(op, allow_error=True)
                assert err and "no faces" in err, (op, err)
                check_state(before)
        # a faceless mesh next to a good one: the good one is still done
        good = grid_box("rb_good", loc=(-6, 0, 0))
        try:
            select([wire, good], good)
            before = snapshot()
            run("smart_seams_unwrap")
            check_state(before)
            assert list(report()) == ["rb_good"] and "no faces" in message(), (report(), message())
        finally:
            remove(good)
    finally:
        remove(wire, empty_mesh)


@case
def messy_topology():
    """Loose verts/edges, an n-gon, a non-manifold fin and disconnected pieces."""
    verts = [
        # piece 1: a quad strip + a fin sharing edge (1, 2)
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0), (1, 0.5, 1),
        # piece 2: an n-gon (hexagon) far away
        (5, 0, 0), (6, 0, 0), (6.5, 0.8, 0), (6, 1.6, 0), (5, 1.6, 0), (4.5, 0.8, 0),
        # loose verts and a loose edge
        (9, 9, 9), (10, 9, 9), (10, 10, 9),
    ]
    faces = [(0, 1, 2, 3), (1, 4, 5, 2), (1, 2, 6), (7, 8, 9, 10, 11, 12)]
    ob = from_pydata("rb_messy", verts, edges=[(14, 15)], faces=faces)
    try:
        for op in ("smart_seams_mark", "smart_seams_unwrap", "smart_seams_analyze"):
            select([ob], ob)
            before = snapshot()
            run(op)
            check_state(before)
        r = report()["rb_messy"]
        assert r["uv_overlap_pct"] == 0.0, r
    finally:
        remove(ob)


@case
def degenerate_faces():
    """Zero-area faces and doubled vertices."""
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 0, 0), (3, 0, 0)]
    faces = [(0, 1, 2, 3), (1, 4, 2), (4, 5, 6)]  # last face is a sliver of zero area
    ob = from_pydata("rb_degenerate", verts, faces=faces)
    try:
        select([ob], ob)
        before = snapshot()
        run("smart_seams_unwrap", allow_error=True)
        check_state(before)
    finally:
        remove(ob)


@case
def uv_layers():
    none = grid_box("rb_no_uv", loc=(0, -6, 0))
    for layer in list(none.data.uv_layers):
        none.data.uv_layers.remove(layer)
    several = grid_box("rb_many_uv", loc=(4, -6, 0))
    me = several.data
    while len(me.uv_layers) < 3:
        me.uv_layers.new(name=f"Layer{len(me.uv_layers)}")
    me.uv_layers.active_index = 1
    active_name = me.uv_layers.active.name
    others = {l.name: [tuple(d.uv) for d in l.data] for l in me.uv_layers if l.name != active_name}
    try:
        # analyze without UVs is a message, not an error
        select([none], none)
        before = snapshot()
        run("smart_seams_analyze")
        check_state(before)
        assert "no UV map" in message(), message()
        run("smart_seams_unwrap")
        check_state(before)
        assert len(none.data.uv_layers) == 1, "a UV map should have been created"

        select([several], several)
        before = snapshot()
        run("smart_seams_unwrap")
        check_state(before)
        assert len(me.uv_layers) == 3 and me.uv_layers.active.name == active_name
        for l in me.uv_layers:
            if l.name in others:
                assert [tuple(d.uv) for d in l.data] == others[l.name], f"inactive UV map {l.name} changed"
        assert f"'{active_name}'" in message(), message()
    finally:
        remove(none, several)


@case
def hidden_objects():
    ob = grid_box("rb_hidden", loc=(0, -10, 0))
    vis = grid_box("rb_visible", loc=(4, -10, 0))
    try:
        select([ob], ob)
        ob.hide_set(True)
        before = snapshot()
        _res, err = run("smart_seams_unwrap", allow_error=True)
        assert err and "hidden" in err, err
        check_state(before)
        # hidden active object, visible mesh selected
        select([vis], ob)
        ob.hide_set(True)
        before = snapshot()
        run("smart_seams_unwrap")
        check_state(before)
        assert list(report()) == ["rb_visible"], report()
    finally:
        ob.hide_set(False)
        remove(ob, vis)


@case
def hidden_faces_in_edit_mode():
    ob = grid_box("rb_hidden_faces", loc=(0, -14, 0))
    try:
        select([ob], ob)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for p in ob.data.polygons[:6]:
            p.select = True
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.hide(unselected=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        hidden = [p.hide for p in ob.data.polygons]
        assert sum(hidden) == 6
        bpy.ops.object.mode_set(mode='EDIT')
        before = snapshot()
        run("smart_seams_unwrap")
        check_state(before)
        bpy.ops.object.mode_set(mode='OBJECT')
        assert [p.hide for p in ob.data.polygons] == hidden, "hidden faces were not restored"
        r = report()["rb_hidden_faces"]
        assert r["uv_overlap_pct"] == 0.0, r
    finally:
        remove(ob)


@case
def other_modes():
    ob = grid_box("rb_modes", loc=(0, -18, 0))
    try:
        for mode in ('SCULPT', 'VERTEX_PAINT', 'WEIGHT_PAINT'):
            select([ob], ob)
            bpy.ops.object.mode_set(mode=mode)
            before = snapshot()
            assert before["mode"] == mode
            run("smart_seams_mark")
            check_state(before, f"({mode})")
    finally:
        remove(ob)


@case
def mark_reports_overwritten_seams():
    ob = grid_box("rb_seams", loc=(0, -22, 0))
    try:
        for e in list(ob.data.edges)[:5]:
            e.use_seam = True
        select([ob], ob)
        before = snapshot()
        run("smart_seams_mark")
        check_state(before)
        assert "replaced 5 existing seam edges" in message(), message()
        seams = sum(e.use_seam for e in ob.data.edges)
        assert seams > 0, "no seams marked"
        # UVs are left alone by Mark
        run("smart_seams_analyze")
    finally:
        remove(ob)


@case
def undo_restores():
    ob = grid_box("rb_undo", loc=(0, -26, 0))
    name = ob.name
    if not ob.data.uv_layers:
        ob.data.uv_layers.new(name="UVMap")
    try:
        select([ob], ob)
        bpy.ops.ed.undo_push(message="before smart seams")
        uv_before = [tuple(d.uv) for d in ob.data.uv_layers.active.data]
        seams_before = [e.use_seam for e in ob.data.edges]
        run("smart_seams_unwrap")
        assert [tuple(d.uv) for d in bpy.data.objects[name].data.uv_layers.active.data] != uv_before
        try:
            bpy.ops.ed.undo()
        except RuntimeError as ex:
            print(f"  (undo not available in this session: {ex}; skipped)")
            return
        ob = bpy.data.objects[name]
        assert [tuple(d.uv) for d in ob.data.uv_layers.active.data] == uv_before, "undo did not restore UVs"
        assert [e.use_seam for e in ob.data.edges] == seams_before, "undo did not restore seams"
        assert not leaked(), leaked()
    finally:
        if name in bpy.data.objects:
            remove(bpy.data.objects[name])


@case
def stale_validator_object_is_cleaned():
    me = bpy.data.meshes.new(blender_validator.TMP_NAME)
    stale = bpy.data.objects.new(blender_validator.TMP_NAME, me)
    scene().collection.objects.link(stale)
    ob = bpy.data.objects["cube"]
    select([ob], ob)
    before = snapshot()
    run("smart_seams_mark")
    check_state(before)


# ---------------------------------------------------------------------- run

def main():
    smart_seams.register()
    t_all = time.perf_counter()
    try:
        settings().samples = 16
        for fn in CASES:
            t0 = time.perf_counter()
            try:
                fn()
                print(f"ROBUST ok    {fn.__name__} ({time.perf_counter() - t0:.1f} s)  {message()[:160]}")
            except Exception:  # noqa: BLE001
                FAILS.append(fn.__name__)
                print(f"ROBUST FAIL  {fn.__name__}")
                traceback.print_exc()
                try:
                    object_mode()
                except Exception:  # noqa: BLE001
                    pass
    finally:
        smart_seams.unregister()
    print(f"ROBUST {len(CASES) - len(FAILS)}/{len(CASES)} passed in {time.perf_counter() - t_all:.1f} s"
          f" on Blender {bpy.app.version_string}")
    if FAILS:
        print("ROBUST FAILED:", ", ".join(FAILS))
        sys.stdout.flush()
        os._exit(1)


main()
