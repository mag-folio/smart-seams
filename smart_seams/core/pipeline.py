"""End-to-end: visibility -> seams -> unwrap -> straighten -> scale -> pack."""

import time

import bmesh
import bpy
from mathutils import Vector

from . import layout, metrics
from . import panels as panels_mod
from .blender_validator import BlenderValidator
from .visibility import face_visibility

FRONT_AXES = {
    'NONE': None,
    'NEG_Y': Vector((0, -1, 0)), 'POS_Y': Vector((0, 1, 0)),
    'NEG_X': Vector((-1, 0, 0)), 'POS_X': Vector((1, 0, 0)),
}


class Options:
    """Plain settings object (the add-on's PropertyGroup maps onto this)."""

    def __init__(self, **kw):
        self.panel_angle = 40.0      # edges sharper than this separate panels
        self.max_islands = 10        # hard limit on UV shells
        self.max_distortion = 0.2    # stretch ceiling the solver may relax to
        self.front = 'NONE'          # NONE, CAMERA, NEG_Y, POS_Y, NEG_X, POS_X
        self.use_ground = True
        self.samples = 48
        self.min_hidden_scale = 0.5
        self.unwrap_method = 'ANGLE_BASED'
        self.straighten = True
        self.margin = 0.01
        self.mark_seams_only = False
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)


def _front_vector(obj, opts):
    if opts.front == 'CAMERA':
        cam = bpy.context.scene.camera
        if cam is None:
            return None
        centre = obj.matrix_world @ (sum((Vector(b) for b in obj.bound_box), Vector()) / 8)
        return (cam.matrix_world.translation - centre).normalized()
    return FRONT_AXES.get(opts.front)


def _ensure_edit_mode(obj):
    view_layer = bpy.context.view_layer
    for o in view_layer.objects:
        o.select_set(o is obj)
    view_layer.objects.active = obj
    if obj.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')


def solve_seams(obj, opts):
    """Compute visibility + shells and mark seams. Returns (shells, visibility, stats)."""
    if obj.mode != 'OBJECT':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.index_update(); bm.edges.index_update(); bm.faces.index_update()

    vis = face_visibility(bm, obj.matrix_world, samples=opts.samples,
                          use_ground=opts.use_ground, front=_front_vector(obj, opts))
    steps = tuple(d for d in (0.08, 0.12, 0.2, 0.3, 0.45) if d < opts.max_distortion) + (opts.max_distortion,)
    settings = panels_mod.PanelSettings(panel_angle=opts.panel_angle,
                                        max_shells=opts.max_islands,
                                        stretch_steps=steps)
    with BlenderValidator(method=opts.unwrap_method) as validator:
        chart_of_face, seams = panels_mod.compute_shells(bm, vis, settings, validator)
    stats = dict(panels_mod.LAST_STATS)
    bm.free()

    _ensure_edit_mode(obj)
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    seam_set = set(seams)
    for e in bm.edges:
        e.seam = e.index in seam_set
    bmesh.update_edit_mesh(obj.data)

    groups = {}
    for fi, cid in enumerate(chart_of_face):
        groups.setdefault(cid, []).append(fi)
    return list(groups.values()), vis, stats


def unwrap(obj, opts=None):
    """Run the whole pipeline on ``obj``. Returns a report dict."""
    opts = opts or Options()
    t0 = time.perf_counter()
    prev_mode = obj.mode
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")

    charts, vis, stats = solve_seams(obj, opts)
    report = {"charts": len(charts), "solver": stats}

    if not opts.mark_seams_only:
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(me)
        bpy.ops.uv.unwrap(method=opts.unwrap_method, fill_holes=True,
                          correct_aspect=True, margin=0.0)
        if opts.straighten:
            report["straightened"] = layout.straighten_quad_islands(obj, charts)
        aligned = [charts[i] for i in layout.align_to_grid(obj, charts)]
        report["grid_aligned"] = len(aligned)
        layout.scale_by_visibility(obj, charts, vis, min_scale=opts.min_hidden_scale)

        bm = bmesh.from_edit_mesh(me)
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(me)
        # grid-aligned shells keep their rotation (pinned); the packer may
        # still turn irregular shells to fit tighter
        layout.set_pinned(obj, aligned, True)
        try:
            bpy.ops.uv.pack_islands(rotate=True, rotate_method='AXIS_ALIGNED', scale=True,
                                    margin_method='FRACTION', margin=opts.margin,
                                    shape_method='CONCAVE', pin=True, pin_method='ROTATION')
        finally:
            layout.set_pinned(obj, aligned, False)

        bm = bmesh.from_edit_mesh(me)
        report.update(metrics.analyze(bm, bm.loops.layers.uv.active, vis))

    report["seconds"] = round(time.perf_counter() - t0, 2)
    if prev_mode != 'EDIT':
        bpy.ops.object.mode_set(mode=prev_mode)
    return report


def analyze_object(obj, opts=None):
    """Metrics for the object's current UVs (no changes)."""
    opts = opts or Options()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    vis = face_visibility(bm, obj.matrix_world, samples=opts.samples,
                          use_ground=opts.use_ground, front=_front_vector(obj, opts))
    uvl = bm.loops.layers.uv.active
    out = metrics.analyze(bm, uvl, vis) if uvl else {"islands": 0}
    bm.free()
    return out
