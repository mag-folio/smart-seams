"""Smart Seams - clean, artist-style UV seams for Blender.

See README.md for the algorithm. The heavy lifting lives in ``core``; this
module only wires it to Blender's UI.
"""

import json

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty

from .core import pipeline

bl_info = {  # legacy add-on support (Blender < 4.2); extensions use blender_manifest.toml
    "name": "Smart Seams",
    "author": "Smart Seams contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport / UV Editor > Sidebar > Smart Seams",
    "description": "Automatic UV seams that follow edge loops, hide in low-visibility areas and avoid stretching",
    "category": "UV",
}


FRONT_ITEMS = [
    ('NONE', "None", "No preferred viewing side; only occlusion and down-facing matter"),
    ('CAMERA', "Scene Camera", "Faces turned away from the scene camera are de-prioritised"),
    ('NEG_Y', "-Y (Front view)", "The object is mostly seen from -Y"),
    ('POS_Y', "+Y", "The object is mostly seen from +Y"),
    ('NEG_X', "-X", "The object is mostly seen from -X"),
    ('POS_X', "+X", "The object is mostly seen from +X"),
]


class SMARTSEAMS_PG_settings(bpy.types.PropertyGroup):
    max_islands: IntProperty(
        name="Max Shells", default=10, min=1, soft_max=64,
        description="Hard limit on UV shells. Simple objects end up with far fewer (cube 1, cylinder 3); "
                    "complex ones use more only when needed to avoid stretch")
    max_distortion: FloatProperty(
        name="Stretch Ceiling", default=0.2, min=0.02, soft_max=0.5, precision=3,
        description="The solver starts strict (0.08) and only relaxes up to this to stay under Max Shells. "
                    "Overlap is never allowed")
    panel_angle: FloatProperty(
        name="Panel Angle", default=0.698, min=0.0, max=3.1416, subtype='ANGLE',
        description="Edges sharper than this separate panels; seams only ever run along complete "
                    "panel borders")
    front: EnumProperty(name="Front", items=FRONT_ITEMS, default='NONE',
                        description="Which way the object is usually seen from")
    use_ground: BoolProperty(name="Sits on Ground", default=True,
                             description="Treat the object's lowest point as a floor that hides faces")
    min_hidden_scale: FloatProperty(
        name="Hidden Shell Scale", default=0.5, min=0.05, max=1.0,
        description="Texel density of fully hidden islands relative to fully visible ones")
    straighten: BoolProperty(name="Straighten Strips", default=True,
                             description="Follow Active Quads on all-quad strips when it doesn't add stretch")
    unwrap_method: EnumProperty(
        name="Unwrap", default='ANGLE_BASED',
        items=[('ANGLE_BASED', "Angle Based", ""), ('CONFORMAL', "Conformal", ""),
               ('MINIMUM_STRETCH', "Minimum Stretch", "")])
    margin: FloatProperty(name="Margin", default=0.01, min=0.0, max=0.2, precision=3)
    samples: IntProperty(name="Visibility Samples", default=32, min=4, max=256,
                         description="Rays per face for the occlusion estimate")
    last_report: bpy.props.StringProperty(default="")

    def to_options(self, **overrides):
        import math
        o = pipeline.Options(
            max_islands=self.max_islands, max_distortion=self.max_distortion,
            panel_angle=math.degrees(self.panel_angle), front=self.front, use_ground=self.use_ground,
            min_hidden_scale=self.min_hidden_scale, straighten=self.straighten,
            unwrap_method=self.unwrap_method, margin=self.margin, samples=self.samples)
        for k, v in overrides.items():
            setattr(o, k, v)
        return o


def _mesh_targets(context):
    objs = [o for o in context.selected_objects if o.type == 'MESH']
    if not objs and context.active_object and context.active_object.type == 'MESH':
        objs = [context.active_object]
    return objs


def _run_each(context, fn):
    """Run ``fn(obj)`` on each target with only that object in edit mode."""
    targets = _mesh_targets(context)
    start_mode = context.mode
    active = context.view_layer.objects.active
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    reports = {}
    for ob in targets:
        for o in context.view_layer.objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        reports[ob.name] = fn(ob)
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    for ob in targets:
        ob.select_set(True)
    context.view_layer.objects.active = active
    if start_mode == 'EDIT_MESH' and active is not None:
        bpy.ops.object.mode_set(mode='EDIT')
    return reports


class SMARTSEAMS_OT_unwrap(bpy.types.Operator):
    """Solve seams, unwrap, straighten, scale hidden islands down and pack"""
    bl_idname = "uv.smart_seams_unwrap"
    bl_label = "Smart Seams Unwrap"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(_mesh_targets(context))

    def execute(self, context):
        s = context.scene.smart_seams
        reports = _run_each(context, lambda ob: pipeline.unwrap(ob, s.to_options()))
        s.last_report = json.dumps(reports)
        msg = ", ".join(f"{n}: {r.get('islands', r.get('charts'))} islands" for n, r in reports.items())
        self.report({'INFO'}, f"Smart Seams - {msg}")
        return {'FINISHED'}


class SMARTSEAMS_OT_mark(bpy.types.Operator):
    """Only solve and mark seams (keep the current UVs), so you can review or tweak them"""
    bl_idname = "uv.smart_seams_mark"
    bl_label = "Mark Smart Seams"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(_mesh_targets(context))

    def execute(self, context):
        s = context.scene.smart_seams
        reports = _run_each(context, lambda ob: pipeline.unwrap(ob, s.to_options(mark_seams_only=True)))
        msg = ", ".join(f"{n}: {r['charts']} islands" for n, r in reports.items())
        self.report({'INFO'}, f"Seams marked - {msg}")
        return {'FINISHED'}


class SMARTSEAMS_OT_analyze(bpy.types.Operator):
    """Measure the current UVs: islands, seam length/visibility, stretch, overlap"""
    bl_idname = "uv.smart_seams_analyze"
    bl_label = "Analyze UVs"

    @classmethod
    def poll(cls, context):
        return bool(_mesh_targets(context))

    def execute(self, context):
        s = context.scene.smart_seams
        reports = _run_each(context, lambda ob: pipeline.analyze_object(ob, s.to_options()))
        s.last_report = json.dumps(reports)
        for n, r in reports.items():
            print(f"[Smart Seams] {n}: {json.dumps(r)}")
        self.report({'INFO'}, "Analysis done (see the Smart Seams panel / system console)")
        return {'FINISHED'}


def _draw(layout, context):
    s = context.scene.smart_seams
    col = layout.column(align=True)
    col.scale_y = 1.3
    col.operator(SMARTSEAMS_OT_unwrap.bl_idname, icon='UV')
    row = layout.row(align=True)
    row.operator(SMARTSEAMS_OT_mark.bl_idname, icon='EDGESEL')
    row.operator(SMARTSEAMS_OT_analyze.bl_idname, icon='VIEWZOOM', text="Analyze")

    box = layout.box()
    box.label(text="Shells & Stretch")
    box.prop(s, "max_islands")
    box.prop(s, "max_distortion")
    box.prop(s, "panel_angle")

    box = layout.box()
    box.label(text="Visibility")
    box.prop(s, "front")
    box.prop(s, "use_ground")
    box.prop(s, "min_hidden_scale")
    box.prop(s, "samples")

    box = layout.box()
    box.label(text="Layout")
    box.prop(s, "unwrap_method")
    box.prop(s, "straighten")
    box.prop(s, "margin")

    if s.last_report:
        try:
            data = json.loads(s.last_report)
        except ValueError:
            data = {}
        box = layout.box()
        box.label(text="Last Result")
        for name, r in data.items():
            box.label(text=name, icon='MESH_DATA')
            for key in ("islands", "seam_length", "seam_visibility", "worst_island_area_distortion",
                        "uv_overlap_pct", "seconds"):
                if key in r and r[key] is not None:
                    box.label(text=f"   {key.replace('_', ' ')}: {r[key]}")


class SMARTSEAMS_PT_view3d(bpy.types.Panel):
    bl_label = "Smart Seams"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Smart Seams"

    def draw(self, context):
        _draw(self.layout, context)


class SMARTSEAMS_PT_uv(bpy.types.Panel):
    bl_label = "Smart Seams"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Smart Seams"

    def draw(self, context):
        _draw(self.layout, context)


def _menu(self, context):
    self.layout.separator()
    self.layout.operator(SMARTSEAMS_OT_unwrap.bl_idname)


CLASSES = (SMARTSEAMS_PG_settings, SMARTSEAMS_OT_unwrap, SMARTSEAMS_OT_mark,
           SMARTSEAMS_OT_analyze, SMARTSEAMS_PT_view3d, SMARTSEAMS_PT_uv)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.smart_seams = PointerProperty(type=SMARTSEAMS_PG_settings)
    bpy.types.VIEW3D_MT_uv_map.append(_menu)


def unregister():
    bpy.types.VIEW3D_MT_uv_map.remove(_menu)
    del bpy.types.Scene.smart_seams
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
