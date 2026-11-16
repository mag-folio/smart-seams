"""Smart Seams - clean, artist-style UV seams for Blender.

See README.md for the algorithm. The heavy lifting lives in ``core``; this
module only wires it to Blender's UI and makes sure a run always leaves the
user where they started: same mode, same object and component selection,
no temporary data behind, and a readable message instead of a traceback.
"""

import json
import math
import time
import traceback

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from .core import blender_validator, pipeline

bl_info = {  # legacy add-on support (Blender < 4.2); extensions use blender_manifest.toml
    "name": "Smart Seams",
    "author": "Smart Seams contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport / UV Editor > Sidebar > Smart Seams",
    "description": "Automatic UV seams that follow edge loops, hide in low-visibility areas and avoid stretching",
    "category": "UV",
}

# Above this many faces (all targets together) the operator asks before
# starting, since the solver can take minutes.
HEAVY_FACES = 20000


FRONT_ITEMS = [
    ('NONE', "None", "No preferred viewing side; only occlusion and down-facing matter"),
    ('CAMERA', "Scene Camera", "Faces turned away from the scene camera are de-prioritised"),
    ('NEG_Y', "-Y (Front view)", "The object is mostly seen from -Y"),
    ('POS_Y', "+Y", "The object is mostly seen from +Y"),
    ('NEG_X', "-X", "The object is mostly seen from -X"),
    ('POS_X', "+X", "The object is mostly seen from +X"),
]

UNWRAP_ITEMS = [
    ('ANGLE_BASED', "Angle Based",
     "Blender's ABF unwrapper; best shape preservation (recommended)"),
    ('CONFORMAL', "Conformal",
     "LSCM; faster, a little more stretch on curved shells"),
    ('MINIMUM_STRETCH', "Minimum Stretch",
     "Blender's SLIM unwrapper; least area distortion, slowest"),
]


if bpy.app.version < (4, 3, 0):  # Minimum Stretch (SLIM) arrived in Blender 4.3
    UNWRAP_ITEMS = UNWRAP_ITEMS[:2]


class SMARTSEAMS_PG_settings(bpy.types.PropertyGroup):
    max_islands: IntProperty(
        name="Max Shells", default=10, min=1, max=256, soft_max=64,
        description="Hard limit on UV shells. Simple objects end up with far fewer (cube 1, cylinder 3); "
                    "complex ones use more only when needed to avoid stretch")
    max_distortion: FloatProperty(
        name="Stretch Ceiling", default=0.2, min=0.02, max=1.0, soft_max=0.5, precision=3, step=1,
        description="The solver starts strict (0.08) and only relaxes up to this to stay under Max Shells. "
                    "Overlap is never allowed")
    panel_angle: FloatProperty(
        name="Panel Angle", default=0.698, min=math.radians(1.0), max=math.pi, soft_max=math.radians(90.0),
        subtype='ANGLE',
        description="Edges sharper than this separate panels; seams only ever run along complete "
                    "panel borders. Lower = more, smaller panels (slower)")
    front: EnumProperty(name="Front", items=FRONT_ITEMS, default='NONE',
                        description="Which way the object is usually seen from; seams move to the far side")
    use_ground: BoolProperty(name="Sits on Ground", default=True,
                             description="Treat the object's lowest point as a floor that hides the faces "
                                         "touching or facing it")
    min_hidden_scale: FloatProperty(
        name="Hidden Shell Scale", default=0.5, min=0.05, max=1.0, subtype='FACTOR',
        description="Texel density of fully hidden shells relative to fully visible ones "
                    "(1 = all shells get the same density)")
    straighten: BoolProperty(name="Straighten Strips", default=True,
                             description="Follow Active Quads on all-quad strips when it adds no stretch or overlap")
    unwrap_method: EnumProperty(
        name="Unwrap", items=UNWRAP_ITEMS, default='ANGLE_BASED',
        description="Blender unwrapper used both to test candidate shells and for the final unwrap")
    margin: FloatProperty(name="Margin", default=0.01, min=0.0, max=0.2, soft_max=0.05, precision=3, step=0.1,
                          description="Space between packed shells, as a fraction of the UV square")
    time_budget: FloatProperty(
        name="Time Budget", default=180.0, min=0.0, soft_max=1800.0, step=1000, precision=0,
        subtype='TIME_ABSOLUTE', unit='TIME_ABSOLUTE',
        description="Seconds per object for merging shells (0 = no limit). Very dense objects can "
                    "take hours to merge fully; when the time runs out the object keeps the shells "
                    "it has. Every shell is still checked for overlap, flips and stretch")
    samples: IntProperty(name="Visibility Samples", default=32, min=4, max=256, soft_max=128,
                         description="Rays per face for the occlusion estimate. More = smoother visibility, "
                                     "slower on dense meshes")
    last_report: StringProperty(default="", options={'HIDDEN'})
    last_message: StringProperty(default="", options={'HIDDEN'})

    def to_options(self, **overrides):
        o = pipeline.Options(
            max_islands=self.max_islands, max_distortion=self.max_distortion,
            panel_angle=math.degrees(self.panel_angle), front=self.front, use_ground=self.use_ground,
            min_hidden_scale=self.min_hidden_scale, straighten=self.straighten,
            unwrap_method=self.unwrap_method or 'ANGLE_BASED', margin=self.margin, samples=self.samples,
            time_budget=self.time_budget)
        for k, v in overrides.items():
            setattr(o, k, v)
        return o


# ---------------------------------------------------------------- targets

def _mesh_candidates(context):
    objs = [o for o in context.selected_objects if o.type == 'MESH']
    active = context.view_layer.objects.active
    if not objs and active is not None and active.type == 'MESH':
        objs = [active]
    return objs


def _is_editable(ob):
    for idb in (ob, ob.data):
        if idb.library is not None:
            return False
        if not getattr(idb, "is_editable", True):
            return False
    return True


def _collect_targets(context):
    """Return ``(targets, notes)``: the meshes to process (one object per
    mesh data-block, active first) and ``{name: reason}`` for skipped ones."""
    active = context.view_layer.objects.active
    cands = _mesh_candidates(context)
    cands.sort(key=lambda o: o is not active)  # stable: active first
    targets, notes, owner = [], {}, {}
    for ob in cands:
        if not ob.visible_get():
            notes[ob.name] = "hidden - unhide it first"
        elif not _is_editable(ob):
            notes[ob.name] = "linked from a library - make it local first"
        elif len(ob.data.polygons) == 0:
            notes[ob.name] = "has no faces to unwrap"
        elif ob.data in owner:
            notes[ob.name] = f"shares its mesh with {owner[ob.data].name} (unwrapped once)"
        else:
            owner[ob.data] = ob
            targets.append(ob)
    return targets, notes


# ------------------------------------------------------------ state guard

def _mode_set(mode):
    try:
        bpy.ops.object.mode_set(mode=mode)
        return True
    except RuntimeError:
        return False


class _SceneState:
    """Object mode, active object and object selection of the view layer."""

    def __init__(self, context):
        vl = context.view_layer
        self.view_layer = vl
        self.active = vl.objects.active
        self.mode = self.active.mode if self.active is not None else 'OBJECT'
        self.selected = [o for o in vl.objects if o.select_get()]
        self.in_mode = [o for o in vl.objects if o.mode == self.mode] if self.mode != 'OBJECT' else []

    def leave_mode(self, context):
        if context.object is not None and context.object.mode != 'OBJECT':
            _mode_set('OBJECT')
        return context.object is None or context.object.mode == 'OBJECT'

    def _select(self, objs):
        for o in self.view_layer.objects:
            try:
                o.select_set(o in objs)
            except RuntimeError:
                pass

    def restore(self, context):
        vl = self.view_layer
        if context.object is not None and context.object.mode != 'OBJECT':
            _mode_set('OBJECT')
        alive = [o for o in self.selected if o.name in vl.objects]
        try:
            vl.objects.active = self.active
        except (RuntimeError, ReferenceError):
            pass
        if self.mode != 'OBJECT' and self.active is not None:
            # re-enter with the same set of objects (multi-object editing)
            self._select([o for o in self.in_mode if o.name in vl.objects] or [self.active])
            _mode_set(self.mode)
        self._select(alive)


class _MeshState:
    """Component selection / hide flags of a mesh (object mode)."""

    def __init__(self, me):
        self.me = me
        self.data = {}
        for name in ("vertices", "edges", "polygons"):
            coll = getattr(me, name)
            for attr in ("select", "hide"):
                arr = np.zeros(len(coll), dtype=bool)
                coll.foreach_get(attr, arr)
                self.data[name, attr] = arr
        arr = np.zeros(len(me.edges), dtype=bool)
        me.edges.foreach_get("use_seam", arr)
        self.seams = int(arr.sum())
        self.uv_name = me.uv_layers.active.name if me.uv_layers.active else None

    def reveal(self):
        """Unhide everything so hidden faces get unwrapped and packed too."""
        for name in ("vertices", "edges", "polygons"):
            coll = getattr(self.me, name)
            if self.data[name, "hide"].any():
                coll.foreach_set("hide", np.zeros(len(coll), dtype=bool))

    def restore(self):
        me = self.me
        for (name, attr), arr in self.data.items():
            coll = getattr(me, name)
            if len(coll) == len(arr):
                coll.foreach_set(attr, arr)
        me.update()


def _count_seams(me):
    arr = np.zeros(len(me.edges), dtype=bool)
    me.edges.foreach_get("use_seam", arr)
    return int(arr.sum())


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _run_each(op, context, fn, verb):
    """Run ``fn(obj)`` on every target with only that object selected, then
    put mode, selection and component selection back as they were.

    Returns ``(results, notes, errors, seconds)`` where ``results`` maps
    object name -> (report dict, _MeshState before the run).
    """
    targets, notes = _collect_targets(context)
    results, errors = {}, {}
    if not targets:
        return results, notes, errors, 0.0
    t_start = time.perf_counter()
    scene_state = _SceneState(context)
    wm = context.window_manager
    wm.progress_begin(0, len(targets))
    workspace = getattr(context, "workspace", None)
    try:
        if not scene_state.leave_mode(context):
            for ob in targets:
                errors[ob.name] = "could not leave the current mode"
            return results, notes, errors, 0.0
        for i, ob in enumerate(targets):
            wm.progress_update(i)
            label = f"Smart Seams: {verb} {ob.name} ({i + 1}/{len(targets)}, {len(ob.data.polygons)} faces)"
            print(f"[Smart Seams] {label} ...")
            if workspace is not None:
                workspace.status_text_set(label + " - this can take a while on dense meshes")

            def tick(calls, _ob=ob, _label=label):
                if calls % 50 == 0:
                    print(f"[Smart Seams]   {_ob.name}: {calls} trial unwraps")
                    if workspace is not None:
                        workspace.status_text_set(f"{_label}: {calls} trial unwraps")

            blender_validator.on_call = tick
            for o in context.view_layer.objects:
                o.select_set(False)
            ob.select_set(True)
            context.view_layer.objects.active = ob
            mesh_state = _MeshState(ob.data)
            mesh_state.reveal()
            t0 = time.perf_counter()
            try:
                results[ob.name] = (fn(ob), mesh_state)
            except Exception as ex:  # never surface a raw traceback to the user
                traceback.print_exc()
                msg = str(ex).strip().splitlines()[0] if str(ex).strip() else ""
                errors[ob.name] = f"failed ({type(ex).__name__}{': ' + msg[:120] if msg else ''})"
            finally:
                blender_validator.on_call = None
                if context.object is not None and context.object.mode != 'OBJECT':
                    _mode_set('OBJECT')
                if ob.mode != 'OBJECT':
                    context.view_layer.objects.active = ob
                    _mode_set('OBJECT')
                blender_validator.cleanup()
                if ob.mode == 'OBJECT':
                    mesh_state.restore()
            print(f"[Smart Seams] {ob.name} done in {time.perf_counter() - t0:.1f} s")
    finally:
        blender_validator.on_call = None
        blender_validator.cleanup()
        scene_state.restore(context)
        wm.progress_end()
        if workspace is not None:
            workspace.status_text_set(None)
    return results, notes, errors, time.perf_counter() - t_start


def _finish(op, context, summary, notes, errors, results):
    """Report to the user; returns the operator result set."""
    s = context.scene.smart_seams
    parts = [summary] if summary else []
    if errors:
        parts.append("Errors: " + "; ".join(f"{n} {r}" for n, r in errors.items())
                     + " (details in the system console)")
    if notes:
        parts.append("Skipped: " + "; ".join(f"{n} {r}" for n, r in notes.items()))
    msg = ". ".join(parts) if parts else "Nothing to do"
    s.last_message = msg
    if not results:
        op.report({'ERROR'}, msg)
        return {'CANCELLED'}
    op.report({'ERROR'} if errors else {'WARNING'} if notes else {'INFO'}, msg)
    return {'FINISHED'}


def _total_faces(context):
    return sum(len(o.data.polygons) for o in _collect_targets(context)[0])


class _SmartSeamsOp:
    @classmethod
    def poll(cls, context):
        if context.view_layer is None or not _mesh_candidates(context):
            if hasattr(cls, "poll_message_set"):
                cls.poll_message_set("Select one or more mesh objects")
            return False
        return True

    def invoke(self, context, event):
        faces = _total_faces(context)
        if faces > HEAVY_FACES and getattr(self, "heavy_confirm", False):
            msg = f"{faces:,} faces: this can take several minutes and Blender won't respond meanwhile"
            try:
                return context.window_manager.invoke_confirm(
                    self, event, title=self.bl_label, message=msg, confirm_text="Continue", icon='WARNING')
            except TypeError:  # older signature
                return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)


class SMARTSEAMS_OT_unwrap(_SmartSeamsOp, bpy.types.Operator):
    """Solve seams, unwrap, straighten, scale hidden shells down and pack.
Replaces the active UV map and the seams of the selected meshes"""
    bl_idname = "uv.smart_seams_unwrap"
    bl_label = "Smart Seams Unwrap"
    bl_options = {'REGISTER', 'UNDO'}
    heavy_confirm = True

    def execute(self, context):
        s = context.scene.smart_seams
        opts = s.to_options()
        results, notes, errors, secs = _run_each(self, context, lambda ob: pipeline.unwrap(ob, opts), "unwrapping")
        s.last_report = json.dumps({n: r for n, (r, _st) in results.items()})
        items, replaced = [], {}
        for n, (r, st) in results.items():
            shells = r.get("islands", r.get("charts"))
            extra = ""
            if r.get("uv_overlap_pct"):
                extra = f", {r['uv_overlap_pct']}% overlap"
            if (r.get("solver") or {}).get("time_budget_hit"):
                extra += " (time budget reached: more shells than ideal)"
            items.append(f"{n} {_plural(shells, 'shell')}{extra}")
            if st.uv_name:
                replaced.setdefault(st.uv_name, []).append(n)
        summary = ""
        if items:
            summary = f"Smart Seams: {', '.join(items)} ({secs:.1f} s)"
            for uv_name, names in replaced.items():
                summary += f". Replaced UV map '{uv_name}' and seams of {', '.join(names)}"
        return _finish(self, context, summary, notes, errors, results)


class SMARTSEAMS_OT_mark(_SmartSeamsOp, bpy.types.Operator):
    """Only solve and mark seams (keep the current UVs), so you can review or tweak them.
Replaces the existing seams of the selected meshes"""
    bl_idname = "uv.smart_seams_mark"
    bl_label = "Mark Smart Seams"
    bl_options = {'REGISTER', 'UNDO'}
    heavy_confirm = True

    def execute(self, context):
        s = context.scene.smart_seams
        opts = s.to_options(mark_seams_only=True)
        meshes = {}

        def run(ob):
            meshes[ob.name] = ob.data
            return pipeline.unwrap(ob, opts)

        results, notes, errors, secs = _run_each(self, context, run, "marking seams on")
        s.last_report = json.dumps({n: r for n, (r, _st) in results.items()})
        items = []
        for n, (r, st) in results.items():
            now = _count_seams(meshes[n])
            txt = f"{n} {_plural(r['charts'], 'shell')}, {_plural(now, 'seam edge')}"
            if st.seams:
                txt += f" (replaced {_plural(st.seams, 'existing seam edge')})"
            items.append(txt)
        summary = f"Seams marked: {', '.join(items)} ({secs:.1f} s)" if items else ""
        return _finish(self, context, summary, notes, errors, results)


class SMARTSEAMS_OT_analyze(_SmartSeamsOp, bpy.types.Operator):
    """Measure the current UVs: shells, seam length/visibility, stretch, overlap"""
    bl_idname = "uv.smart_seams_analyze"
    bl_label = "Analyze UVs"

    def execute(self, context):
        s = context.scene.smart_seams
        opts = s.to_options()
        results, notes, errors, _secs = _run_each(
            self, context, lambda ob: pipeline.analyze_object(ob, opts), "analyzing")
        s.last_report = json.dumps({n: r for n, (r, _st) in results.items()})
        items = []
        for n, (r, st) in results.items():
            print(f"[Smart Seams] {n}: {json.dumps(r)}")
            if st.uv_name is None:
                items.append(f"{n} has no UV map")
            else:
                items.append(f"{n} {_plural(r.get('islands', 0), 'shell')}, "
                             f"worst stretch {r.get('worst_island_area_distortion', 0)}, "
                             f"{r.get('uv_overlap_pct', 0)}% overlap")
        summary = ("Analysis: " + "; ".join(items)) if items else ""
        return _finish(self, context, summary, notes, errors, results)


RESULT_LABELS = (
    ("islands", "Shells", "{}"),
    ("charts", "Shells", "{}"),
    ("worst_island_area_distortion", "Worst stretch", "{:.3f}"),
    ("uv_overlap_pct", "Overlap", "{}%"),
    ("flipped_faces", "Flipped faces", "{}"),
    ("seam_length", "Seam length", "{:.2f}"),
    ("seam_visibility", "Seam visibility", "{:.2f}"),
    ("seconds", "Time", "{:.1f} s"),
)


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
    box.prop(s, "time_budget")

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
            grid = box.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
            shown = set()
            for key, label, fmt in RESULT_LABELS:
                if key in r and r[key] is not None and label not in shown:
                    shown.add(label)
                    try:
                        value = fmt.format(r[key])
                    except (ValueError, TypeError):
                        value = str(r[key])
                    grid.label(text=label)
                    grid.label(text=value)


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
