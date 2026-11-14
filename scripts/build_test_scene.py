"""Build the test scene from the original scene export.

Usage (from a terminal):
    blender -b --factory-startup --python scripts/build_test_scene.py -- <scene.obj> <out_dir>

Imports the full scene OBJ, keeps only the test subject (the table) and the
reference (the lamp base), applies transforms, drops them on the ground plane
and writes:

    <out_dir>/table.obj, <out_dir>/lamp_base.obj   (small, committed test data)
    <out_dir>/uv_test.blend                        (scene to open in Blender)
"""

import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_save import save_clean  # noqa: E402
from script_util import UsageError, run, script_args  # noqa: E402

USAGE = "blender -b --factory-startup --python scripts/build_test_scene.py -- <scene.obj> <out_dir>"

OBJECTS = {
    "table_square.001": "table",
    "cubicle_david_2.106": "lamp_base",
}


def main():
    argv = script_args()
    if len(argv) != 2:
        raise UsageError(f"expected <scene.obj> <out_dir>, got {argv}")
    src, out_dir = (os.path.abspath(a) for a in argv)
    if not os.path.isfile(src):
        raise UsageError(f"scene OBJ not found: {src}")
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=src)
    missing = [n for n in OBJECTS if n not in bpy.data.objects]
    if missing:
        raise UsageError(f"{src} has no object(s) {', '.join(missing)}")
    for ob in list(bpy.data.objects):
        if ob.name not in OBJECTS:
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    x_offset = 0.0
    for src_name, new_name in OBJECTS.items():
        ob = bpy.data.objects[src_name]
        ob.name = ob.data.name = new_name
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        # origin at bottom centre, then place on the ground
        cos = [v.co for v in ob.data.vertices]
        lo = Vector((min(c.x for c in cos), min(c.y for c in cos), min(c.z for c in cos)))
        hi = Vector((max(c.x for c in cos), max(c.y for c in cos), max(c.z for c in cos)))
        pivot = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
        for v in ob.data.vertices:
            v.co -= pivot
        ob.location = (x_offset, 0.0, 0.0)
        x_offset += (hi.x - lo.x) + 1.0

        bpy.ops.wm.obj_export(filepath=os.path.join(out_dir, new_name + ".obj"),
                              export_selected_objects=True, export_materials=False,
                              apply_modifiers=False)

    # simple reference shapes: the island count should scale with complexity
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_offset, 0, 0.5))
    bpy.context.active_object.name = "cube"
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.4, depth=1.0,
                                        location=(x_offset + 1.5, 0, 0.5))
    bpy.context.active_object.name = "cylinder"
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.name in ("cube", "cylinder"):
            while ob.data.uv_layers:
                ob.data.uv_layers.remove(ob.data.uv_layers[0])

    save_clean(os.path.join(out_dir, "uv_test.blend"))


run(main, USAGE)
