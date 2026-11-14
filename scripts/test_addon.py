"""Smoke test: register the add-on and run its operators like a user would.

    blender -b test_data/uv_test.blend --factory-startup --python scripts/test_addon.py

Exits with a non-zero code if any check fails.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import smart_seams  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from script_util import mesh_object, run  # noqa: E402

USAGE = "blender -b test_data/uv_test.blend --factory-startup --python scripts/test_addon.py"


def main():
    smart_seams.register()
    try:
        # expected shell counts scale with complexity
        limits = {"cube": 2, "cylinder": 3, "lamp_base": 6, "table": 10}
        for name, max_shells in limits.items():
            ob = mesh_object(name)
            for o in bpy.context.view_layer.objects:
                o.select_set(False)
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            assert bpy.ops.uv.smart_seams_unwrap() == {'FINISHED'}
            report = json.loads(bpy.context.scene.smart_seams.last_report)[name]
            print("ADDON", name, "shells", report["islands"], "overlap", report["uv_overlap_pct"],
                  "worst stretch", report["worst_island_area_distortion"], "flipped", report["flipped_faces"])
            assert report["islands"] <= max_shells, report
            assert report["uv_overlap_pct"] == 0.0, report
            assert report["flipped_faces"] == 0, report
            assert report["worst_island_area_distortion"] <= 0.2, report
        assert "__smart_seams_validator__" not in bpy.data.objects, "temp object leaked"
        assert bpy.ops.uv.smart_seams_analyze() == {'FINISHED'}
        print("ADDON OK")
    finally:
        smart_seams.unregister()


run(main, USAGE)
