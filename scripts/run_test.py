"""Headless regression test.

    blender -b test_data/uv_test.blend --factory-startup --python scripts/run_test.py -- [out.blend] [key=value ...]

Prints metrics for the original UVs and for Smart Seams on every mesh in the
file, and optionally saves the result.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

from smart_seams.core import pipeline  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from clean_save import save_clean  # noqa: E402


def parse(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"True": True, "False": False}.get(v, v)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = next((a for a in argv if a.endswith(".blend")), None)
    kw = dict(a.split("=", 1) for a in argv if "=" in a)
    opts = pipeline.Options(**{k: parse(v) for k, v in kw.items()})

    results = {}
    for ob in [o for o in bpy.data.objects if o.type == 'MESH']:
        before = pipeline.analyze_object(ob, opts)
        for o in bpy.data.objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        after = pipeline.unwrap(ob, opts)
        bpy.ops.object.mode_set(mode='OBJECT')
        results[ob.name] = {"before": before, "after": after}
        print("RESULT", ob.name)
        print("  before", json.dumps(before))
        print("  after ", json.dumps(after))
    if out:
        save_clean(out)
    return results


main()
