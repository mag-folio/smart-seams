"""Development helper: run the panel solver and print stats per mesh.

    blender -b test_data/uv_test.blend --factory-startup --python scripts/debug_panels.py -- [object ...]
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bmesh  # noqa: E402
import bpy  # noqa: E402

from smart_seams.core import panels as P  # noqa: E402
from smart_seams.core.blender_validator import BlenderValidator  # noqa: E402
from smart_seams.core.visibility import face_visibility  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from script_util import mesh_object, run, script_args  # noqa: E402

USAGE = "blender -b <file.blend> --factory-startup --python scripts/debug_panels.py -- [object ...]"


def main():
    argv = script_args()
    # unknown names used to be skipped silently; now they are reported
    objects = ([mesh_object(n) for n in argv] if argv
               else [o for o in bpy.data.objects if o.type == 'MESH'])
    for ob in objects:
        bm = bmesh.new(); bm.from_mesh(ob.data)
        vis = face_visibility(bm, ob.matrix_world, samples=8)
        t = time.perf_counter()
        with BlenderValidator() as val:
            cof, seams = P.compute_shells(bm, vis, validator=val)
        print(f"PNL validations {val.calls}")
        sizes = {}
        for c in cof:
            sizes[c] = sizes.get(c, 0) + 1
        print(f"PNL {ob.name}: {time.perf_counter() - t:.1f}s {P.LAST_STATS} sizes {sorted(sizes.values(), reverse=True)}")
        bm.free()


run(main, USAGE)
