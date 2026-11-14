"""Run the full Smart Seams pipeline on a set of procedurally built shapes.

    blender -b --factory-startup --python tests/test_shapes.py [-- shape ...]

For every shape the README's hard guarantees are asserted:

- no exception, and no temporary validator object/mesh left behind,
- 0 % UV overlap (raster) and 0 crossing UV boundary segments (exact),
- 0 flipped faces,
- shells <= Max Shells,
- worst shell stretch <= Stretch Ceiling.

Shapes that currently break a guarantee are listed in KNOWN_FAILURES with the
numbers measured when they were added. They are reported as XFAIL and do not
fail the run; anything else that breaks a guarantee fails the run (exit 1). A
known failure that starts passing is reported as XPASS so the list can be
trimmed. Do not add entries here to hide a regression: fix the solver.
"""

import math
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bmesh  # noqa: E402
import bpy  # noqa: E402

from smart_seams.core import metrics, pipeline  # noqa: E402
from smart_seams.core.blender_validator import TMP_NAME  # noqa: E402

# shape -> what was measured when the failure was recorded (see module docstring)
KNOWN_FAILURES = {}

# the same geometry built with a different vertex order must give the same shell count
SAME_AS = {"l_extrusion_rot": "l_extrusion"}


# --------------------------------------------------------------------------- shapes
def _from_bmesh(name, bm):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    # without this the view layer's object list can hold stale None entries
    # after objects were removed through bpy.data (pipeline iterates it)
    bpy.context.view_layer.update()
    # select + activate like a user would before running the operator
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    return ob


def _from_op(name, op, **kw):
    op(**kw)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.name = name
    return ob


def make_plane():
    return _from_op("plane", bpy.ops.mesh.primitive_plane_add, size=2)


def make_uv_sphere():
    return _from_op("uv_sphere", bpy.ops.mesh.primitive_uv_sphere_add, segments=32, ring_count=16, radius=1)


def make_ico_sphere():
    return _from_op("ico_sphere", bpy.ops.mesh.primitive_ico_sphere_add, subdivisions=2, radius=1)


def make_cone():
    return _from_op("cone", bpy.ops.mesh.primitive_cone_add, vertices=32, radius1=1, depth=2,
                    end_fill_type='NGON')


def make_torus():
    return _from_op("torus", bpy.ops.mesh.primitive_torus_add, major_segments=48, minor_segments=12,
                    major_radius=1, minor_radius=0.25)


def make_suzanne():
    return _from_op("suzanne", bpy.ops.mesh.primitive_monkey_add, size=2)


def make_bevelled_cube():
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2)
    bmesh.ops.bevel(bm, geom=list(bm.edges) + list(bm.verts), offset=0.15, segments=3,
                    affect='EDGES', profile=0.5)
    return _from_bmesh("bevelled_cube", bm)


def make_l_extrusion(start=0, name="l_extrusion"):
    # L-shaped n-gon (concave) extruded upwards: two concave n-gon caps + 6 walls.
    # ``start`` rotates the cap's corner order; the geometry is identical.
    bm = bmesh.new()
    pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    pts = pts[start:] + pts[:start]
    vs = [bm.verts.new((x, y, 0)) for x, y in pts]
    base = bm.faces.new(vs)
    res = bmesh.ops.extrude_face_region(bm, geom=[base])
    top = [v for v in res["geom"] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=top, vec=(0, 0, 1))
    bmesh.ops.reverse_faces(bm, faces=[base])
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return _from_bmesh(name, bm)


def make_l_extrusion_rot():
    # same L, but the caps start at the corner next to the reflex corner, so a
    # triangle fan from corner 0 is not a valid triangulation of the cap
    return make_l_extrusion(start=2, name="l_extrusion_rot")


def make_pipe():
    # hollow cylinder: outer wall, inner wall, top and bottom rings (all quads)
    n, r_out, r_in, h = 24, 1.0, 0.7, 2.0
    bm = bmesh.new()
    ring = {}
    for k in range(n):
        a = 2 * math.pi * k / n
        for r in (r_out, r_in):
            for z in (0.0, h):
                ring[k, r, z] = bm.verts.new((r * math.cos(a), r * math.sin(a), z))
    for k in range(n):
        k1 = (k + 1) % n
        bm.faces.new([ring[k, r_out, 0], ring[k1, r_out, 0], ring[k1, r_out, h], ring[k, r_out, h]])
        bm.faces.new([ring[k1, r_in, 0], ring[k, r_in, 0], ring[k, r_in, h], ring[k1, r_in, h]])
        bm.faces.new([ring[k, r_out, h], ring[k1, r_out, h], ring[k1, r_in, h], ring[k, r_in, h]])
        bm.faces.new([ring[k1, r_out, 0], ring[k, r_out, 0], ring[k, r_in, 0], ring[k1, r_in, 0]])
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return _from_bmesh("pipe", bm)


def make_star_prism():
    # 5-point star n-gon (5 reflex corners) extruded: concave n-gon caps
    bm = bmesh.new()
    vs = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        r = 1.0 if k % 2 == 0 else 0.45
        vs.append(bm.verts.new((r * math.cos(a), r * math.sin(a), 0)))
    base = bm.faces.new(vs)
    res = bmesh.ops.extrude_face_region(bm, geom=[base])
    top = [v for v in res["geom"] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=top, vec=(0, 0, 0.5))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return _from_bmesh("star_prism", bm)


def make_cylinder_trifan():
    return _from_op("cylinder_trifan", bpy.ops.mesh.primitive_cylinder_add, vertices=32, radius=1,
                    depth=2, end_fill_type='TRIFAN')


def make_suzanne_subdiv():
    ob = make_suzanne()
    ob.name = ob.data.name = "suzanne_subdiv"
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=1, use_grid_fill=True)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


SHAPES = {
    "plane": make_plane,
    "bevelled_cube": make_bevelled_cube,
    "l_extrusion": make_l_extrusion,
    "l_extrusion_rot": make_l_extrusion_rot,
    "pipe": make_pipe,
    "cone": make_cone,
    "uv_sphere": make_uv_sphere,
    "ico_sphere": make_ico_sphere,
    "torus": make_torus,
    "suzanne": make_suzanne,
    # beyond the README list
    "star_prism": make_star_prism,
    "cylinder_trifan": make_cylinder_trifan,
    "suzanne_subdiv": make_suzanne_subdiv,
}


# --------------------------------------------------------------------------- run
def clear_scene():
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)
    bpy.context.view_layer.update()


def independent_metrics(ob):
    """Re-measure from the saved mesh data, not from the pipeline's own report."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    m = metrics.analyze(bm, bm.loops.layers.uv.active)
    bm.free()
    return m


def run_shape(name, opts):
    clear_scene()
    ob = SHAPES[name]()
    row = {"shape": name, "faces": len(ob.data.polygons), "error": None}
    t = time.perf_counter()
    try:
        rep = pipeline.unwrap(ob, opts)
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:  # noqa: BLE001
        row["error"] = traceback.format_exc().strip().splitlines()[-1]
        traceback.print_exc()
        rep = None
    row["seconds"] = round(time.perf_counter() - t, 2)
    leaked = [o.name for o in bpy.data.objects if o.name.startswith(TMP_NAME)] + \
             [m.name for m in bpy.data.meshes if m.name.startswith(TMP_NAME)]
    row["leaked"] = leaked
    if rep is not None:
        m = independent_metrics(ob)
        row.update({
            "shells": m["islands"],
            "overlap": m["uv_overlap_pct"],
            "crossings": m["uv_boundary_crossings"],
            "stretch": m["worst_island_area_distortion"],
            "angle": m["angle_distortion_deg"],
            "flipped": m["flipped_faces"],
            "mirrored": m["mirrored_islands"],
            "report_shells": rep.get("islands"),
            "solver": rep.get("solver"),
        })
    problems = []
    if row["error"]:
        problems.append(f"exception: {row['error']}")
    if leaked:
        problems.append(f"leaked temp data: {leaked}")
    if rep is not None:
        if row["shells"] > opts.max_islands:
            problems.append(f"shells {row['shells']} > max {opts.max_islands}")
        if row["overlap"] != 0.0:
            problems.append(f"overlap {row['overlap']} %")
        if row["crossings"] != 0:
            problems.append(f"{row['crossings']} UV boundary crossings")
        if row["flipped"] != 0:
            problems.append(f"{row['flipped']} flipped faces")
        if row["stretch"] > opts.max_distortion:
            problems.append(f"worst stretch {row['stretch']} > ceiling {opts.max_distortion}")
    row["problems"] = problems
    return row


def check_same_as(rows):
    by_name = {r["shape"]: r for r in rows}
    for name, ref in SAME_AS.items():
        a, b = by_name.get(name), by_name.get(ref)
        if a is None or b is None or "shells" not in a or "shells" not in b:
            continue
        if a["shells"] != b["shells"]:
            a["problems"].append(f"{a['shells']} shells vs {b['shells']} for the same geometry ({ref})")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    names = argv or list(SHAPES)
    opts = pipeline.Options()
    rows = []
    for name in names:
        print(f"--- {name}", flush=True)
        row = run_shape(name, opts)
        print("   ", row, flush=True)
        rows.append(row)
    clear_scene()
    check_same_as(rows)

    print("\n| Shape | Faces | Shells | Overlap % | Crossings | Worst stretch | Angle err | Flipped "
          "| Mirrored | Seconds | Result |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    unexpected = []
    for r in rows:
        known = r["shape"] in KNOWN_FAILURES
        if r["problems"]:
            status = "XFAIL" if known else "FAIL"
            if not known:
                unexpected.append(r)
        else:
            status = "XPASS" if known else "pass"
        print(f"| {r['shape']} | {r['faces']} | {r.get('shells', '-')} | {r.get('overlap', '-')} | "
              f"{r.get('crossings', '-')} | {r.get('stretch', '-')} | {r.get('angle', '-')} | "
              f"{r.get('flipped', '-')} | {r.get('mirrored', '-')} | {r['seconds']} | {status} |")
    for r in rows:
        if r["problems"]:
            tag = "XFAIL" if r["shape"] in KNOWN_FAILURES else "FAIL"
            print(f"{tag} {r['shape']}: " + "; ".join(r["problems"]))
            if r["shape"] in KNOWN_FAILURES:
                print(f"      recorded: {KNOWN_FAILURES[r['shape']]}")
    print(f"\n{len(rows)} shapes, {len(unexpected)} unexpected failures")
    sys.exit(1 if unexpected else 0)


# Blender exits with status 0 after an uncaught exception in a --python script
try:
    main()
except SystemExit:
    raise
except BaseException:  # noqa: BLE001
    traceback.print_exc()
    sys.exit(1)
