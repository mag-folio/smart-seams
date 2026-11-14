"""Render a preview of an object's UVs: islands coloured on the model, seams in
red, from two angles (front-top and below), plus the UV layout.

    blender -b <file.blend> --factory-startup --python scripts/render_preview.py -- <object> <out_prefix>

Writes <out_prefix>_top.png, <out_prefix>_bottom.png and <out_prefix>_uv.png.
"""

import colorsys
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smart_seams.core import metrics  # noqa: E402


def island_colors(ob):
    me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    uvl = bm.loops.layers.uv.active
    islands = metrics.uv_islands(bm, uvl)
    col = {}
    for i, isl in enumerate(islands):
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.95)
        for f in isl:
            col[f.index] = (r, g, b, 1.0)
    attr = me.color_attributes.get("island") or me.color_attributes.new("island", 'BYTE_COLOR', 'CORNER')
    for poly in me.polygons:
        for li in poly.loop_indices:
            attr.data[li].color = col[poly.index]
    me.color_attributes.active_color = attr
    # seam edges = any UV discontinuity (island borders and cuts inside islands)
    seam_edges = []
    for e in bm.edges:
        ll = e.link_loops
        if len(ll) != 2:
            continue
        a, b = ll
        a1, a2 = a[uvl].uv, a.link_loop_next[uvl].uv
        b1, b2 = ((b[uvl].uv, b.link_loop_next[uvl].uv) if b.vert is a.vert
                  else (b.link_loop_next[uvl].uv, b[uvl].uv))
        if (a1 - b1).length > 1e-5 or (a2 - b2).length > 1e-5:
            seam_edges.append((e.verts[0].co.copy(), e.verts[1].co.copy()))
    bm.free()
    return seam_edges, len(islands)


def seam_object(ob, edges):
    me = bpy.data.meshes.new("seams")
    verts, idx = [], []
    for a, b in edges:
        verts += [a, b]
        idx.append((len(verts) - 2, len(verts) - 1))
    me.from_pydata(verts, idx, [])
    so = bpy.data.objects.new("seams", me)
    so.matrix_world = ob.matrix_world
    bpy.context.scene.collection.objects.link(so)
    bpy.context.view_layer.objects.active = so
    for o in bpy.context.scene.objects:
        o.select_set(o is so)
    bpy.ops.object.convert(target='CURVE')
    so = bpy.context.view_layer.objects.active
    dims = ob.dimensions
    so.data.bevel_depth = max(dims) * 0.004
    bpy.ops.object.convert(target='MESH')
    so = bpy.context.view_layer.objects.active
    attr = so.data.color_attributes.new("island", 'BYTE_COLOR', 'CORNER')
    for d in attr.data:
        d.color = (1.0, 0.02, 0.02, 1.0)
    so.data.color_attributes.active_color = attr
    return so


def look_at(cam, target, direction, dist):
    cam.location = target + direction.normalized() * dist
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    name, prefix = argv[0], argv[1]
    ob = bpy.data.objects[name]
    if ob.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in list(bpy.data.objects):
        if o is not ob:
            bpy.data.objects.remove(o, do_unlink=True)
    edges, n = island_colors(ob)
    seam_object(ob, edges)

    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'STUDIO'
    sh.color_type = 'VERTEX'
    sh.show_cavity = False
    sh.show_object_outline = True
    sc.render.resolution_x = sc.render.resolution_y = 900
    # published images: no embedded metadata (file path, date, render time...)
    for attr in dir(sc.render):
        if attr.startswith("use_stamp"):
            try:
                setattr(sc.render, attr, False)
            except (AttributeError, TypeError):
                pass
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("w") if not sc.world else sc.world
    sc.world.color = (0.12, 0.12, 0.13)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    centre = ob.matrix_world @ (sum((Vector(b) for b in ob.bound_box), Vector()) / 8)
    dist = max(ob.dimensions) * 2.6

    for tag, d in (("top", Vector((0.9, -1.4, 0.9))), ("bottom", Vector((-0.8, 1.2, -0.9)))):
        look_at(cam, centre, d, dist)
        sc.render.filepath = f"{prefix}_{tag}.png"
        bpy.ops.render.render(write_still=True)

    # checker render: squares stay square and even where there is no stretch
    img = bpy.data.images.new("checker", 1024, 1024)
    img.generated_type = 'COLOR_GRID'
    mat = bpy.data.materials.new("checker")
    mat.use_nodes = True
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = 'Closest'
    nt.links.new(tex.outputs[0], nt.nodes["Principled BSDF"].inputs[0])
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    nt.nodes.active = tex
    sh.color_type = 'TEXTURE'
    look_at(cam, centre, Vector((0.9, -1.4, 0.9)), dist)
    sc.render.filepath = f"{prefix}_checker.png"
    bpy.ops.render.render(write_still=True)
    sh.color_type = 'VERTEX'

    # UV layout
    bpy.context.view_layer.objects.active = ob
    for o in sc.objects:
        o.select_set(o is ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.export_layout(filepath=f"{prefix}_uv.png", mode='PNG', size=(900, 900), opacity=0.6,
                             check_existing=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    print("PREVIEW", name, "islands", n)


main()
