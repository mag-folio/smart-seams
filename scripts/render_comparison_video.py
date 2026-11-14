"""Side-by-side stretch comparison video: the lamp base's original hand-made UVs vs Smart Seams.

    blender -b test_data/uv_test.blend --factory-startup --python scripts/render_comparison_video.py -- <out_dir> [mode]

mode: video (default, 14 s with labels, metrics and the UV layouts), still (three test frames),
hero (10 s seamless loop for a page header, models on the right), hero-still.

The Smart Seams unwrap runs on a copy of the object in memory; nothing is saved back to the
.blend. The heat map colours each face by its full stretch (area and one-directional): the
singular values of the 3D->UV map after removing the shell's overall scale, so it agrees with
the checker. The on-screen numbers are ``core.metrics.analyze`` (area-only "worst shell stretch").
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smart_seams.core import metrics, pipeline  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
MODE = argv[1] if len(argv) > 1 else "video"   # video | still | hero | hero-still
STILL = MODE.endswith("still")
HERO = MODE.startswith("hero")
os.makedirs(OUT, exist_ok=True)

FPS = 30
SEG = 150            # frames per turntable segment (one full turn)
UVSEG = 120          # frames for the UV layout segment
END = 2 * SEG if HERO else 2 * SEG + UVSEG
# hero loop: checker, cross-fade to heat map, cross-fade back (seamless loop)
MIX_EXPR = (f"max(0, min(1, (frame - {SEG - 15}) / 15)) - max(0, min(1, (frame - {2 * SEG - 15}) / 15))"
            if HERO else f"1.0 if frame > {SEG} else 0.0")
# label font for the hero loop (Blender's built-in font if it is missing)
FONT = r"C:\Windows\Fonts\seguisb.ttf"
STRETCH_RED = 1.0    # |log(area ratio)| at which a face is full red (2.7x)


def srgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


RAMP = [srgb("#2f7fd8"), srgb("#3fb67a"), srgb("#f2d03b"), srgb("#e8572a"), srgb("#c0102a")]
FLIP = srgb("#d23cd2")


def ramp(t):
    t = min(max(t, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(t), len(RAMP) - 2)
    f = t - i
    return tuple(a + (b - a) * f for a, b in zip(RAMP[i], RAMP[i + 1]))


def face_stretch(ob):
    """Per-face |log(area ratio)| normalised per shell (same definition as metrics.analyze)
    and per-face flip flag. Returns ({face_index: (err, flipped)}, metrics dict)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    uvl = bm.loops.layers.uv.active
    out = {}
    for isl in metrics.uv_islands(bm, uvl):
        a3f = [f.calc_area() for f in isl]
        s = [metrics._uv_area(f, uvl) for f in isl]
        a3, a2 = sum(a3f), sum(abs(x) for x in s)
        collapsed = a3 <= 0 or a2 <= 1e-12 * a3
        k = a2 / a3 if a3 else 0
        major = 1.0 if sum(s) >= 0 else -1.0
        for f, x, fa in zip(isl, s, a3f):
            if fa <= 0:
                out[f.index] = (0.0, False)
            elif collapsed:
                out[f.index] = (metrics.COLLAPSED, False)
            else:
                out[f.index] = (abs(math.log(max(abs(x) / (fa * k), 1e-6))), x * major < 0)
    m = metrics.analyze(bm, uvl)
    # full stretch (area AND one-directional): singular values of the 3D->UV map per
    # triangle, after removing each shell's overall scale. 0 = undistorted.
    shell_k = {}
    for isl in metrics.uv_islands(bm, uvl):
        a3 = sum(f.calc_area() for f in isl)
        a2 = sum(abs(metrics._uv_area(f, uvl)) for f in isl)
        for f in isl:
            shell_k[f.index] = (a2 / a3) if a3 > 0 else 0.0
    bm.free()
    me = ob.data
    me.calc_loop_triangles()
    uvd = me.uv_layers.active.data
    acc = {}
    for lt in me.loop_triangles:
        pi = lt.polygon_index
        p = [me.vertices[v].co for v in lt.vertices]
        uv = [uvd[li].uv for li in lt.loops]
        e1, e2 = p[1] - p[0], p[2] - p[0]
        a = e1.cross(e2).length / 2
        if a <= 1e-14:
            continue
        x = e1.normalized()
        y = e1.cross(e2).cross(e1).normalized()
        P = np.array([[e1.dot(x), e2.dot(x)], [e1.dot(y), e2.dot(y)]])
        Q = np.array([[uv[1].x - uv[0].x, uv[2].x - uv[0].x], [uv[1].y - uv[0].y, uv[2].y - uv[0].y]])
        k = shell_k.get(pi, 0.0)
        if k <= 0:
            d = metrics.COLLAPSED
        else:
            s = np.linalg.svd(Q @ np.linalg.inv(P) / math.sqrt(k), compute_uv=False)
            d = max(abs(math.log(max(float(v), 1e-6))) for v in s)
        w, t = acc.get(pi, (0.0, 0.0))
        acc[pi] = (w + a, t + a * d)
    full = {i: (min(acc[i][1] / acc[i][0], metrics.COLLAPSED) if i in acc else 0.0, out[i][1]) for i in out}
    return full, m


def paint(me, per_face):
    attr = me.color_attributes.new("stretch", 'FLOAT_COLOR', 'CORNER')
    for p in me.polygons:
        e, fl = per_face[p.index]
        c = FLIP if fl else ramp(e / STRETCH_RED)
        for li in p.loop_indices:
            attr.data[li].color = (*c, 1.0)


def emission_mat(name, color, strength=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (*color, 1.0)
    em.inputs[1].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], o.inputs[0])
    return mat


def add_driver(target, path, expr, index=-1):
    fc = target.driver_add(path, index) if index >= 0 else target.driver_add(path)
    fc.driver.type = 'SCRIPTED'
    fc.driver.expression = expr
    return fc


def surface_mat(name, checker_scale):
    """Checker on UVs (segment 1) -> stretch heat map (segment 2), switched by a driver."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    uv = nt.nodes.new("ShaderNodeUVMap")
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = checker_scale
    chk.inputs["Color1"].default_value = (*srgb("#f4f1ea"), 1)
    chk.inputs["Color2"].default_value = (*srgb("#2b2b33"), 1)
    nt.links.new(uv.outputs[0], chk.inputs["Vector"])
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "stretch"
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    nt.links.new(chk.outputs["Color"], mix.inputs[6])
    nt.links.new(attr.outputs["Color"], mix.inputs[7])
    add_driver(mix.inputs[0], "default_value", MIX_EXPR)
    dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
    nt.links.new(mix.outputs[2], dif.inputs[0])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[1].default_value = 0.35
    nt.links.new(mix.outputs[2], em.inputs[0])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(dif.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(add.outputs[0], o.inputs[0])
    return mat


def show_between(ob, a, b):
    """Visible in render only for a < frame <= b."""
    add_driver(ob, "hide_render", f"not ({a} < frame <= {b})")


def text(name, body, parent, loc, size, color, align='CENTER'):
    cu = bpy.data.curves.new(name, 'FONT')
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = 'CENTER'
    if HERO and os.path.isfile(FONT):
        cu.font = bpy.data.fonts.load(FONT, check_existing=True)
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.location = loc
    ob.data.materials.append(emission_mat(name + "_m", color))
    return ob


def uv_mesh(src, name, parent, center, size):
    """The object's UV layout as a flat mesh (coloured by stretch), 0-1 square = size."""
    me = src.data
    uvd = me.uv_layers.active.data
    verts, faces, cols = [], [], []
    sattr = me.color_attributes["stretch"].data
    for p in me.polygons:
        idx = []
        for li in p.loop_indices:
            u, v = uvd[li].uv
            verts.append(((u - 0.5) * size, (v - 0.5) * size, 0.0))
            idx.append(len(verts) - 1)
            cols.append(tuple(sattr[li].color))
        faces.append(idx)
    m = bpy.data.meshes.new(name)
    m.from_pydata(verts, [], faces)
    a = m.color_attributes.new("stretch", 'FLOAT_COLOR', 'CORNER')
    for i, c in enumerate(cols):
        a.data[i].color = c
    mat = bpy.data.materials.new(name + "_m")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "stretch"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(at.outputs["Color"], em.inputs[0])
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], o.inputs[0])
    m.materials.append(mat)
    ob = bpy.data.objects.new(name, m)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.location = center
    # backing square (UV 0-1 space) and frame
    bg = bpy.data.meshes.new(name + "_bg")
    h = size / 2
    bg.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    bg.materials.append(emission_mat(name + "_bgm", srgb("#1d1d22")))
    bgo = bpy.data.objects.new(name + "_bg", bg)
    bpy.context.scene.collection.objects.link(bgo)
    bgo.parent = parent
    bgo.location = (center[0], center[1], center[2] - size * 0.01)
    return [ob, bgo]


def main():
    sc = bpy.context.scene
    hand = bpy.data.objects["lamp_base"]
    for o in list(bpy.data.objects):
        if o is not hand:
            bpy.data.objects.remove(o, do_unlink=True)
    if hand.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # second copy gets the automatic unwrap (in memory only)
    auto = hand.copy()
    auto.data = hand.data.copy()
    auto.name = "lamp_base_auto"
    sc.collection.objects.link(auto)
    for o in sc.objects:
        o.select_set(o is auto)
    bpy.context.view_layer.objects.active = auto
    rep = pipeline.unwrap(auto, pipeline.Options())
    bpy.ops.object.mode_set(mode='OBJECT')
    print("AUTO REPORT", {k: v for k, v in rep.items() if k != "solver"})

    stats = {}
    for ob in (hand, auto):
        pf, m = face_stretch(ob)
        paint(ob.data, pf)
        stats[ob.name] = m
        print("METRICS", ob.name, m)

    # layout
    D = max(hand.dimensions)
    H = hand.dimensions.z
    S = (5.0 if HERO else 3.7) * D  # ortho width
    xs = (0.13 * S, 0.35 * S) if HERO else (-0.78 * D, 0.78 * D)
    for ob, x in ((hand, xs[0]), (auto, xs[1])):
        ob.location = (x, 0.0, -H / 2)
        ob.rotation_euler = (0, 0, 0)
        add_driver(ob, "rotation_euler", f"min(frame - 1, {2 * SEG}) * 2 * pi / {SEG}", 2)
        bm = bmesh.new(); bm.from_mesh(ob.data)
        a3 = sum(f.calc_area() for f in bm.faces)
        uvl = bm.loops.layers.uv.active
        a2 = sum(abs(metrics._uv_area(f, uvl)) for f in bm.faces)
        bm.free()
        # same average checker square size in 3D on both objects
        ob.data.materials.clear()
        ob.data.materials.append(surface_mat(ob.name + "_mat", 14.0 * math.sqrt(a3 / a2) / math.sqrt(a3)))
        show_between(ob, 0, 2 * SEG)

    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = 'ORTHO'
    cam_d.ortho_scale = S
    cam_d.clip_start = 0.001
    cam_d.clip_end = 100 * D
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    d = Vector((0, -1, 0.6)).normalized()
    cam.location = d * 10 * D + Vector((0, 0, 0.0 if HERO else -0.03 * S))
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()

    half_h = S * 9 / 32
    white, grey = srgb("#f2f2f2"), srgb("#a8a8b0")
    zt = -2 * D  # text plane in camera space

    if HERO:
        pill = srgb("#2a2a30")
        for name, body, x in (("h_left", "Hand-made UVs", xs[0]), ("h_right", "Smart Seams", xs[1])):
            t = text(name, body, cam, (x, 0.088 * S, zt), 0.019 * S, white)
            bpy.context.view_layer.update()
            tw = t.dimensions.x
            bg = bpy.data.meshes.new(name + "_pill")
            w, h = tw + 0.03 * S, 0.034 * S
            r = h / 2
            pts = []
            for cx, a0 in ((w / 2 - r, -90), (-(w / 2 - r), 90)):
                for i in range(13):
                    a = math.radians(a0 + 15 * i)
                    pts.append((cx + r * math.cos(a), r * math.sin(a), 0))
            bg.from_pydata(pts, [], [list(range(len(pts)))])
            bg.materials.append(emission_mat(name + "_pm", pill))
            po = bpy.data.objects.new(name + "_pill", bg)
            sc.collection.objects.link(po)
            po.parent = cam
            po.location = (x, 0.0898 * S, zt - 0.01 * D)
        mid = (xs[0] + xs[1]) / 2
        for name, body, a, b in (("hc1", "checker test", 0, SEG - 7),
                                 ("hc2", "stretch heat map:  blue = even,  red = stretched,  pink = flipped",
                                  SEG - 7, 2 * SEG - 7),
                                 ("hc3", "checker test", 2 * SEG - 7, 2 * SEG + 1)):
            c = text(name, body, cam, (mid, -0.085 * S, zt), 0.016 * S, srgb("#e4e4ea"))
            show_between(c, a, b)
        bg_col = srgb("#606064")
    else:
        bg_col = srgb("#26262c")
    if not HERO:
        annotate(sc, cam, S, D, half_h, zt, white, grey, stats, hand, auto)

    # world / light / render
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (*bg_col, 1)
    render_out(sc, D)


def annotate(sc, cam, S, D, half_h, zt, white, grey, stats, hand, auto):
    text("t_left", "HAND-MADE (original file)", cam, (-0.25 * S, half_h - 0.06 * S, zt), 0.034 * S, white)
    text("t_right", "SMART SEAMS (automatic)", cam, (0.25 * S, half_h - 0.06 * S, zt), 0.034 * S, white)

    def line(m):
        return (f"worst shell stretch {m['worst_island_area_distortion']:.2f}   "
                f"angle error {m['angle_distortion_deg']:.1f} deg\n"
                f"flipped faces {m['flipped_faces']}   overlap {m['uv_overlap_pct']:.1f} %   "
                f"shells {m['islands']}")
    text("m_left", line(stats[hand.name]), cam, (-0.25 * S, -half_h + 0.07 * S, zt), 0.019 * S, grey)
    text("m_right", line(stats[auto.name]), cam, (0.25 * S, -half_h + 0.07 * S, zt), 0.019 * S, grey)

    caps = [
        ("c1", "Checker on the UVs: even, square tiles = no stretch", 0, SEG),
        ("c2", "Stretch heat map: blue = none, red = 2.7x or more in any direction, magenta = flipped",
         SEG, 2 * SEG),
        ("c3", "The UV layouts, coloured by the same stretch heat map", 2 * SEG, END),
    ]
    for name, body, a, b in caps:
        c = text(name, body, cam, (0, half_h - 0.115 * S, zt), 0.019 * S, srgb("#ffd25a"))
        show_between(c, a, b)

    # segment 3: UV layouts (camera space)
    size = 0.30 * S
    for src, x in ((hand, -0.25 * S), (auto, 0.25 * S)):
        for o in uv_mesh(src, src.name + "_uv", cam, (x, -0.03 * S, zt - 0.5 * D), size):
            show_between(o, 2 * SEG, END)


def render_out(sc, D):
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
    sun.data.energy = 2.5
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    sc.collection.objects.link(sun)

    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            pass
    try:
        sc.eevee.taa_render_samples = 16
    except AttributeError:
        pass
    sc.view_settings.view_transform = 'Standard'
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.resolution_percentage = 100
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, END
    for attr in dir(sc.render):
        if attr.startswith("use_stamp"):
            try:
                setattr(sc.render, attr, False)
            except (AttributeError, TypeError):
                pass

    if STILL:
        frames = (1, 75, SEG - 8, 220, 2 * SEG) if HERO else (40, 2 * SEG - 110, 2 * SEG + 30)
        for f in frames:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(OUT, f"{'hero' if HERO else 'still'}_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
        return

    ims = sc.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = 'VIDEO'
    ims.file_format = 'FFMPEG'
    sc.render.ffmpeg.format = 'MPEG4'
    sc.render.ffmpeg.codec = 'H264'
    sc.render.ffmpeg.constant_rate_factor = 'MEDIUM' if HERO else 'HIGH'
    sc.render.ffmpeg.audio_codec = 'NONE'
    sc.render.filepath = os.path.join(
        OUT, "lamp_base_hero_loop.mp4" if HERO else "lamp_base_handmade_vs_smart_seams.mp4")
    bpy.ops.render.render(animation=True)
    print("VIDEO", sc.render.filepath)


main()
