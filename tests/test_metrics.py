"""Known-answer tests for smart_seams.core.metrics and smart_seams.core.flatten.

    blender -b --factory-startup --python tests/test_metrics.py

Every case is a mesh built procedurally with BMesh and given hand-computed UVs, so
the expected value of each metric is known exactly (or within the stated raster
tolerance). Exits with status 1 if any check fails.
"""

import math
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bmesh  # noqa: E402
from mathutils import Vector  # noqa: E402

from smart_seams.core import flatten, metrics  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(name, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


def close(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


# --------------------------------------------------------------------------- helpers
def build(faces, uv_layer=True):
    """faces: list of [(xyz, uv), ...] per face. Vertices are shared by position,
    UVs are per corner, so seams appear wherever the UVs given for a shared edge
    differ. Returns (bm, uv_layer)."""
    bm = bmesh.new()
    verts = {}

    def vert(co):
        key = tuple(round(c, 9) for c in co)
        if key not in verts:
            verts[key] = bm.verts.new(co)
        return verts[key]
    made = []
    for corners in faces:
        made.append(bm.faces.new([vert(c) for c, _ in corners]))
    bm.verts.index_update(); bm.edges.index_update(); bm.faces.index_update()
    bm.normal_update()
    uvl = bm.loops.layers.uv.new("UVMap") if uv_layer else None
    if uvl is not None:
        for f, corners in zip(made, faces):
            for loop, (_, uv) in zip(f.loops, corners):
                loop[uvl].uv = uv
    return bm, uvl


def quad(x0, y0, x1, y1, uv=None, z=0.0):
    """Axis-aligned 3D quad (CCW seen from +Z); uv(x, y) maps a 3D xy to a UV."""
    uv = uv or (lambda x, y: (x, y))
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [((x, y, z), uv(x, y)) for x, y in pts]


def cube_cross_net():
    """Unit cube unfolded into a cross (top in the middle, the four sides around
    it, the bottom below the -Y side), scaled by 1/4 so it fits the unit square.
    Orientation preserving (every face CCW in UV seen from outside) and an
    exact isometry up to scale."""
    s = 0.25
    (bottom, bq), (top, tq), (ym, _), (yp, _), (xm, _), (xp, _) = _cube_faces()
    faces = [
        [(p, ((1 + p[0]) * s, (2 + p[1]) * s)) for p in top],
        [(p, ((1 + p[0]) * s, (1 + p[2]) * s)) for p in ym],
        [(p, ((1 + p[0]) * s, (4 - p[2]) * s)) for p in yp],
        [(p, (p[2] * s, (2 + p[1]) * s)) for p in xm],
        [(p, ((3 - p[2]) * s, (2 + p[1]) * s)) for p in xp],
        [(p, ((1 + p[0]) * s, (1 - p[1]) * s)) for p in bottom],
    ]
    return faces


def cube_six_islands():
    """Unit cube, every face its own 0.2 x 0.2 UV square in a 3 x 2 grid."""
    faces = []
    for i, (p, uvmap) in enumerate(_cube_faces()):
        ox, oy = (i % 3) * 0.3 + 0.05, (i // 3) * 0.3 + 0.05
        faces.append([(c, (ox + 0.2 * a, oy + 0.2 * b)) for c, (a, b) in zip(p, uvmap)])
    return faces


def _cube_faces():
    """(corners, local 2D coords in [0,1]^2, CCW seen from outside) for a unit cube."""
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return [
        ([(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], sq),  # -Z
        ([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], sq),  # +Z
        ([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], sq),  # -Y
        ([(1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)], sq),  # +Y
        ([(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)], sq),  # -X
        ([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], sq),  # +X
    ]


def run(name, fn):
    print(name)
    try:
        fn()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append(f"{name}: exception")


# --------------------------------------------------------------------------- cases
def t_plane():
    bm, uvl = build([quad(0, 0, 1, 1)])
    m = metrics.analyze(bm, uvl)
    check("plane islands == 1", m["islands"] == 1, m["islands"])
    check("plane overlap == 0", m["uv_overlap_pct"] == 0.0, m["uv_overlap_pct"])
    check("plane crossings == 0", m["uv_boundary_crossings"] == 0, m["uv_boundary_crossings"])
    check("plane worst stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("plane angle error == 0", m["angle_distortion_deg"] == 0.0)
    check("plane flipped == 0", m["flipped_faces"] == 0)
    check("plane mirrored == 0", m["mirrored_islands"] == 0)
    check("plane coverage == 100", m["uv_coverage_pct"] == 100.0, m["uv_coverage_pct"])
    check("plane seams == 0", m["seam_edges"] == 0)
    bm.free()


def t_grid_triangles():
    # 4x4 grid of quads each split into two triangles along the diagonal. The
    # shared diagonals run exactly through texel centres at res 512 (u == v), so
    # an inclusive edge test counts those texels twice (false overlap).
    faces = []
    n = 4
    for i in range(n):
        for j in range(n):
            x0, y0, x1, y1 = i / n, j / n, (i + 1) / n, (j + 1) / n
            faces.append([((x0, y0, 0), (x0, y0)), ((x1, y0, 0), (x1, y0)), ((x1, y1, 0), (x1, y1))])
            faces.append([((x0, y0, 0), (x0, y0)), ((x1, y1, 0), (x1, y1)), ((x0, y1, 0), (x0, y1))])
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("tri grid islands == 1", m["islands"] == 1, m["islands"])
    check("tri grid overlap == 0 (diagonals through texel centres)", m["uv_overlap_pct"] == 0.0,
          m["uv_overlap_pct"])
    check("tri grid stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("tri grid angle == 0", m["angle_distortion_deg"] == 0.0)
    check("tri grid coverage == 100", m["uv_coverage_pct"] == 100.0, m["uv_coverage_pct"])
    bm.free()


def t_concave_ngon():
    # L-shaped hexagon whose first corner is next to the reflex corner, so a
    # triangle fan from corner 0 spills into the notch; the notch is filled by a
    # separate quad sharing two edges with the L. True overlap is 0.
    s = 0.4
    L = [(2, 1), (1, 1), (1, 2), (0, 2), (0, 0), (2, 0)]
    notch = [(1, 1), (2, 1), (2, 2), (1, 2)]
    faces = [[((x, y, 0), (x * s, y * s)) for x, y in L],
             [((x, y, 0), (x * s, y * s)) for x, y in notch]]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("L ngon islands == 1", m["islands"] == 1, m["islands"])
    check("L ngon overlap == 0", m["uv_overlap_pct"] == 0.0, m["uv_overlap_pct"])
    check("L ngon flipped == 0", m["flipped_faces"] == 0, m["flipped_faces"])
    check("L ngon stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("L ngon angle == 0", m["angle_distortion_deg"] == 0.0, m["angle_distortion_deg"])
    check("L ngon coverage == 64", close(m["uv_coverage_pct"], 64.0, 0.05), m["uv_coverage_pct"])
    # numpy LSCM fallback must flatten a flat chart with a concave n-gon perfectly
    glued = [True] * len(bm.edges)
    res = flatten.flatten(list(bm.faces), glued)
    dist, flipped, crossings = flatten.quality(*res)
    check("flatten L ngon: no flipped triangles", flipped == 0.0, flipped)
    check("flatten L ngon: no distortion", dist < 1e-6, dist)
    check("flatten L ngon: chart_ok", flatten.chart_ok(list(bm.faces), glued))
    bm.free()


def t_hexagon_and_triangles():
    # convex hexagon n-gon fan of 6 triangles around it, all exact
    faces = []
    r = 0.2
    c = (0.5, 0.5)
    hexa = [(c[0] + r * math.cos(k * math.pi / 3), c[1] + r * math.sin(k * math.pi / 3)) for k in range(6)]
    faces.append([((x, y, 0), (x, y)) for x, y in hexa])
    R = 0.4
    for k in range(6):
        a, b = hexa[k], hexa[(k + 1) % 6]
        mid = (k + 0.5) * math.pi / 3
        o = (c[0] + R * math.cos(mid), c[1] + R * math.sin(mid))
        faces.append([((b[0], b[1], 0), b), ((a[0], a[1], 0), a), ((o[0], o[1], 0), o)])
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("hexagon+tris islands == 1", m["islands"] == 1, m["islands"])
    check("hexagon+tris overlap == 0", m["uv_overlap_pct"] == 0.0, m["uv_overlap_pct"])
    check("hexagon+tris flipped == 0", m["flipped_faces"] == 0, m["flipped_faces"])
    check("hexagon+tris stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("hexagon+tris angle == 0", m["angle_distortion_deg"] == 0.0)
    bm.free()


def t_cube_cross():
    bm, uvl = build(cube_cross_net())
    m = metrics.analyze(bm, uvl)
    check("cube cross islands == 1", m["islands"] == 1, m["islands"])
    check("cube cross seam edges == 7", m["seam_edges"] == 7, m["seam_edges"])
    check("cube cross seam components == 1", m["seam_components"] == 1)
    # the cut is a tree on the 8 corners: 4 bottom corners are leaves
    check("cube cross dead ends == 4 (tree leaves)", m["seam_dead_ends"] == 4, m["seam_dead_ends"])
    check("cube cross overlap == 0", m["uv_overlap_pct"] == 0.0, m["uv_overlap_pct"])
    check("cube cross crossings == 0", m["uv_boundary_crossings"] == 0, m["uv_boundary_crossings"])
    check("cube cross stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("cube cross area dist == 0", m["area_distortion"] == 0.0)
    check("cube cross angle == 0", m["angle_distortion_deg"] == 0.0)
    check("cube cross flipped == 0", m["flipped_faces"] == 0)
    check("cube cross mirrored == 0", m["mirrored_islands"] == 0)
    check("cube cross seams on hard edges == 100%", m["seam_on_hard_edges_pct"] == 100.0)
    check("cube cross coverage == 37.5", close(m["uv_coverage_pct"], 37.5, 0.05), m["uv_coverage_pct"])
    check("cube cross seam length == 7", close(m["seam_length"], 7.0))
    check("cube cross seam length pct == 58.33", close(m["seam_length_pct"], 58.33, 0.01))
    bm.free()


def t_cube_six():
    bm, uvl = build(cube_six_islands())
    m = metrics.analyze(bm, uvl)
    check("cube six islands == 6", m["islands"] == 6, m["islands"])
    check("cube six seam edges == 12", m["seam_edges"] == 12, m["seam_edges"])
    check("cube six dead ends == 0", m["seam_dead_ends"] == 0, m["seam_dead_ends"])
    check("cube six overlap == 0", m["uv_overlap_pct"] == 0.0)
    check("cube six stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("cube six coverage == 24", close(m["uv_coverage_pct"], 24.0, 0.05), m["uv_coverage_pct"])
    bm.free()


def t_open_pipe_dead_ends():
    # open tube of 8 quads, cut along one straight edge from rim to rim and
    # unrolled exactly. The seam runs border to border: no dangling end.
    n, r, h = 8, 1.0, 1.0
    side = 2 * r * math.sin(math.pi / n)
    s = 1.0 / (n * side)
    faces = []
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        p0 = (r * math.cos(a0), r * math.sin(a0))
        p1 = (r * math.cos(a1), r * math.sin(a1))
        u0, u1 = k * side * s, (k + 1) * side * s
        faces.append([((p0[0], p0[1], 0), (u0, 0)), ((p1[0], p1[1], 0), (u1, 0)),
                      ((p1[0], p1[1], h), (u1, h * s)), ((p0[0], p0[1], h), (u0, h * s))])
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("pipe islands == 1", m["islands"] == 1, m["islands"])
    check("pipe seam edges == 1", m["seam_edges"] == 1, m["seam_edges"])
    check("pipe dead ends == 0 (seam ends on the mesh border)", m["seam_dead_ends"] == 0, m["seam_dead_ends"])
    check("pipe seam on hard edges == 100% (45 deg facet edge)", m["seam_on_hard_edges_pct"] == 100.0)
    check("pipe stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("pipe angle == 0", m["angle_distortion_deg"] == 0.0)
    check("pipe overlap == 0", m["uv_overlap_pct"] == 0.0)
    bm.free()


def t_flat_seam_not_hard():
    # 2 coplanar quads cut apart: the seam lies on a flat edge
    faces = [quad(0, 0, 1, 1, lambda x, y: (0.4 * x, 0.4 * y)),
             quad(1, 0, 2, 1, lambda x, y: (0.5 + 0.4 * (x - 1), 0.4 * y))]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("flat cut islands == 2", m["islands"] == 2, m["islands"])
    check("flat cut seam on hard edges == 0%", m["seam_on_hard_edges_pct"] == 0.0, m["seam_on_hard_edges_pct"])
    check("flat cut dead ends == 0 (cut runs border to border)", m["seam_dead_ends"] == 0, m["seam_dead_ends"])
    bm.free()


def t_two_overlapping_shells():
    # A = [0, .5]^2, B = [.25, .75] x [0, .5]: union .375, intersection .125
    faces = [quad(0, 0, 1, 1, lambda x, y: (0.5 * x, 0.5 * y)),
             quad(0, 0, 1, 1, lambda x, y: (0.25 + 0.5 * x, 0.5 * y), z=1.0)]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("overlap shells islands == 2", m["islands"] == 2)
    check("overlap shells == 33.333%", close(m["uv_overlap_pct"], 33.333, 0.002), m["uv_overlap_pct"])
    # here the shells only touch each other's corners / share a collinear
    # border, so there is no *proper* boundary crossing (documented limit)
    check("overlap shells: touching boundaries are not crossings", m["uv_boundary_crossings"] == 0,
          m["uv_boundary_crossings"])
    bm.free()
    # B moved up by 0.1: intersection .25 x .4 = .1, union .4 -> 25 %
    faces = [quad(0, 0, 1, 1, lambda x, y: (0.5 * x, 0.5 * y)),
             quad(0, 0, 1, 1, lambda x, y: (0.25 + 0.5 * x, 0.1 + 0.5 * y), z=1.0)]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("offset shells overlap == 25%", close(m["uv_overlap_pct"], 25.0, 0.3), m["uv_overlap_pct"])
    check("offset shells crossings == 2", m["uv_boundary_crossings"] == 2, m["uv_boundary_crossings"])
    bm.free()


def t_stacked_shells():
    # two identical shells on top of each other: 100% overlap, no crossings
    faces = [quad(0, 0, 1, 1, lambda x, y: (0.5 * x, 0.5 * y)),
             quad(0, 0, 1, 1, lambda x, y: (0.5 * x, 0.5 * y), z=1.0)]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("stacked shells overlap == 100%", m["uv_overlap_pct"] == 100.0, m["uv_overlap_pct"])
    bm.free()


def t_thin_overlap():
    # sliver overlap 0.0005 wide (narrower than a 1/512 texel, no texel centre
    # inside): raster misses it, the exact boundary test must not
    faces = [quad(0, 0, 1, 1, lambda x, y: (0.1 + 0.1995 * x, 0.1 + 0.2 * y)),
             quad(0, 0, 1, 1, lambda x, y: (0.299 + 0.2 * x, 0.15 + 0.2 * y), z=1.0)]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    print(f"    (raster overlap for the sliver: {m['uv_overlap_pct']} %)")
    check("thin overlap crossings > 0", m["uv_boundary_crossings"] > 0, m["uv_boundary_crossings"])
    bm.free()


def t_tiny_overlap_not_rounded_away():
    # one large shell + overlap of a few texels: must not round to 0.0
    faces = [quad(0, 0, 1, 1, lambda x, y: (x * 0.99, y * 0.99)),
             quad(0, 0, 1, 1, lambda x, y: (0.985 + 0.01 * x, 0.5 + 0.004 * y), z=1.0)]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("tiny overlap reported > 0", m["uv_overlap_pct"] > 0.0, m["uv_overlap_pct"])
    bm.free()


def t_self_overlap_one_shell():
    # one connected strip of 12 quads wrapped 480 degrees round a ring in UV:
    # quads 9..11 lie on quads 0..2 without any flipped face
    n, step = 12, math.radians(40)
    r1, r2, c = 0.2, 0.4, (0.5, 0.5)

    def ring(k, r):
        return (c[0] + r * math.cos(k * step), c[1] + r * math.sin(k * step))
    faces = []
    for k in range(n):
        uvs = [ring(k, r1), ring(k + 1, r1), ring(k + 1, r2), ring(k, r2)]
        co = [(k, 0, 0), (k + 1, 0, 0), (k + 1, 1, 0), (k, 1, 0)]
        faces.append(list(zip(co, uvs)))
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("wrapped strip is one island", m["islands"] == 1, m["islands"])
    check("wrapped strip flipped == 0", m["flipped_faces"] == 0, m["flipped_faces"])
    check("wrapped strip overlap ~= 33.3% (3 of 9 covered quads)", 31.0 < m["uv_overlap_pct"] < 36.0,
          m["uv_overlap_pct"])
    bm.free()
    # same with a 37 degree step: overlapping but not coincident -> boundaries cross
    step = math.radians(37)
    faces = []
    for k in range(n):
        uvs = [ring(k, r1), ring(k + 1, r1), ring(k + 1, r2), ring(k, r2)]
        co = [(k, 0, 0), (k + 1, 0, 0), (k + 1, 1, 0), (k, 1, 0)]
        faces.append(list(zip(co, uvs)))
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("wrapped strip (37 deg) overlap > 0", m["uv_overlap_pct"] > 0)
    check("wrapped strip (37 deg) crossings > 0", m["uv_boundary_crossings"] > 0, m["uv_boundary_crossings"])
    bm.free()


def t_mirrored_and_folded():
    bm, uvl = build([quad(0, 0, 1, 1, lambda x, y: (1 - x, y))])
    m = metrics.analyze(bm, uvl)
    check("mirrored shell: flipped (folds) == 0", m["flipped_faces"] == 0, m["flipped_faces"])
    check("mirrored shell: mirrored_islands == 1", m["mirrored_islands"] == 1, m["mirrored_islands"])
    check("mirrored shell: stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("mirrored shell: overlap == 0", m["uv_overlap_pct"] == 0.0)
    bm.free()
    # 3-quad strip, third quad folded back over the second
    faces = [quad(0, 0, 1, 1, lambda x, y: (0.25 * x, 0.25 * y)),
             quad(1, 0, 2, 1, lambda x, y: (0.25 * x, 0.25 * y)),
             quad(2, 0, 3, 1, lambda x, y: (0.25 * (4 - x), 0.25 * y))]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    check("fold: one island", m["islands"] == 1, m["islands"])
    check("fold: flipped == 1", m["flipped_faces"] == 1, m["flipped_faces"])
    check("fold: mirrored_islands == 0", m["mirrored_islands"] == 0)
    check("fold: overlap == 50% (1 of 2 covered quads doubled)", close(m["uv_overlap_pct"], 50.0, 0.01),
          m["uv_overlap_pct"])
    bm.free()


def t_scaling():
    # uniform scale of a whole shell is not stretch (density is normalised per shell)
    bm, uvl = build([quad(0, 0, 1, 1, lambda x, y: (0.3 * x, 0.3 * y)),
                     quad(1, 0, 2, 1, lambda x, y: (0.3 * x, 0.3 * y))])
    m = metrics.analyze(bm, uvl)
    check("uniform scale: stretch == 0", m["worst_island_area_distortion"] == 0.0)
    check("uniform scale: area dist == 0", m["area_distortion"] == 0.0)
    bm.free()
    # one face at twice the UV area of its neighbour (both rectangles): stretch
    # = mean |log r| with r = 4/3 and 2/3 -> (ln(4/3) + ln(3/2)) / 2 = 0.3466
    bm, uvl = build([quad(0, 0, 1, 1, lambda x, y: (0.2 * x, 0.1 * y)),
                     quad(1, 0, 2, 1, lambda x, y: (0.2 + 0.1 * (x - 1), 0.1 * y))])
    m = metrics.analyze(bm, uvl)
    expect = (math.log(4 / 3) + math.log(3 / 2)) / 2
    check("2:1 area: worst stretch == 0.3466", close(m["worst_island_area_distortion"], expect, 1e-4),
          m["worst_island_area_distortion"])
    check("2:1 area: area dist == 0.3466", close(m["area_distortion"], expect, 1e-4), m["area_distortion"])
    check("2:1 area: angle == 0 (all rectangles)", m["angle_distortion_deg"] == 0.0)
    bm.free()
    # anisotropic stretch of a whole single-face shell: invisible to area
    # distortion (by definition), visible as angle error on a non-rectangle
    tri = [((0, 0, 0), (0, 0)), ((1, 0, 0), (0.5, 0)), ((0, 1, 0), (0, 1))]
    bm, uvl = build([tri])
    m = metrics.analyze(bm, uvl)
    check("anisotropic: area stretch == 0", m["worst_island_area_distortion"] == 0.0)
    # 3D angles 90/45/45, UV angles 90/63.43/26.57 -> mean |err| = 2*18.43/3
    check("anisotropic: angle error == 12.29 deg", close(m["angle_distortion_deg"], 12.29, 0.01),
          m["angle_distortion_deg"])
    bm.free()


def t_collapsed_uvs():
    # a fresh UV layer has every corner at (0, 0): that is not "no stretch"
    bm, uvl = build([quad(0, 0, 1, 1, lambda x, y: (0, 0)), quad(1, 0, 2, 1, lambda x, y: (0, 0))])
    m = metrics.analyze(bm, uvl)
    check("collapsed UVs: worst stretch is large", m["worst_island_area_distortion"] > 10,
          m["worst_island_area_distortion"])
    check("collapsed UVs: area dist is large", m["area_distortion"] > 10, m["area_distortion"])
    check("collapsed UVs: coverage 0", m["uv_coverage_pct"] == 0.0)
    bm.free()


def t_degenerate():
    # no UV layer
    bm, uvl = build([quad(0, 0, 1, 1)], uv_layer=False)
    m = metrics.analyze(bm, None)
    check("no UV layer: no crash, islands == 0", m["islands"] == 0, m)
    check("no UV layer: same keys as normal", set(m) == set(_ref_keys()), sorted(set(m) ^ set(_ref_keys())))
    bm.free()
    # empty mesh with a UV layer
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    m = metrics.analyze(bm, uvl)
    check("empty mesh: islands == 0", m["islands"] == 0)
    check("empty mesh: overlap == 0", m["uv_overlap_pct"] == 0.0)
    check("empty mesh: flipped == 0", m["flipped_faces"] == 0)
    bm.free()
    # zero-area face next to a good one (collinear triangle sharing an edge)
    faces = [quad(0, 0, 1, 1, lambda x, y: (0.5 * x, 0.5 * y)),
             [((1, 0, 0), (0.5, 0)), ((1, 1, 0), (0.5, 0.5)), ((1, 0.5, 0), (0.5, 0.25))]]
    bm, uvl = build(faces)
    m = metrics.analyze(bm, uvl)
    finite = all(v is None or not isinstance(v, float) or math.isfinite(v) for v in m.values())
    check("zero-area face: all metrics finite", finite, m)
    check("zero-area face: stretch == 0", m["worst_island_area_distortion"] == 0.0,
          m["worst_island_area_distortion"])
    check("zero-area face: flipped == 0", m["flipped_faces"] == 0)
    check("zero-area face: overlap == 0", m["uv_overlap_pct"] == 0.0, m["uv_overlap_pct"])
    bm.free()
    # flatten on degenerate input
    bm, uvl = build([[((0, 0, 0), (0, 0)), ((1, 0, 0), (0, 0)), ((2, 0, 0), (0, 0))]])
    check("flatten: zero-area chart -> None", flatten.flatten(list(bm.faces), [True] * len(bm.edges)) is None)
    bm.free()


def _ref_keys():
    bm, uvl = build([quad(0, 0, 1, 1)])
    keys = list(metrics.analyze(bm, uvl))
    bm.free()
    return keys


def t_huge_uv_range():
    # UVs far outside 0..1 (raw world units): the raster must stay bounded
    bm, uvl = build([quad(0, 0, 1, 1, lambda x, y: (300 * x, 300 * y)),
                     quad(0, 0, 1, 1, lambda x, y: (150 + 300 * x, 300 * y), z=1)])
    t = time.perf_counter()
    m = metrics.analyze(bm, uvl)
    dt = time.perf_counter() - t
    check("huge UV range: fast (< 5 s)", dt < 5.0, f"{dt:.1f}s")
    check("huge UV range: overlap == 33.3%", close(m["uv_overlap_pct"], 33.333, 0.5), m["uv_overlap_pct"])
    bm.free()


def t_flatten_basic():
    # planar 3x3 grid: LSCM reproduces it exactly
    faces = []
    for i in range(3):
        for j in range(3):
            faces.append(quad(i, j, i + 1, j + 1))
    bm, uvl = build(faces)
    glued = [True] * len(bm.edges)
    dist, flipped, crossings = flatten.quality(*flatten.flatten(list(bm.faces), glued))
    check("flatten grid: dist ~ 0", dist < 1e-6, dist)
    check("flatten grid: no flips/crossings", flipped == 0 and crossings == 0, (flipped, crossings))
    bm.free()
    # closed cube with no cut cannot be flattened
    faces = [[(c, (0, 0)) for c in p] for p, _ in _cube_faces()]
    bm, uvl = build(faces)
    check("flatten closed cube: refused", not flatten.chart_ok(list(bm.faces), [True] * len(bm.edges)))
    bm.free()
    # cross-net cut: flat-foldable, only the 8 corner cones distort it
    bm, uvl = build(cube_cross_net())
    glued = [False] * len(bm.edges)
    seam_free = set()
    for e in bm.edges:
        a, b = e.link_loops
        a1, a2 = a[uvl].uv, a.link_loop_next[uvl].uv
        b1, b2 = (b.link_loop_next[uvl].uv, b[uvl].uv) if b.vert is not a.vert else (b[uvl].uv, b.link_loop_next[uvl].uv)
        glued[e.index] = (a1 - b1).length < 1e-6 and (a2 - b2).length < 1e-6
        if glued[e.index]:
            seam_free.add(e.index)
    check("cross net has 5 glued edges", len(seam_free) == 5, len(seam_free))
    dist, flipped, crossings = flatten.quality(*flatten.flatten(list(bm.faces), glued))
    check("flatten cross net: no flips, no crossings", flipped == 0 and crossings == 0, (flipped, crossings))
    bm.free()


def t_boundary_crossings_small_segments():
    # two short boundary segments (1/1000 of the chart size) crossing by 2e-5:
    # a length-blind tolerance misses this
    import numpy as np
    uv = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],               # big triangle (sets the span)
        [0.5, 0.5], [0.501, 0.5], [0.5005, 0.5003],       # small triangle 1
        [0.50045, 0.50002], [0.5005, 0.4995], [0.50055, 0.50002],  # small triangle 2 poking up
    ])
    T = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    check("boundary_crossings detects tiny crossing", flatten.boundary_crossings(uv, T) > 0,
          flatten.boundary_crossings(uv, T))


def main():
    cases = [(n, f) for n, f in globals().items() if n.startswith("t_") and callable(f)]
    for name, fn in cases:
        run(name, fn)
    print(f"\n{CHECKS[0]} checks, {len(FAILURES)} failures")
    for f in FAILURES:
        print("FAILED:", f)
    sys.exit(1 if FAILURES else 0)


# Blender exits with status 0 after an uncaught exception in a --python script
try:
    main()
except SystemExit:
    raise
except BaseException:  # noqa: BLE001
    traceback.print_exc()
    sys.exit(1)
