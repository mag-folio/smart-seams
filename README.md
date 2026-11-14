# Smart Seams

**Automatic, artist-style UV seams for Blender.** Smart Seams splits a mesh into the
panels an artist would see, cuts only along complete panel borders, hides seams where
the camera rarely looks, and merges panels into as few UV shells as the object needs.
It never allows overlap, and it only accepts stretch within a strict limit. Rarely seen
shells (undersides, faces on the floor, inner faces) get less texture space.

Built because Blender's *Smart UV Project* and Substance Painter's auto-unwrap tend to:

- ignore clean edge loops and cut through the middle of surfaces,
- make no attempt to hide seams,
- shatter one object into many tiny shells.

| | Original UVs (table) | Smart Seams (table) |
|---|---|---|
| Shells on model (seams in red) | ![](docs/images/table_before_top.png) | ![](docs/images/table_after_top.png) |
| From below | ![](docs/images/table_before_bottom.png) | ![](docs/images/table_after_bottom.png) |
| Checker texture (stretch test) | ![](docs/images/table_before_checker.png) | ![](docs/images/table_after_checker.png) |
| UV layout | ![](docs/images/table_before_uv.png) | ![](docs/images/table_after_uv.png) |

| Cube | Cylinder | Lamp base |
|---|---|---|
| ![](docs/images/cube_after_checker.png) | ![](docs/images/cylinder_after_checker.png) | ![](docs/images/lamp_base_after_checker.png) |
| ![](docs/images/cube_after_uv.png) | ![](docs/images/cylinder_after_uv.png) | ![](docs/images/lamp_base_after_uv.png) |
| 1 shell: the four sides as one strip, caps above and below | 3 shells (wall + 2 caps) | 4 shells: top with holes, bottom, side band, triangle recess |

## Results

Measured by [`scripts/run_test.py`](scripts/run_test.py) on the test scene (`test_data/`):

| Object | UVs | Shells | UV overlap | Worst shell stretch¹ | Mean angle error | Flipped faces | Seam length |
|---|---|---|---|---|---|---|---|
| Cube | Smart Seams | **1** | 0 % | 0.00 | 0.0° | 0 | 7 edges |
| Cylinder | Smart Seams | **3** | 0 % | 0.00 | 0.0° | 0 | |
| Lamp base | hand-made reference | 3 | 2.4 % | 9.63 | 31.0° | 82 | 5.6 |
| Lamp base | Smart Seams | 4 | **0 %** | **0.06** | **0.1°** | **0** | 6.6 |
| Table | previous auto-unwrap | 19 | 17.4 % | 1.05 | 1.8° | 200 | 58.6 |
| Table | Smart Seams | **10** | **0 %** | **0.10** | 5.6° | **0** | **49.5** |

¹ area-weighted mean of |log(UV area / 3D area)| inside the shell (0 = no stretch).

The shell count follows the object's complexity: one shell for a cube, three for a
cylinder, four for the lamp base, and the 10-shell limit for the table. On the table, every
shell flattens with no overlap and no flipped faces, using shorter and less visible seams
than the original auto-unwrap. The table's mean angle error is higher (5.6°) because all 10
shells had to be complete chunks; the checker render shows it stays even.

## How it works

Artists don't place seams edge by edge. They see an object as a set of panels (the
tabletop, each side of a leg, a hole wall, a bevel band) and decide which panels share a UV
shell. The solver ([`core/panels.py`](smart_seams/core/panels.py)) works the same way:

1. **Visibility per face.** Ambient-occlusion style ray casting against the object itself plus
   a virtual ground plane at its lowest point, combined with down-facing and (optional) front
   direction / scene camera. Faces between legs, under the tabletop or on the floor score low.
2. **Panels.** Faces are flood-filled across smooth edges (below the *Panel Angle*). Seams only
   ever run along complete panel borders. A seam can therefore never stop part way round a
   hole, and no stray "whiskers" or relief cuts are created.
3. **Tube cuts.** A panel that wraps round (cylinder wall, hole wall, a smooth leg) gets exactly
   one cut. Stacked quad rings are detected and cut along the same **edge loop**, at the least
   visible column, so the cut is one straight seam instead of a staircase.
4. **Splitting.** A panel that still cannot flatten without stretching is split along its
   dominant normal direction, recursively.
5. **Side bands.** A ring of side panels (the four sides of a box, the facets of a prism, a
   recess wall) is treated like one tube panel: it is joined into a single strip and cut once,
   on the least visible vertical border. A texture then flows round the sides with one seam,
   and the caps attach to the strip (band plus caps, the way an artist lays out a box).
6. **Shells.** Panels are merged into shells. Tiny and thin panels (bevel bands, small trims) go
   first, onto the neighbour they share the longest border with, so they lie flat instead of
   becoming skinny strips. After that, merges happen across the most visible borders first, so
   the surviving seams are the hidden ones. A cap folds open along one straight edge, the most
   visible one, so its remaining borders are the hidden seams. Hole rims are always cut all the
   way round or not at all.
7. **Validation with Blender's own unwrapper.** Every candidate merge is unwrapped on a hidden
   temporary mesh with the same solver the final unwrap uses. It is refused if any face flips,
   if the UV boundary crosses itself (overlap), or if stretch exceeds the current limit. The limit
   starts strict (0.08) and is relaxed step by step, up to the *Stretch Ceiling*, only while the
   object is still above *Max Shells*.
8. **Layout.** Final unwrap; all-quad strips are straightened with *Follow Active Quads* when that
   adds no stretch or overlap; shells are normalised to equal texel density, then scaled by
   visibility (*Hidden Shell Scale*). Grid-shaped shells are turned to line up with the UV grid,
   with the object's up direction pointing up, and keep that rotation while packing.

Related reading: Sheffer & Hart, *Seamster: Inconspicuous Low-Distortion Texture Seam Layout*
(2002); Lévy et al., *Least Squares Conformal Maps* (2002); Garland et al., *Hierarchical Face
Clustering* (2001); Hoffman & Richards, *Parts of Recognition* (1984).

## Install

Requires Blender 4.2 or newer (tested on 5.0.1).

1. Download `smart_seams-<version>.zip` from Releases, or build it:
   `blender --command extension build --source-dir smart_seams --output-dir dist`
2. In Blender: *Edit › Preferences › Get Extensions › ⌄ › Install from Disk…* and pick the zip.

## Use

Select one or more mesh objects, open the sidebar (`N`) › **Smart Seams**, press
**Smart Seams Unwrap**. Also available from *UV › Smart Seams Unwrap* in Edit Mode.

- **Mark Smart Seams** only marks seams so you can review or tweak them before unwrapping.
- **Analyze** measures the current UVs (shells, seam length/visibility, stretch, overlap).

| Setting | What it does |
|---|---|
| Max Shells | Hard limit on UV shells; simple objects use far fewer |
| Stretch Ceiling | How far the stretch limit may be relaxed to stay under Max Shells (overlap is never allowed) |
| Panel Angle | Edges sharper than this separate panels; seams only run along complete panel borders |
| Front / Sits on Ground | Visibility cues used to hide seams and shrink hidden shells |
| Hidden Shell Scale | Texel density of fully hidden shells relative to visible ones |
| Straighten Strips | Follow Active Quads on quad strips when it adds no stretch or overlap |

## Development

```
smart_seams/                 the add-on (Blender extension)
  core/visibility.py         occlusion + facing -> per-face visibility
  core/panels.py             panel solver: panels, tube cuts, splitting, shell merging
  core/seams.py              seam cost, edge loops, aligned ring cuts
  core/blender_validator.py  trial unwrap of candidate shells with Blender's unwrapper
  core/flatten.py            numpy LSCM fallback + boundary self-intersection test
  core/layout.py             straightening, visibility-weighted texel density
  core/pipeline.py           end-to-end pipeline
  core/metrics.py            UV quality metrics
scripts/                     headless tests, preview renderer, scene builder
test_data/                   test meshes (table, lamp base, cube, cylinder)
```

Run the tests and render previews headlessly:

```
blender -b test_data/uv_test.blend --factory-startup --python scripts/test_addon.py
blender -b test_data/uv_test.blend --factory-startup --python scripts/run_test.py -- out.blend
blender -b out.blend --factory-startup --python scripts/render_preview.py -- table preview/table
```

`test_addon.py` registers the add-on, runs it on every test object, and fails if any object
exceeds its expected shell count, overlaps, flips faces or stretches past the ceiling.

## Roadmap

- Lower the table's angle error further: when a merge fails only on stretch, try re-cutting a
  tube at a different column before relaxing the limit.
- Symmetry-aware seams (mirror seam placement on symmetric meshes).
- Batch mode for whole collections.

## License

GPL-3.0-or-later (required for Blender extensions). See [LICENSE](LICENSE).
