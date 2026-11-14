"""Small helpers shared by the headless scripts.

Blender does not resolve relative paths against the shell's current directory
(on Windows "preview/out.blend" ends up at the drive root), so every output
path goes through out_path() before it reaches Blender.
"""

import os
import sys
import traceback

import bpy


class UsageError(Exception):
    pass


def script_args():
    """Arguments after "--" on the Blender command line."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def out_path(path):
    """Absolute path against the process cwd, with its folder created."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def mesh_object(name):
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != 'MESH':
        meshes = sorted(o.name for o in bpy.data.objects if o.type == 'MESH')
        raise UsageError(f"no mesh object named {name!r}; meshes in this file: {', '.join(meshes) or 'none'}")
    return ob


def run(main, usage):
    """Run main(); exit Blender with a non-zero code on any error.

    Without this, an exception in a --python script only prints a traceback and
    Blender still exits with code 0.
    """
    try:
        main()
    except UsageError as e:
        print(f"ERROR: {e}\nusage: {usage}", file=sys.stderr)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
