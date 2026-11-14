"""Save a .blend for publishing: scene data only, no UI state.

A normal save stores window/screen/file-browser state, which includes local
folders (for example the last *Save As* directory, often inside the user's
home folder). Writing only the scene and its dependencies keeps published
test files free of personal paths.
"""

import bpy


def save_clean(filepath):
    scene = bpy.context.scene
    bpy.data.libraries.write(filepath, {scene}, path_remap='NONE', compress=True)
