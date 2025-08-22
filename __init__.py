bl_info = {
    "name": "Speech To Blender",
    "author": "Jordan",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D View > N-panel > STB",
    "description": "Voice control + Meshy integration for Blender",
    "category": "3D View",
}

# Make sure this package root is importable
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Delegate register/unregister to your real module
from .addon import register as _register, unregister as _unregister

def register():
    _register()

def unregister():
    _unregister()
