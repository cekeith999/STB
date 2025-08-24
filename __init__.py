bl_info = {
    "name": "SpeechToBlender",
    "author": "Jordan",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D View > N-panel > STB",
    "description": "Voice control + Meshy integration for Blender",
    "category": "3D View",
}

_load_err = ""

def register():
    global _load_err
    _load_err = ""
    try:
        import importlib, sys, pathlib
        HERE = pathlib.Path(__file__).resolve().parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        # Lazy import the real add-on only now
        mod = importlib.import_module(".addon", package=__package__)
        importlib.reload(mod)  # in case we’re reloading
        mod.register()
    except Exception as e:
        import traceback
        _load_err = traceback.format_exc()
        print("[SpeechToBlender] STARTUP ERROR:\n", _load_err)

def unregister():
    try:
        import sys
        mod = sys.modules.get(__package__ + ".addon")
        if mod and hasattr(mod, "unregister"):
            mod.unregister()
    except Exception as e:
        print("[SpeechToBlender] unregister note:", e)
