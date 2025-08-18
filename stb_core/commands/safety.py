# stb_core/commands/safety.py
from typing import Tuple
from .schema import Command

_ALLOWED_PRIMITIVES = {"CUBE","UV_SPHERE","ICO_SPHERE","CYLINDER","CONE","TORUS"}
_ALLOWED_IMPORT_EXT = {"fbx","obj","glb","gltf"}

def is_allowed(cmd: Command, cfg) -> Tuple[bool, str]:
    wl = set(cfg.get("safety", {}).get("whitelist_ops", []))
    if cmd.type not in wl:
        return False, f"Command {cmd.type} not allowed by whitelist"

    t = cmd.type
    a = cmd.args

    if t == "ADD_MESH":
        prim = str(a.get("primitive", "CUBE")).upper()
        if prim not in _ALLOWED_PRIMITIVES:
            return False, f"Primitive {prim} not allowed"

    elif t == "IMPORT":
        path = a.get("path", "")
        if not isinstance(path, str) or "." not in path:
            return False, "IMPORT requires a file path"
        ext = path.rsplit(".", 1)[-1].lower()
        if ext not in _ALLOWED_IMPORT_EXT:
            return False, f"File type .{ext} not allowed"

    elif t == "TRANSFORM":
        # Example: {"translate":[x,y,z]} — all numbers
        tr = a.get("translate")
        if tr is not None:
            if not (isinstance(tr, (list, tuple)) and len(tr) == 3 and all(isinstance(x, (int,float)) for x in tr)):
                return False, "TRANSFORM.translate must be [x,y,z] numbers"

    elif t == "SET_CAMERA":
        # just require a name; existence checked later
        if not a.get("name"):
            return False, "SET_CAMERA requires a 'name'"

    # SET_MATERIAL, RENDER kept permissive for now
    return True, ""
