# addon/command_exec.py
import bpy
from typing import Dict, Any
from stb_core.commands.schema import Command
from stb_core.commands.safety import is_allowed

# --- import helpers ---
import bpy, os

def _ensure_collection(name: str = "Meshy_Imports"):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
    # make sure it's linked into the Scene
    if coll not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    return coll

def _import_file_to_collection(path: str, collection_name: str = "Meshy_Imports"):
    ext = os.path.splitext(path)[1].lower()
    coll = _ensure_collection(collection_name)

    # remember current objects to detect what's new
    before = set(bpy.data.objects.keys())

    # run the appropriate importer
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
    else:
        return {"ok": False, "error": f"Unsupported format: {ext}", "path": path}

    # detect newly added objects
    added_names = list(set(bpy.data.objects.keys()) - before)
    added = [bpy.data.objects[n] for n in added_names]

    # link new objects to our collection and unlink from others
    for o in added:
        if o.name not in coll.objects:
            coll.objects.link(o)
        for c in list(o.users_collection):
            if c != coll:
                c.objects.unlink(o)

    # select them and set an active object
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in added:
        o.select_set(True)
    if added:
        bpy.context.view_layer.objects.active = added[0]

    return {"ok": True, "count": len(added), "collection": coll.name, "files": [{"path": path}]}



def execute_command(cmd_dict: Dict[str, Any], CFG: Dict[str, Any]) -> Dict[str, Any]:
    """
    cmd_dict: {"type": "ADD_MESH", "args": {...}}
    returns: {"ok": True} or {"ok": False, "error": "..."}
    """
    try:
        cmd = Command(type=cmd_dict.get("type"), args=cmd_dict.get("args", {}))
    except Exception as e:
        return {"ok": False, "error": f"Invalid command shape: {e}"}

    ok, reason = is_allowed(cmd, CFG)
    if not ok:
        print(f"[SAFETY BLOCK] {reason}")
        return {"ok": False, "error": reason}

    t = cmd.type
    a = cmd.args

    try:
        if t == "ADD_MESH":
            prim = str(a.get("primitive","CUBE")).upper()
            if prim == "CUBE":
                bpy.ops.mesh.primitive_cube_add()
            elif prim == "UV_SPHERE":
                bpy.ops.mesh.primitive_uv_sphere_add()
            elif prim == "ICO_SPHERE":
                bpy.ops.mesh.primitive_ico_sphere_add()
            elif prim == "CYLINDER":
                bpy.ops.mesh.primitive_cylinder_add()
            elif prim == "CONE":
                bpy.ops.mesh.primitive_cone_add()
            elif prim == "TORUS":
                bpy.ops.mesh.primitive_torus_add()
            else:
                raise ValueError(f"Unsupported primitive: {prim}")

        elif t == "IMPORT":
            path = a["path"]
            ext = path.rsplit(".", 1)[-1].lower()
            if ext == "fbx":
                bpy.ops.import_scene.fbx(filepath=path)
            elif ext == "obj":
                bpy.ops.wm.obj_import(filepath=path)
            elif ext in {"glb","gltf"}:
                bpy.ops.import_scene.gltf(filepath=path)
            else:
                raise ValueError(f"Unsupported import type: {ext}")

        elif t == "TRANSFORM":
            obj = bpy.context.active_object
            if not obj:
                raise RuntimeError("No active object to transform")
            if "translate" in a:
                dx, dy, dz = a["translate"]
                obj.location.x += dx
                obj.location.y += dy
                obj.location.z += dz

        elif t == "RENDER":
            bpy.ops.render.render(animation=a.get("animation", False), write_still=True)

        elif t == "SET_MATERIAL":
            # Placeholder hook (we’ll flesh this out later)
            pass

        elif t == "SET_CAMERA":
            cam_name = a.get("name")
            cam = bpy.data.objects.get(cam_name) if cam_name else None
            if not (cam and cam.type == 'CAMERA'):
                raise RuntimeError(f"Camera not found or invalid: {cam_name}")
            bpy.context.scene.camera = cam

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
