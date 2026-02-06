bl_info = {
    "name": "Speech To Blender",
    "author": "STB",
    "version": (0, 5, 0),
    "blender": (3, 6, 0),
    "category": "System",
    "description": "Voice Tools for Blender",
}

import bpy
import os
import sys
import threading
import queue
import socket
import time
import subprocess
import shutil
import bmesh
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from bpy.props import StringProperty, EnumProperty, BoolProperty
from bpy.types import AddonPreferences, Operator, Panel

ADDON_ROOT = (__package__ or __name__).split(".")[0]  # "SpeechToBlender"

# ───────── RPC Server State ─────────
HOST = "127.0.0.1"
PORT = 8765  # Match voice_to_blender.py expectation
_SERVER_THREAD = None
_SERVER = None
_SERVER_RUNNING = False
_TASKQ = queue.Queue()
_NEED_UNDO_PUSH = False  # Flag to track when to push undo point for voice commands

# ───────── Voice Process State ─────────
_VOICE_POPEN = None
_VOICE_RUNNING = False
_VOICE_LISTENING_ENABLED = True  # Toggle for voice listening (Alt+F)
DEFAULT_VOICE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_to_blender.py")

# ───────── Live pipeline: Action Queue (Phase 3) ─────────
# Thread-safe queue: Listener thread pushes, Modal Executor pops on main thread.
# Message format: {"type": "PREDICT"|"CODE"|"ERROR", "payload": str}
#   PREDICT = instant primitive (e.g. "cube", "sphere", "iphone")
#   CODE   = Python/Blender code to exec (e.g. "bpy.ops.mesh.primitive_cube_add(...)")
#   ERROR  = error message to show (e.g. console)
_LIVE_ACTION_QUEUE = queue.Queue()
_LIVE_SESSION_RUNNING = False  # Set True when Start Live; False when Stop Live

# ───────── Preferences (single class) ─────────
class STB_AddonPreferences(AddonPreferences):
    bl_idname = ADDON_ROOT

    meshy_api_key: StringProperty(
        name="Meshy API Key",
        description="Stored locally in Blender prefs",
        subtype="PASSWORD",
        default="",
    )
    meshy_base_url: StringProperty(
        name="Meshy Base URL",
        default="https://api.meshy.ai",
    )
    meshy_model: StringProperty(
        name="Meshy Model",
        description="e.g. v5",
        default="v5",
    )
    meshy_mode: EnumProperty(
        name="Mode",
        items=[
            ("preview", "Preview", "Faster, lower cost"),
            ("standard", "Standard", "Full quality"),
        ],
        default="standard",
    )
    meshy_formats: StringProperty(
        name="Export Formats",
        description="Comma separated: glb,fbx,obj",
        default="glb,fbx,obj",
    )
    
    use_react_reasoning: bpy.props.BoolProperty(
        name="Use ReAct Reasoning",
        description="Let GPT iterate with Thought/Action/Observation loops for complex requests (slower, more API calls)",
        default=False,
    )
    
    def _update_openai_key_part1(self, context):
        """Update callback to strip whitespace from part 1."""
        if self.openai_api_key_part1:
            stripped = self.openai_api_key_part1.strip()
            if stripped != self.openai_api_key_part1:
                self.openai_api_key_part1 = stripped
    
    def _update_openai_key_part2(self, context):
        """Update callback to strip whitespace from part 2."""
        if self.openai_api_key_part2:
            stripped = self.openai_api_key_part2.strip()
            if stripped != self.openai_api_key_part2:
                self.openai_api_key_part2 = stripped
    
    openai_api_key_part1: StringProperty(
        name="OpenAI API Key (Part 1)",
        description="First part of API key (paste first ~80 characters here)",
        subtype="PASSWORD",
        default="",
        maxlen=100,
        update=_update_openai_key_part1,
    )
    
    openai_api_key_part2: StringProperty(
        name="OpenAI API Key (Part 2)",
        description="Second part of API key (paste remaining characters here)",
        subtype="PASSWORD",
        default="",
        maxlen=100,
        update=_update_openai_key_part2,
    )
    
    # Legacy field for backward compatibility (will be combined with parts)
    openai_api_key: StringProperty(
        name="OpenAI API Key (Legacy)",
        description="Legacy field - use Part 1 and Part 2 instead",
        subtype="PASSWORD",
        default="",
        maxlen=200,
    )
    
    # Modeling context (optional hint; real-time mode uses context injection instead)
    super_mode_target_object: StringProperty(
        name="Modeling Target",
        description="Optional context hint for GPT (e.g., 'Echo Dot', 'Coffee Mug'). Not required to speak.",
        default="",
    )
    enable_quality_assessment: bpy.props.BoolProperty(
        name="Enable Quality Assessment",
        description="Run assess_object_quality in voice pipeline (turn-based ReAct). Disabled for real-time Live mode.",
        default=False,
    )
    
    # AI Model Selection
    ai_model_provider: EnumProperty(
        name="AI Model Provider",
        description="Choose which AI model to use for command understanding",
        items=[
            ("openai-gpt-5", "OpenAI GPT-5", "OpenAI GPT-5 (latest, most capable)"),
            ("openai-gpt-4o", "OpenAI GPT-4o", "OpenAI GPT-4o (Vision capable, high quality)"),
            ("google-gemini-3-pro", "Google Gemini 3 Pro", "Google Gemini 3 Pro (Vision capable, high quality)"),
            ("google-gemini-3-fast", "Google Gemini 3 Fast", "Google Gemini 3 Fast (Vision capable, faster)"),
        ],
        default="openai-gpt-5",
    )
    
    gemini_api_key: StringProperty(
        name="Google Gemini API Key",
        description="Get from https://aistudio.google.com/app/apikey",
        subtype="PASSWORD",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        
        # Meshy Settings
        box = layout.box()
        box.label(text="Meshy Settings", icon="MESH_CUBE")
        col = box.column(align=True)
        col.prop(self, "meshy_api_key")
        col.prop(self, "meshy_base_url")
        col.prop(self, "meshy_model")
        col.prop(self, "meshy_mode")
        col.prop(self, "meshy_formats")
        
        # Voice/AI Settings
        box = layout.box()
        box.label(text="Voice & AI Settings", icon="SPEAKER")
        col = box.column(align=True)
        
        # Model Selection
        col.label(text="AI Model Selection:", icon="SETTINGS")
        col.prop(self, "ai_model_provider", icon="WORLD")
        
        # Show API key fields based on selected provider
        if self.ai_model_provider.startswith("openai"):
            col.separator()
            col.label(text="OpenAI API Key (split into two parts):", icon="INFO")
            col.prop(self, "openai_api_key_part1")
            col.prop(self, "openai_api_key_part2")
            
            # Show combined length
            part1_len = len(self.openai_api_key_part1) if self.openai_api_key_part1 else 0
            part2_len = len(self.openai_api_key_part2) if self.openai_api_key_part2 else 0
            total_len = part1_len + part2_len
            
            if total_len > 0:
                col.separator()
                col.label(text=f"Combined key length: {total_len} characters", icon="INFO")
                if total_len < 150:
                    col.label(text="⚠️ Key may be incomplete (expected ~164 chars)", icon="ERROR")
                elif total_len >= 150:
                    col.label(text="✅ Key length looks good", icon="CHECKMARK")
        elif self.ai_model_provider.startswith("google"):
            col.separator()
            col.label(text="Google Gemini API Key:", icon="INFO")
            col.prop(self, "gemini_api_key")
            if self.gemini_api_key:
                key_len = len(self.gemini_api_key)
                col.label(text=f"Key length: {key_len} characters", icon="INFO")
                if key_len < 30:
                    col.label(text="⚠️ Key may be incomplete", icon="ERROR")
                else:
                    col.label(text="✅ Key length looks good", icon="CHECKMARK")
        
        col.separator()
        col.prop(self, "use_react_reasoning", icon="LOOP_BACK")
        col.label(text="ReAct enables step-by-step reasoning (slower, more API credits)", icon="INFO")
        
        col.separator()
        col.label(text="Used for natural language command understanding", icon="INFO")
        
        # Modeling Context (optional; real-time uses context injection)
        box = layout.box()
        box.label(text="Modeling Context (optional)", icon="LIGHT")
        col = box.column(align=True)
        col.prop(self, "super_mode_target_object")
        col.prop(self, "enable_quality_assessment")
        col.separator()
        col.label(text="Target is optional hint. Quality assessment is for turn-based ReAct only.", icon="INFO")


# ───────── Minimal operator and panels so UI never crashes ─────────
class STB_OT_MeshyGenerate(bpy.types.Operator):
    bl_idname = "stb.meshy_generate"
    bl_label = "Generate with Meshy"

    def execute(self, context):
        try:
            from .stb_core.providers import meshy as meshy_provider
        except Exception as e:
            self.report({'ERROR'}, f"core import failed: {e}")
            return {'CANCELLED'}

        prompt = context.window_manager.stb_meshy_prompt or "simple low-poly test object"

        # 👇 get prefs safely
        try:
            addon_prefs = context.preferences.addons[ADDON_ROOT].preferences
        except Exception:
            addon_prefs = None

        cfg = {
            "api_key": getattr(addon_prefs, "meshy_api_key", "") or os.environ.get("MESHY_API_KEY", ""),
            "base_url": getattr(addon_prefs, "meshy_base_url", "https://api.meshy.ai"),
            "model":    getattr(addon_prefs, "meshy_model", "v5"),
            "mode":     getattr(addon_prefs, "meshy_mode", "standard"),
            "formats":  getattr(addon_prefs, "meshy_formats", "glb,fbx,obj"),
        }

        try:
            job = meshy_provider.generate_from_prompt(context, prompt, cfg=cfg)
            self.report({'INFO'}, f"Meshy job submitted: {job}")
        except Exception as e:
            self.report({'ERROR'}, f"Meshy call failed: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


class STB_PT_MeshyTools(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STB"
    bl_label = "Meshy Tools"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        layout.prop(wm, "stb_meshy_prompt", text="Prompt")
        layout.operator("stb.meshy_generate", icon="MESH_CUBE", text="Generate")


class STB_PT_MeshyStatus(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STB"
    bl_label = "Meshy Status"
    bl_parent_id = ""
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        try:
            from .stb_core.providers import meshy as meshy_provider
            status = meshy_provider.get_meshy_status()
        except Exception as e:
            status = f"core not ready: {e}"
        layout.label(text=f"Status: {status}")


# ───────── RPC Safety Gate ─────────
def _is_safe_op(op_fullname: str):
    """Check if an operator is safe to run. Returns (is_allowed, reason)."""
    if not op_fullname:
        return False, "Empty operator name"
    
    # Dangerous operators to block
    dangerous = [
        "file.quit",
        "wm.quit_blender",
        "script.reload",
        "preferences.addon_disable",
        "preferences.addon_remove",
    ]
    if op_fullname in dangerous:
        return False, f"Blocked: {op_fullname} is dangerous"
    
    # Allow common safe operators
    safe_prefixes = [
        "mesh.", "object.", "transform.", "view3d.", "wm.redraw",
        "import_scene.", "wm.obj_import", "wm.gltf_import", "wm.usd_import",
    ]
    for prefix in safe_prefixes:
        if op_fullname.startswith(prefix):
            return True, "Allowed by prefix whitelist"
    
    # Default: allow but warn
    return True, "Allowed (not in deny list)"


def _ensure_import_addon_for_operator(op_fullname: str) -> bool:
    """Enable required import add-on for well-known operators if missing.
    Returns True if an add-on was enabled (or already enabled), False otherwise.
    """
    try:
        import bpy
        op = op_fullname.lower()
        mod = None
        if op == "import_mesh.stl":
            mod = "io_mesh_stl"
        elif op == "import_mesh.ply":
            mod = "io_mesh_ply"
        elif op == "import_scene.fbx":
            mod = "io_scene_fbx"
        elif op == "import_scene.obj" or op == "wm.obj_import":
            mod = "io_scene_obj"
        elif op == "import_scene.gltf" or op == "wm.gltf_import":
            mod = "io_scene_gltf2"
        elif op == "wm.usd_import":
            mod = "io_scene_usd"
        if not mod:
            return False
        # If already enabled, nothing to do
        if mod in bpy.context.preferences.addons:
            return True
        try:
            bpy.ops.preferences.addon_enable(module=mod)
            print(f"[SpeechToBlender] Auto-enabled add-on: {mod} for {op_fullname}")
            return True
        except Exception as e:
            print(f"[SpeechToBlender] Failed to enable add-on {mod}: {e}")
            return False
    except Exception:
        return False


def _safe_call_operator(op_fullname: str, kwargs: dict):
    """Safely call a Blender operator. Returns (success, message)."""
    ok, reason = _is_safe_op(op_fullname)
    if not ok:
        print(f"[SpeechToBlender] Safety blocked: {op_fullname} - {reason}")
        return False, reason
    try:
        cat, name = op_fullname.split(".", 1)
        cat_obj = getattr(bpy.ops, cat)
        fn = getattr(cat_obj, name)
        
        # Validate filepath for import operators
        if "import" in op_fullname.lower() and "filepath" in kwargs:
            filepath = kwargs.get("filepath", "")
            if not filepath:
                return False, "Missing filepath in kwargs"
            if not os.path.isfile(filepath):
                return False, f"File not found: {filepath}"
            # Normalize path
            kwargs["filepath"] = os.path.normpath(os.path.abspath(filepath))
            print(f"[SpeechToBlender] Importing file: {kwargs['filepath']}")
        
        result = fn(**(kwargs or {}))
        if result and isinstance(result, set):
            if 'CANCELLED' in result:
                return False, f"Operator cancelled: {op_fullname}"
            elif 'FINISHED' in result:
                return True, "OK"
        return True, "OK"
    except AttributeError as e:
        # Try auto-enable required add-on once, then retry
        if _ensure_import_addon_for_operator(op_fullname):
            try:
                cat, name = op_fullname.split(".", 1)
                cat_obj = getattr(bpy.ops, cat)
                fn = getattr(cat_obj, name)
                result = fn(**(kwargs or {}))
                if result and isinstance(result, set) and 'CANCELLED' in result:
                    return False, f"Operator cancelled: {op_fullname}"
                return True, "OK"
            except Exception as e2:
                error_msg = f"Operator not found after enabling add-on: {op_fullname} ({e2})"
                print(f"[SpeechToBlender] {error_msg}")
                return False, error_msg
        error_msg = f"Operator not found: {op_fullname} ({e})"
        print(f"[SpeechToBlender] {error_msg}")
        return False, error_msg
    except TypeError as e:
        error_msg = f"Bad arguments for {op_fullname}: {e}"
        print(f"[SpeechToBlender] {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg
    except Exception as e:
        error_msg = f"Operator error {op_fullname}: {e}"
        print(f"[SpeechToBlender] {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg


# ───────── RPC Server Thread ─────────
class _RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


def _rpc_execute(cmd_dict):
    """XML-RPC method: accepts a dict and routes through safety + executor."""
    try:
        # Support both old format (cmd_dict) and new format (from command_exec)
        if isinstance(cmd_dict, dict):
            cmd_type = cmd_dict.get("type")
            if cmd_type:
                # New format: use command_exec
                try:
                    from .addon.command_exec import execute_command
                    from .stb_core.config import load_config
                    cfg = load_config()
                    return execute_command(cmd_dict, cfg)
                except Exception as e:
                    return {"ok": False, "error": f"Command exec failed: {e}"}
            else:
                # Old format: direct operator call
                op = cmd_dict.get("op")
                kwargs = cmd_dict.get("kwargs", {})
                
                # Special case: execute Python code
                if op == "execute" and "code" in kwargs:
                    return _execute_python_code(kwargs.get("code", ""))
                
                if op:
                    ok, msg = _safe_call_operator(op, kwargs)
                    return {"ok": ok, "message": msg}
        return {"ok": False, "error": "Invalid command format"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _execute_python_code(code: str):
    """Queue Python code execution on main thread. Returns immediately."""
    if not code or not isinstance(code, str):
        return {"ok": False, "error": "No code provided"}
    
    # Queue execution on main thread (like operators)
    _TASKQ.put(("EXEC_PYTHON", code))
    return {"ok": True, "message": "Code queued for execution"}


def _server_loop():
    """Run the XML-RPC server in a background thread."""
    global _SERVER, _SERVER_RUNNING
    try:
        with SimpleXMLRPCServer(
            (HOST, PORT),
            requestHandler=_RequestHandler,
            allow_none=True,
            logRequests=False,
        ) as server:
            _SERVER = server
            
            # Register introspection
            server.register_introspection_functions()
            server.register_multicall_functions()
            
            # Focus Stack data (shared state for reference resolution)
            _FOCUS_STACK_DATA = {
                "last_created": None,
                "last_modified": None,
                "last_mentioned": None,
                "command_history": [],
            }
            
            # Register RPC methods
            def ping():
                return "pong"
            
            def enqueue_op_safe(op_name="wm.redraw_timer", kwargs=None):
                kwargs = kwargs or {}
                print(f"[SpeechToBlender] RPC enqueue: {op_name} {kwargs}")
                _TASKQ.put(("OP_SAFE", op_name, kwargs))
                return "enqueued"
            
            def get_ai_model_provider():
                """RPC method: Get selected AI model provider."""
                try:
                    if ADDON_ROOT in bpy.context.preferences.addons:
                        addon_prefs = bpy.context.preferences.addons[ADDON_ROOT].preferences
                        provider = getattr(addon_prefs, "ai_model_provider", "openai-gpt-5") or "openai-gpt-5"
                        return provider
                except Exception as e:
                    print(f"[SpeechToBlender] Error getting AI model provider: {e}")
                return "openai-gpt-5"  # Default fallback
            
            def get_openai_api_key():
                """RPC method: Get OpenAI API key from preferences (combines parts if split)."""
                try:
                    # Access preferences directly (works in background threads)
                    if ADDON_ROOT in bpy.context.preferences.addons:
                        addon_prefs = bpy.context.preferences.addons[ADDON_ROOT].preferences
                        
                        # Try split fields first (new method)
                        part1 = getattr(addon_prefs, "openai_api_key_part1", "") or ""
                        part2 = getattr(addon_prefs, "openai_api_key_part2", "") or ""
                        
                        if part1 or part2:
                            # Combine the parts
                            key = (part1 + part2).strip()
                            if key:
                                print(f"[SpeechToBlender] Retrieved API key from split fields: part1={len(part1)}, part2={len(part2)}, total={len(key)} chars")
                        else:
                            # Fall back to legacy single field
                            key = getattr(addon_prefs, "openai_api_key", "") or ""
                            if key:
                                print(f"[SpeechToBlender] Retrieved API key from legacy field: {len(key)} chars")
                        
                        if key:
                            original_len = len(key)
                            print(f"[SpeechToBlender] Retrieved API key from preferences: {original_len} chars")
                            
                            # Only remove actual whitespace (spaces, newlines, tabs) - don't remove all whitespace
                            # This preserves the key structure
                            key = key.strip()  # Only leading/trailing
                            # Remove any embedded newlines or tabs that might have been pasted
                            key = key.replace('\n', '').replace('\r', '').replace('\t', ' ')
            
                            # Remove any double spaces
                            while '  ' in key:
                                key = key.replace('  ', ' ')
                            # For API keys, there should be no spaces - remove any that exist
                            key = key.replace(' ', '')
                            
                            final_len = len(key)
                            if original_len != final_len:
                                print(f"[SpeechToBlender] Cleaned key: {original_len} -> {final_len} chars (removed whitespace)")
                            
                            # Check for hidden characters (non-printable)
                            non_printable = [c for c in key if not c.isprintable()]
                            if non_printable:
                                print(f"[SpeechToBlender] ⚠️ WARNING: Key contains {len(non_printable)} non-printable characters!")
                                # Remove non-printable characters
                                key = ''.join(c for c in key if c.isprintable())
                                print(f"[SpeechToBlender] Cleaned key length after removing non-printable: {len(key)}")
                            
                            # Show raw representation for debugging
                            print(f"[SpeechToBlender] Final key length: {len(key)} chars")
                            if len(key) > 60:
                                print(f"[SpeechToBlender] Key preview (first 30): {repr(key[:30])}")
                                print(f"[SpeechToBlender] Key preview (last 30): {repr(key[-30:])}")
                                # Warn if key seems shorter than expected for project keys
                                if len(key) < 150:
                                    print(f"[SpeechToBlender] ⚠️ Key length ({len(key)} chars) is shorter than expected for project keys (~150-170 chars)")
                                    print(f"[SpeechToBlender] This suggests the key may have been truncated when stored in Blender preferences.")
                                    print(f"[SpeechToBlender] Please try clearing and re-pasting the full key.")
                            
                            if key:
                                # Basic validation: OpenAI keys start with "sk-" or "sk-proj-"
                                if key.startswith("sk-") or key.startswith("sk-proj-"):
                                    return key
                                else:
                                    print(f"[SpeechToBlender] ⚠️ OpenAI API key format looks invalid (should start with 'sk-' or 'sk-proj-')")
                                    return key  # Still return it, let OpenAI API validate
                    else:
                        print(f"[SpeechToBlender] ⚠️ Add-on '{ADDON_ROOT}' not found in preferences")
                except Exception as e:
                    print(f"[SpeechToBlender] Error getting OpenAI API key: {e}")
                    import traceback
                    traceback.print_exc()
                return ""
            
            def get_gemini_api_key():
                """RPC method: Get Google Gemini API key from preferences."""
                try:
                    if ADDON_ROOT in bpy.context.preferences.addons:
                        addon_prefs = bpy.context.preferences.addons[ADDON_ROOT].preferences
                        key = getattr(addon_prefs, "gemini_api_key", "") or ""
                        if key:
                            key = key.strip()
                            print(f"[SpeechToBlender] Retrieved Gemini API key: {len(key)} chars")
                        return key
                except Exception as e:
                    print(f"[SpeechToBlender] Error getting Gemini API key: {e}")
                return ""
            
            def start_voice_command():
                """RPC method: Mark the start of a new voice command (pushes undo point)."""
                global _NEED_UNDO_PUSH
                _NEED_UNDO_PUSH = True
                return "OK"
            
            def get_super_mode_state():
                """RPC method: Return modeling context state (always enabled)."""
                try:
                    target_object = ""
                    use_react = False
                    enable_quality = False
                    if ADDON_ROOT in bpy.context.preferences.addons:
                        addon_prefs = bpy.context.preferences.addons[ADDON_ROOT].preferences
                        target_object = getattr(addon_prefs, "super_mode_target_object", "")
                        use_react = getattr(addon_prefs, "use_react_reasoning", False)
                        enable_quality = getattr(addon_prefs, "enable_quality_assessment", False)
                    
                    return {
                        "enabled": True,
                        "target_object": target_object,
                        "use_react": use_react,
                        "enable_quality_assessment": enable_quality,
                    }
                except Exception as e:
                    print(f"[SpeechToBlender] Error getting modeling state: {e}")
                    return {"enabled": True, "target_object": "", "use_react": False, "enable_quality_assessment": False}
            
            def get_modeling_context():
                """RPC method: Get current modeling context (scene state, selected objects, mode, modifiers)."""
                try:
                    context = bpy.context
                    scene = context.scene
                    
                    # Get selected objects (use view_layer for proper access)
                    selected_objects = []
                    for obj in context.view_layer.objects.selected:
                        obj_info = {
                            "name": obj.name,
                            "type": obj.type,
                            "location": tuple(obj.location),
                            "rotation": tuple(obj.rotation_euler),
                            "scale": tuple(obj.scale),
                        }
                        selected_objects.append(obj_info)
                    
                    # Get active object (use view_layer for proper access)
                    active_obj = None
                    if context.view_layer.objects.active:
                        active = context.view_layer.objects.active
                        active_obj = {
                            "name": active.name,
                            "type": active.type,
                            "mode": active.mode if hasattr(active, "mode") else "OBJECT",
                        }
                    
                    # Get current mode
                    current_mode = "OBJECT"
                    if context.view_layer.objects.active:
                        active = context.view_layer.objects.active
                        if hasattr(active, "mode"):
                            current_mode = active.mode
                    
                    # Get modifiers on active object
                    modifiers = []
                    if context.view_layer.objects.active:
                        active = context.view_layer.objects.active
                        if hasattr(active, "modifiers"):
                            for mod in active.modifiers:
                                mod_info = {
                                    "name": mod.name,
                                    "type": mod.type,
                                }
                                modifiers.append(mod_info)
                    
                    # Get scene info
                    scene_info = {
                        "object_count": len(scene.objects),
                        "collection_count": len(bpy.data.collections),
                        "material_count": len(bpy.data.materials),
                    }
                    
                    return {
                        "selected_objects": selected_objects,
                        "active_object": active_obj,
                        "current_mode": current_mode,
                        "modifiers": modifiers,
                        "scene_info": scene_info,
                    }
                except Exception as e:
                    print(f"[SpeechToBlender] Error getting modeling context: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"error": str(e)}
            
            def analyze_current_mesh():
                """RPC method: Analyze current mesh geometry (vertex count, edge loops, face topology)."""
                try:
                    context = bpy.context
                    obj = context.view_layer.objects.active if context.view_layer.objects.active else None
                    if not obj or obj.type != 'MESH':
                        return {"error": "No active mesh object"}
                    
                    mesh = obj.data
                    
                    # Basic mesh stats
                    vertex_count = len(mesh.vertices)
                    edge_count = len(mesh.edges)
                    face_count = len(mesh.polygons)
                    
                    # FINE DETAILS: ALL vertices (not just sample)
                    all_vertices = []
                    for i, v in enumerate(mesh.vertices):
                        all_vertices.append({
                            "index": i,
                            "co": tuple(v.co),
                            "normal": tuple(v.normal) if hasattr(v, 'normal') else None,
                            "select": v.select if hasattr(v, 'select') else False,
                        })
                    
                    # FINE DETAILS: All edges with connectivity
                    all_edges = []
                    for i, edge in enumerate(mesh.edges):
                        all_edges.append({
                            "index": i,
                            "vertices": tuple(edge.vertices),
                            "select": edge.select if hasattr(edge, 'select') else False,
                        })
                    
                    # FINE DETAILS: All faces with vertex indices
                    all_faces = []
                    for i, face in enumerate(mesh.polygons):
                        all_faces.append({
                            "index": i,
                            "vertices": tuple(face.vertices),
                            "edges": tuple(face.edge_keys) if hasattr(face, 'edge_keys') else None,
                            "normal": tuple(face.normal),
                            "center": tuple(face.center),
                            "area": face.area,
                            "select": face.select if hasattr(face, 'select') else False,
                        })
                    
                    # Edge loop detection (enhanced)
                    edge_loops = []
                    if edge_count > 0:
                        # Build edge connectivity graph
                        vertex_edge_count = {}
                        vertex_edges = {}  # Map vertex -> list of connected edges
                        for edge in mesh.edges:
                            for v_idx in edge.vertices:
                                vertex_edge_count[v_idx] = vertex_edge_count.get(v_idx, 0) + 1
                                if v_idx not in vertex_edges:
                                    vertex_edges[v_idx] = []
                                vertex_edges[v_idx].append(edge.index)
                        
                        # Find actual loops (vertices with exactly 2 edges)
                        loop_vertices = [v_idx for v_idx, count in vertex_edge_count.items() if count == 2]
                        if loop_vertices:
                            edge_loops.append({
                                "type": "potential_loop",
                                "vertex_count": len(loop_vertices),
                                "vertices": loop_vertices[:50],  # Limit to first 50 for size
                            })
                    
                    # Face topology analysis (detailed)
                    # XML-RPC requires dictionary keys to be strings
                    face_types = {}
                    n_gons = []  # Track problematic n-gons
                    for face in mesh.polygons:
                        vert_count = len(face.vertices)
                        key = str(vert_count)  # Convert to string for XML-RPC compatibility
                        face_types[key] = face_types.get(key, 0) + 1
                        if vert_count > 4:
                            n_gons.append({
                                "index": face.index,
                                "vertex_count": vert_count,
                            })
                    
                    # Mesh bounds with enhanced geometry hints - USE WORLD-SPACE to account for transforms
                    if mesh.vertices:
                        # Get world-space coordinates (accounts for all transforms: scale, rotation, translation)
                        import mathutils
                        matrix_world = obj.matrix_world
                        coords = [matrix_world @ v.co for v in mesh.vertices]
                        min_co = tuple(min(c[i] for c in coords) for i in range(3))
                        max_co = tuple(max(c[i] for c in coords) for i in range(3))
                        size = tuple(max_co[i] - min_co[i] for i in range(3))
                        
                        # Enhanced geometry hints
                        # Sort dimensions: width (x), depth (y), height (z) - typically largest to smallest
                        sorted_dims = sorted(size, reverse=True)
                        width, depth, height = size[0], size[1], size[2]
                        
                        # Average thickness (minimum dimension)
                        avg_thickness = min(size)
                        
                        # Aspect ratios
                        aspect_ratio_xy = width / depth if depth > 0.001 else 0
                        aspect_ratio_xz = width / height if height > 0.001 else 0
                        aspect_ratio_yz = depth / height if height > 0.001 else 0
                        
                        # Classification: flat vs tall
                        # Flat: height is significantly smaller than width/depth
                        # Tall: height is similar to or larger than width/depth
                        is_flat = height < (width * 0.3) and height < (depth * 0.3)
                        is_tall = height > (width * 0.7) or height > (depth * 0.7)
                        shape_class = "flat" if is_flat else ("tall" if is_tall else "balanced")
                        
                        bounds = {
                            "min": min_co,
                            "max": max_co,
                            "size": size,
                            "width": width,
                            "depth": depth,
                            "height": height,
                            "avg_thickness": avg_thickness,
                            "aspect_ratios": {
                                "width_depth": aspect_ratio_xy,
                                "width_height": aspect_ratio_xz,
                                "depth_height": aspect_ratio_yz,
                            },
                            "shape_class": shape_class,
                            "sorted_dimensions": sorted_dims,  # Largest to smallest
                        }
                    else:
                        bounds = None
                    
                    # Selection state
                    selected_vertices = []
                    selected_edges = []
                    selected_faces = []
                    
                    if obj.mode == 'EDIT':
                        # In edit mode, get selected elements
                        # Need to use bmesh from edit mesh
                        bm = bmesh.from_edit_mesh(mesh)
                        bm.verts.ensure_lookup_table()
                        bm.edges.ensure_lookup_table()
                        bm.faces.ensure_lookup_table()
                        
                        for v in bm.verts:
                            if v.select:
                                selected_vertices.append(v.index)
                        for e in bm.edges:
                            if e.select:
                                selected_edges.append(e.index)
                        for f in bm.faces:
                            if f.select:
                                selected_faces.append(f.index)
                        
                        # Don't free bmesh from edit mesh - it's managed by Blender
                    
                    return {
                        "vertex_count": vertex_count,
                        "edge_count": edge_count,
                        "face_count": face_count,
                        "all_vertices": all_vertices,  # ALL vertices, not sample
                        "all_edges": all_edges,  # ALL edges
                        "all_faces": all_faces,  # ALL faces
                        "edge_loops": edge_loops,
                        "face_topology": {
                            "face_types": face_types,  # {3: count, 4: count, ...} for triangles, quads, etc.
                            "n_gons": n_gons[:20],  # First 20 n-gons
                        },
                        "bounds": bounds,
                        "selection": {
                            "vertices": selected_vertices,
                            "edges": selected_edges,
                            "faces": selected_faces,
                        },
                        "object_name": obj.name,
                        "object_mode": obj.mode,
                    }
                except Exception as e:
                    print(f"[SpeechToBlender] Error analyzing mesh: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"error": str(e)}
            
            def capture_viewport_screenshot():
                """RPC method: Capture current Blender viewport as base64-encoded PNG image.
                Uses screen capture to get the Blender window directly."""
                try:
                    import base64
                    import tempfile
                    import os
                    import sys
                    
                    print("[SpeechToBlender] 📸 Starting viewport screenshot capture (screen capture method)...")
                    
                    # Ensure viewport is in Material Preview mode so materials are visible
                    try:
                        context = bpy.context
                        viewport_area = None
                        for area in context.screen.areas:
                            if area.type == 'VIEW_3D':
                                viewport_area = area
                                break
                        
                        if viewport_area:
                            # Set viewport shading to Material Preview
                            for space in viewport_area.spaces:
                                if space.type == 'VIEW_3D':
                                    if hasattr(space, 'shading'):
                                        old_mode = space.shading.type
                                        space.shading.type = 'MATERIAL'
                                        print(f"[SpeechToBlender] 📸 Set viewport shading to Material Preview (was: {old_mode})")
                                    break
                    except Exception as e:
                        print(f"[SpeechToBlender] ⚠️ Could not set viewport shading mode: {e}")
                    
                    screenshot_data = None
                    
                    # Method 1: Try Blender render.opengl (captures viewport only, preferred)
                    try:
                        context = bpy.context
                        
                        # Find a 3D viewport area
                        viewport_area = None
                        for area in context.screen.areas:
                            if area.type == 'VIEW_3D':
                                viewport_area = area
                                break
                        
                        if viewport_area:
                            # Override context to use the 3D viewport
                            override = context.copy()
                            override['area'] = viewport_area
                            override['region'] = viewport_area.regions[-1]  # Main region
                            
                            try:
                                # Use OpenGL render with context override
                                bpy.ops.render.opengl(override, write_still=False)
                                print("[SpeechToBlender] ✅ Used render.opengl with context override")
                            except Exception as e:
                                print(f"[SpeechToBlender] ⚠️ render.opengl failed: {e}, trying render.render()...")
                                # Fall back to regular render
                                bpy.ops.render.render(override, write_still=False)
                        else:
                            # No 3D viewport found, use regular render
                            print("[SpeechToBlender] ⚠️ No 3D viewport found, using render.render()...")
                            bpy.ops.render.render(write_still=False)
                        
                        # Get the rendered image from Render Result
                        render_result = bpy.data.images.get('Render Result')
                        if not render_result:
                            raise Exception("Could not get Render Result image")
                        
                        print(f"[SpeechToBlender] ✅ Got Render Result: {render_result.size[0]}x{render_result.size[1]}")
                        
                        # Save to temporary file
                        temp_path = tempfile.mktemp(suffix='.png')
                        render_result.save_render(temp_path)
                        print(f"[SpeechToBlender] 💾 Saved render to temp file: {temp_path}")
                        
                        # Read and encode as base64
                        with open(temp_path, 'rb') as f:
                            image_data = f.read()
                            screenshot_data = base64.b64encode(image_data).decode('utf-8')
                        
                        print(f"[SpeechToBlender] ✅ Encoded to base64: {len(image_data)} bytes raw, {len(screenshot_data)} chars base64")
                        
                        # Cleanup temp file
                        try:
                            os.remove(temp_path)
                            print(f"[SpeechToBlender] 🗑️ Cleaned up temp file")
                        except Exception:
                            pass
                        
                        return {"image_base64": screenshot_data, "format": "png"}
                    except Exception as e:
                        print(f"[SpeechToBlender] ⚠️ Blender render method failed: {e}, falling back to screen capture...")
                        import traceback
                        traceback.print_exc()
                    
                    # Method 2: Fallback to screen capture (captures entire screen)
                    # Try using PIL/Pillow with screen capture (if available)
                    try:
                        from PIL import ImageGrab
                        print("[SpeechToBlender] 📸 Attempting PIL.ImageGrab screen capture (entire screen)...")
                        # Capture entire screen
                        screenshot = ImageGrab.grab()
                        temp_path = tempfile.mktemp(suffix='.png')
                        screenshot.save(temp_path, 'PNG')
                        
                        with open(temp_path, 'rb') as f:
                            image_data = f.read()
                            screenshot_data = base64.b64encode(image_data).decode('utf-8')
                        
                        os.remove(temp_path)
                        print(f"[SpeechToBlender] ⚠️ Captured entire screen using PIL.ImageGrab: {len(image_data)} bytes raw, {len(screenshot_data)} chars base64")
                        return {"image_base64": screenshot_data, "format": "png"}
                    except ImportError as e:
                        print(f"[SpeechToBlender] ⚠️ PIL/Pillow not available: {e}, trying mss...")
                    except Exception as e:
                        print(f"[SpeechToBlender] ⚠️ PIL.ImageGrab failed: {e}, trying mss...")
                        import traceback
                        traceback.print_exc()
                    
                    # Method 3: Try using mss (if available) - faster and more reliable
                    try:
                        import mss
                        print("[SpeechToBlender] 📸 Attempting mss screen capture (entire screen)...")
                        with mss.mss() as sct:
                            # Capture entire screen
                            screenshot = sct.grab(sct.monitors[0])  # Primary monitor
                            temp_path = tempfile.mktemp(suffix='.png')
                            mss.tools.to_png(screenshot.rgb, screenshot.size, output=temp_path)
                            
                            with open(temp_path, 'rb') as f:
                                image_data = f.read()
                                screenshot_data = base64.b64encode(image_data).decode('utf-8')
                            
                            os.remove(temp_path)
                            print(f"[SpeechToBlender] ⚠️ Captured entire screen using mss: {len(image_data)} bytes raw, {len(screenshot_data)} chars base64")
                            return {"image_base64": screenshot_data, "format": "png"}
                    except ImportError as e:
                        print(f"[SpeechToBlender] ⚠️ mss not available: {e}")
                    except Exception as e:
                        print(f"[SpeechToBlender] ⚠️ mss failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # All methods failed
                    return {"error": f"All screenshot methods failed. Install Pillow (pip install Pillow) or mss (pip install mss) for screen capture."}
                    
                except Exception as e:
                    print(f"[SpeechToBlender] ❌ Error capturing screenshot: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"error": str(e)}
            
            def rename_active_object(new_name: str):
                """RPC method: Rename the currently active object."""
                try:
                    context = bpy.context
                    obj = context.view_layer.objects.active
                    if not obj:
                        return {"ok": False, "error": "No active object"}
                    
                    old_name = obj.name
                    obj.name = new_name
                    print(f"[SpeechToBlender] Renamed object: {old_name} -> {new_name}")
                    return {"ok": True, "old_name": old_name, "new_name": new_name}
                except Exception as e:
                    print(f"[SpeechToBlender] Error renaming object: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"ok": False, "error": str(e)}
            
            def assess_object_quality(target_object: str = ""):
                """RPC method: Assess the quality and completeness of the current object/scene relative to a target object.
                Returns detailed assessment including what's missing, what needs refinement, and suggestions."""
                try:
                    context = bpy.context
                    scene = context.scene
                    active_obj = context.view_layer.objects.active
                    
                    # Get all mesh objects and their names
                    mesh_objects = [o for o in scene.objects if o.type == 'MESH']
                    object_names = [obj.name.lower() for obj in mesh_objects]
                    
                    assessment = {
                        "target_object": target_object,
                        "timestamp": time.time(),
                        "scene_summary": {
                            "total_objects": len(scene.objects),
                            "mesh_objects": len(mesh_objects),
                            "object_names": [obj.name for obj in mesh_objects],
                        },
                        "active_object": None,
                        "quality_score": 0.0,
                        "target_match_score": 0.0,  # How well it matches the target object
                        "completeness": {
                            "basic_structure": False,
                            "details": False,
                            "proportions": False,
                            "refinement": False,
                            "target_features": False,
                        },
                        "issues": [],
                        "suggestions": [],
                        "missing_features": [],
                        "found_features": [],
                    }
                    
                    # TARGET-SPECIFIC CHECKS
                    target_lower = target_object.lower() if target_object else ""
                    
                    # iPhone-specific checks
                    if "iphone" in target_lower or "phone" in target_lower:
                        required_parts = {
                            "body": ["body", "phone body", "iphone body"],
                            "screen": ["screen", "display", "phone screen", "iphone screen"],
                            "camera": ["camera", "camera module", "lens"],
                        }
                        optional_parts = {
                            "buttons": ["button", "volume", "power", "home"],
                            "speaker": ["speaker", "audio"],
                            "port": ["port", "charging", "lightning"],
                        }
                        
                        found_parts = {}
                        for part_type, keywords in required_parts.items():
                            found = any(any(kw in name for kw in keywords) for name in object_names)
                            found_parts[part_type] = found
                            if found:
                                assessment["found_features"].append(f"{part_type.title()} found")
                            else:
                                assessment["missing_features"].append(f"Missing {part_type}")
                                assessment["issues"].append(f"Missing required part: {part_type}")
                        
                        for part_type, keywords in optional_parts.items():
                            found = any(any(kw in name for kw in keywords) for name in object_names)
                            if found:
                                assessment["found_features"].append(f"{part_type.title()} found")
                        
                        # Calculate target match score for iPhone
                        required_count = sum(1 for found in found_parts.values() if found)
                        total_required = len(required_parts)
                        assessment["target_match_score"] = required_count / total_required if total_required > 0 else 0.0
                        
                        if assessment["target_match_score"] < 0.5:
                            assessment["issues"].append(f"Object does not look like {target_object} - missing {total_required - required_count} required parts")
                            assessment["suggestions"].append(f"Add missing parts: {', '.join([k for k, v in found_parts.items() if not v])}")
                    
                    # Car-specific checks
                    elif "car" in target_lower or "vehicle" in target_lower:
                        required_parts = {
                            "body": ["body", "car body", "chassis"],
                            "wheels": ["wheel", "tire"],
                        }
                        optional_parts = {
                            "windows": ["window", "windshield", "glass"],
                            "lights": ["light", "headlight", "taillight"],
                            "spoiler": ["spoiler"],
                        }
                        
                        found_parts = {}
                        for part_type, keywords in required_parts.items():
                            found = any(any(kw in name for kw in keywords) for name in object_names)
                            found_parts[part_type] = found
                            if found:
                                assessment["found_features"].append(f"{part_type.title()} found")
                            else:
                                assessment["missing_features"].append(f"Missing {part_type}")
                        
                        # Check wheel count
                        wheel_count = sum(1 for name in object_names if any(kw in name for kw in ["wheel", "tire"]))
                        if wheel_count < 4:
                            assessment["issues"].append(f"Car should have 4 wheels, found {wheel_count}")
                            assessment["missing_features"].append(f"Need {4 - wheel_count} more wheels")
                        
                        required_count = sum(1 for found in found_parts.values() if found)
                        total_required = len(required_parts)
                        assessment["target_match_score"] = required_count / total_required if total_required > 0 else 0.0
                    
                    # Cup/Mug-specific checks
                    elif "cup" in target_lower or "mug" in target_lower:
                        required_parts = {
                            "body": ["body", "cup body", "mug body"],
                            "handle": ["handle"],
                        }
                        
                        found_parts = {}
                        for part_type, keywords in required_parts.items():
                            found = any(any(kw in name for kw in keywords) for name in object_names)
                            found_parts[part_type] = found
                            if found:
                                assessment["found_features"].append(f"{part_type.title()} found")
                            else:
                                assessment["missing_features"].append(f"Missing {part_type}")
                        
                        required_count = sum(1 for found in found_parts.values() if found)
                        total_required = len(required_parts)
                        assessment["target_match_score"] = required_count / total_required if total_required > 0 else 0.0
                    
                    # Generic object check (if no specific match)
                    else:
                        # Just check if we have objects
                        if len(mesh_objects) == 0:
                            assessment["target_match_score"] = 0.0
                            assessment["issues"].append("No objects created yet")
                        elif len(mesh_objects) == 1:
                            assessment["target_match_score"] = 0.3
                            assessment["suggestions"].append("Consider if the target object requires multiple parts/components")
                        else:
                            assessment["target_match_score"] = 0.5  # Some objects exist
                    
                    # Technical quality checks
                    if active_obj and active_obj.type == 'MESH':
                        mesh = active_obj.data
                        assessment["active_object"] = {
                            "name": active_obj.name,
                            "vertex_count": len(mesh.vertices),
                            "face_count": len(mesh.polygons),
                            "has_modifiers": len(active_obj.modifiers) > 0,
                            "modifier_count": len(active_obj.modifiers),
                        }
                        
                        # Check if object is just a basic primitive
                        if len(mesh.vertices) <= 8 and len(mesh.polygons) <= 6:
                            assessment["issues"].append("Object appears to be a basic primitive - needs refinement")
                            assessment["suggestions"].append("Use edit mode operations (extrude, inset, loop cut) or modifiers (subdivision surface, bevel) to add detail")
                        
                        # Check for detail (workflow-agnostic - modifiers OR high vertex count)
                        has_active_modifiers = len(active_obj.modifiers) > 0
                        has_high_detail = len(mesh.vertices) > 100 or len(mesh.polygons) > 200
                        
                        if not has_active_modifiers and not has_high_detail:
                            assessment["issues"].append("Object may lack detail - consider adding geometry or modifiers")
                            assessment["suggestions"].append("Add detail via edit mode operations, modifiers, or subdivision")
                        
                        # Check complexity (workflow-agnostic)
                        if len(mesh.vertices) < 100:
                            assessment["issues"].append("Low vertex count - may need more geometry for detailed objects")
                            assessment["suggestions"].append("Use Subdivision Surface modifier or edit mode operations to increase detail")
                    
                    # Scene-level assessment
                    if len(mesh_objects) == 0:
                        assessment["issues"].append("No mesh objects in scene")
                    
                    # Calculate overall quality score (workflow-agnostic - focuses on actual detail, not workflow)
                    technical_score = 0.0
                    if assessment["active_object"]:
                        obj_info = assessment["active_object"]
                        # Points for actual geometry detail (not workflow-specific)
                        if obj_info["vertex_count"] > 500:
                            technical_score += 0.3
                        elif obj_info["vertex_count"] > 100:
                            technical_score += 0.2
                        elif obj_info["vertex_count"] > 20:
                            technical_score += 0.1
                        
                        # Points for detail (modifiers OR high vertex count - both are valid)
                        if obj_info["has_modifiers"] or obj_info["vertex_count"] > 100:
                            technical_score += 0.2
                        
                        # Points for naming
                        if obj_info["name"] and not obj_info["name"].startswith(("Cube", "Cylinder", "Sphere", "Plane")):
                            technical_score += 0.1
                    
                    # Combine technical (40%) and target match (60%)
                    assessment["quality_score"] = (technical_score * 0.4) + (assessment["target_match_score"] * 0.6)
                    
                    # Final verdict
                    if assessment["quality_score"] < 0.3:
                        assessment["issues"].append("CRITICAL: Object does not match target and lacks detail")
                    elif assessment["quality_score"] < 0.6:
                        assessment["issues"].append("Object partially matches target but needs more work")
                    
                    return assessment
                except Exception as e:
                    print(f"[SpeechToBlender] Error assessing object quality: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"error": str(e)}
            
            def analyze_scene():
                """RPC method: Analyze entire scene - high-level overview of all objects and their relationships."""
                try:
                    context = bpy.context
                    scene = context.scene
                    
                    # HIGH-LEVEL: Scene summary stats
                    scene_summary = {
                        "total_objects": len(scene.objects),
                        "mesh_objects": 0,
                        "light_objects": 0,
                        "camera_objects": 0,
                        "collection_count": len(bpy.data.collections),
                        "material_count": len(bpy.data.materials),
                    }
                    
                    # HIGH-LEVEL: Object groups by type
                    objects_by_type = {}
                    scene_objects = []  # Initialize list to store all object info
                    
                    # Analyze all objects in scene (keep lightweight for scene view)
                    for obj in scene.objects:
                        obj_type = obj.type
                        scene_summary[f"{obj_type.lower()}_objects"] = scene_summary.get(f"{obj_type.lower()}_objects", 0) + 1
                        
                        if obj_type not in objects_by_type:
                            objects_by_type[obj_type] = []
                        
                        obj_info = {
                            "name": obj.name,
                            "type": obj_type,
                            "location": tuple(obj.location),
                            "scale": tuple(obj.scale),
                        }
                        
                        # For mesh objects, get summary info (not full detail)
                        if obj.type == 'MESH':
                            mesh = obj.data
                            vertex_count = len(mesh.vertices)
                            face_count = len(mesh.polygons)
                            
                            # Quick bounds calculation - USE WORLD-SPACE to account for transforms
                            if mesh.vertices:
                                # Get world-space coordinates (accounts for all transforms)
                                import mathutils
                                matrix_world = obj.matrix_world
                                coords = [matrix_world @ v.co for v in mesh.vertices]
                                min_co = tuple(min(c[i] for c in coords) for i in range(3))
                                max_co = tuple(max(c[i] for c in coords) for i in range(3))
                                size = tuple(max_co[i] - min_co[i] for i in range(3))
                                volume_estimate = size[0] * size[1] * size[2]
                                thickness = min(size)
                            else:
                                size = (0, 0, 0)
                                volume_estimate = 0
                                thickness = 0
                            
                            obj_info["mesh_summary"] = {
                                "vertex_count": vertex_count,
                                "face_count": face_count,
                                "size": size,
                                "volume_estimate": volume_estimate,
                                "thickness": thickness,
                            }
                            
                            # Materials (summary)
                            materials = []
                            if obj.data.materials:
                                for mat in obj.data.materials:
                                    if mat:
                                        materials.append({
                                            "name": mat.name,
                                            "use_nodes": getattr(mat, "use_nodes", False),
                                        })
                            obj_info["materials"] = materials
                            
                            # Modifiers (summary)
                            modifiers = []
                            if hasattr(obj, "modifiers"):
                                for mod in obj.modifiers:
                                    modifiers.append({
                                        "name": mod.name,
                                        "type": mod.type,
                                    })
                            obj_info["modifiers"] = modifiers
                        
                        objects_by_type[obj_type].append(obj_info["name"])
                        scene_objects.append(obj_info)
                    
                    # Categorize objects by part type (based on names and geometry)
                    def categorize_object_parts(objects):
                        """Categorize objects into parts (screen, camera, button, body, etc.)."""
                        categories = {
                            "body": [],
                            "screen": [],
                            "camera": [],
                            "button": [],
                            "bezel": [],
                            "other": [],
                        }
                        
                        for obj in objects:
                            name_lower = obj.get("name", "").lower()
                            obj_type = obj.get("type", "")
                            mesh_info = obj.get("mesh_info", {})
                            
                            # Skip non-mesh objects for part detection
                            if obj_type != "MESH" or not mesh_info:
                                categories["other"].append(obj.get("name"))
                                continue
                            
                            bounds = mesh_info.get("bounds", {})
                            size = bounds.get("size", (0, 0, 0))
                            thickness = bounds.get("thickness", 0)
                            volume = bounds.get("volume_estimate", 0)
                            
                            # Part detection based on name keywords
                            is_screen = any(kw in name_lower for kw in ["screen", "display", "panel", "face"])
                            is_camera = any(kw in name_lower for kw in ["camera", "lens", "sensor"])
                            is_button = any(kw in name_lower for kw in ["button", "btn", "key", "switch"])
                            is_bezel = any(kw in name_lower for kw in ["bezel", "frame", "rim"])
                            is_body = any(kw in name_lower for kw in ["body", "case", "housing", "chassis", "main"])
                            
                            # Part detection based on geometry
                            # Screen: very flat, large area, thin
                            if not is_screen and thickness > 0:
                                aspect_flat = max(size[0], size[1]) / thickness if thickness > 0.001 else 0
                                if aspect_flat > 20 and volume > 0.1:  # Very flat and reasonably large
                                    is_screen = True
                            
                            # Camera: small extruded part (small volume, but not too thin)
                            if not is_camera and volume > 0:
                                if volume < 0.5 and thickness > 0.05:  # Small but not paper-thin
                                    is_camera = True
                            
                            # Button: very small, often circular/cylindrical
                            if not is_button and volume > 0:
                                if volume < 0.1:  # Very small
                                    is_button = True
                            
                            # Material-based detection
                            materials = obj.get("materials", [])
                            for mat in materials:
                                mat_name = mat.get("name", "").lower()
                                if "glass" in mat_name or mat.get("is_glass"):
                                    is_screen = True
                            
                            # Categorize
                            if is_screen:
                                categories["screen"].append({
                                    "name": obj.get("name"),
                                    "size": size,
                                    "thickness": thickness,
                                })
                            elif is_camera:
                                categories["camera"].append({
                                    "name": obj.get("name"),
                                    "size": size,
                                    "volume": volume,
                                })
                            elif is_button:
                                categories["button"].append({
                                    "name": obj.get("name"),
                                    "size": size,
                                    "volume": volume,
                                })
                            elif is_bezel:
                                categories["bezel"].append({
                                    "name": obj.get("name"),
                                    "size": size,
                                })
                            elif is_body:
                                categories["body"].append({
                                    "name": obj.get("name"),
                                    "size": size,
                                    "volume": volume,
                                })
                            else:
                                # Try to infer from size/position
                                # Largest object is likely the body
                                if volume > 1.0:
                                    categories["body"].append({
                                        "name": obj.get("name"),
                                        "size": size,
                                        "volume": volume,
                                        "inferred": True,
                                    })
                                else:
                                    categories["other"].append(obj.get("name"))
                        
                        return categories
                    
                    # HIGH-LEVEL: Spatial relationships
                    spatial_info = {
                        "objects_in_scene": len(scene_objects),
                        "largest_object": None,
                        "smallest_object": None,
                    }
                    
                    if scene_objects:
                        mesh_objs = [o for o in scene_objects if o.get("type") == "MESH" and "mesh_summary" in o]
                        if mesh_objs:
                            # Use a better size metric: for flat objects (volume near 0), use largest dimension
                            # For 3D objects, use volume. This handles screens/planes correctly.
                            def get_size_metric(obj):
                                summary = obj.get("mesh_summary", {})
                                volume = summary.get("volume_estimate", 0)
                                size = summary.get("size", (0, 0, 0))
                                # If volume is very small (flat object), use largest dimension squared as proxy
                                if volume < 0.001:
                                    max_dim = max(size) if size else 0
                                    return max_dim * max_dim  # Surface area proxy for flat objects
                                return volume
                            
                            largest = max(mesh_objs, key=get_size_metric)
                            smallest = min(mesh_objs, key=get_size_metric)
                            spatial_info["largest_object"] = largest.get("name")
                            spatial_info["smallest_object"] = smallest.get("name")
                    
                    categorized_parts = categorize_object_parts(scene_objects)
                    
                    return {
                        "scene_summary": scene_summary,
                        "objects_by_type": objects_by_type,
                        "spatial_info": spatial_info,
                        "object_count": len(scene_objects),
                        "objects": scene_objects,  # Full list but with summary-level detail
                        "categorized_parts": categorized_parts,
                    }
                except Exception as e:
                    print(f"[SpeechToBlender] Error analyzing scene: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"error": str(e)}
            
            def get_scene_state():
                """
                RPC method: Get comprehensive scene state for context-aware command generation.
                Returns object names, positions, dimensions, bounds (world-space), and hierarchy.
                """
                try:
                    from mathutils import Vector
                    scene = bpy.context.scene
                    view_layer = bpy.context.view_layer
                    
                    objects = []
                    for obj in scene.objects:
                        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                        min_corner = Vector((
                            min(c.x for c in bbox_corners),
                            min(c.y for c in bbox_corners),
                            min(c.z for c in bbox_corners)
                        ))
                        max_corner = Vector((
                            max(c.x for c in bbox_corners),
                            max(c.y for c in bbox_corners),
                            max(c.z for c in bbox_corners)
                        ))
                        
                        obj_data = {
                            "name": obj.name,
                            "type": obj.type,
                            "location": list(obj.location),
                            "rotation_euler": list(obj.rotation_euler),
                            "scale": list(obj.scale),
                            "dimensions": list(obj.dimensions),
                            "bounds": {
                                "min": list(min_corner),
                                "max": list(max_corner),
                                "center": list((min_corner + max_corner) / 2)
                            },
                            "parent": obj.parent.name if obj.parent else None,
                            "visible": obj.visible_get(),
                        }
                        
                        if obj.type == 'MESH' and obj.data:
                            obj_data["vertex_count"] = len(obj.data.vertices)
                            obj_data["face_count"] = len(obj.data.polygons)
                        
                        objects.append(obj_data)
                    
                    active_name = view_layer.objects.active.name if view_layer.objects.active else None
                    selected_names = [obj.name for obj in view_layer.objects.selected]
                    
                    return {
                        "objects": objects,
                        "active_object": active_name,
                        "selected_objects": selected_names,
                        "object_count": len(objects),
                        "scene_name": scene.name,
                    }
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {"error": str(e), "objects": []}
            
            
            server.register_function(ping, "ping")
            server.register_function(enqueue_op_safe, "enqueue_op_safe")
            server.register_function(enqueue_op_safe, "enqueue_op")  # Alias
            server.register_function(_rpc_execute, "execute")
            server.register_function(get_ai_model_provider, "get_ai_model_provider")
            server.register_function(get_openai_api_key, "get_openai_api_key")
            server.register_function(get_gemini_api_key, "get_gemini_api_key")
            server.register_function(start_voice_command, "start_voice_command")
            server.register_function(get_super_mode_state, "get_super_mode_state")
            server.register_function(get_modeling_context, "get_modeling_context")
            server.register_function(analyze_current_mesh, "analyze_current_mesh")
            server.register_function(analyze_scene, "analyze_scene")
            server.register_function(get_scene_state, "get_scene_state")
            server.register_function(capture_viewport_screenshot, "capture_viewport_screenshot")
            server.register_function(rename_active_object, "rename_active_object")
            server.register_function(assess_object_quality, "assess_object_quality")
            
            def get_focus_context():
                """RPC method: Get focus stack for reference resolution."""
                context = bpy.context
                
                selected = [obj.name for obj in context.view_layer.objects.selected]
                active = context.view_layer.objects.active.name if context.view_layer.objects.active else None
                
                return {
                    "selected_objects": selected,
                    "active_object": active,
                    "last_created": _FOCUS_STACK_DATA.get("last_created"),
                    "last_modified": _FOCUS_STACK_DATA.get("last_modified"),
                    "last_mentioned": _FOCUS_STACK_DATA.get("last_mentioned"),
                    "recent_commands": _FOCUS_STACK_DATA.get("command_history", [])[-5:],
                }
            
            def update_focus_stack(event_type: str, object_name: str, command_text: str = ""):
                """RPC method: Update focus stack after operations."""
                
                if event_type == "created":
                    _FOCUS_STACK_DATA["last_created"] = object_name
                    _FOCUS_STACK_DATA["last_modified"] = object_name
                elif event_type == "modified":
                    _FOCUS_STACK_DATA["last_modified"] = object_name
                elif event_type == "mentioned":
                    _FOCUS_STACK_DATA["last_mentioned"] = object_name
                
                if command_text:
                    _FOCUS_STACK_DATA["command_history"].append({
                        "command": command_text,
                        "target": object_name,
                        "timestamp": time.time()
                    })
                    _FOCUS_STACK_DATA["command_history"] = _FOCUS_STACK_DATA["command_history"][-20:]
                
                return "OK"
            
            server.register_function(get_focus_context, "get_focus_context")
            server.register_function(update_focus_stack, "update_focus_stack")
            
            def get_voice_listening_state():
                """RPC method: Get voice listening enabled state."""
                global _VOICE_LISTENING_ENABLED
                return {"enabled": _VOICE_LISTENING_ENABLED}
            
            def set_voice_listening_state(enabled):
                """RPC method: Set voice listening enabled state."""
                global _VOICE_LISTENING_ENABLED
                _VOICE_LISTENING_ENABLED = bool(enabled)
                return {"enabled": _VOICE_LISTENING_ENABLED}
            
            server.register_function(get_voice_listening_state, "get_voice_listening_state")
            server.register_function(set_voice_listening_state, "set_voice_listening_state")
            
            _SERVER_RUNNING = True
            print(f"[SpeechToBlender] XML-RPC listening on http://{HOST}:{PORT}/RPC2")
            
            # Main server loop
            while _SERVER_RUNNING:
                server.handle_request()
                
    except OSError as e:
        print(f"[SpeechToBlender] Server bind error on {HOST}:{PORT}: {e}")
    except Exception as e:
        print(f"[SpeechToBlender] Server error: {e}")
    finally:
        _SERVER_RUNNING = False
        print("[SpeechToBlender] Server loop ended")


def _port_in_use(host, port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def _start_server_thread():
    """Start the RPC server in a background thread."""
    global _SERVER_THREAD, _SERVER_RUNNING
    if _SERVER_RUNNING:
        print("[SpeechToBlender] Server already running")
        return True
    if _port_in_use(HOST, PORT):
        print(f"[SpeechToBlender] Port {PORT} already in use; not starting.")
        return False
    _SERVER_THREAD = threading.Thread(target=_server_loop, daemon=True)
    _SERVER_THREAD.start()
    # Wait for server to start
    for _ in range(20):
        if _SERVER_RUNNING:
            break
        time.sleep(0.05)
    return _SERVER_RUNNING


def _stop_server_thread():
    """Stop the RPC server."""
    global _SERVER_RUNNING
    if not _SERVER_RUNNING:
        return
    _SERVER_RUNNING = False
    # Wake up the server to check the flag
    try:
        with socket.create_connection((HOST, PORT), timeout=0.2) as s:
            pass
    except Exception:
        pass
    print("[SpeechToBlender] Requested server stop")
    # Stop voice process when server stops
    _stop_voice_process()


def _drain_task_queue():
    """Drain the task queue on the main thread (called by timer)."""
    global _NEED_UNDO_PUSH
    try:
        # Push undo ONCE at the start if needed (marks start of new voice command)
        if _NEED_UNDO_PUSH:
            try:
                bpy.ops.ed.undo_push(message="Voice Command")
                print("[SpeechToBlender] Pushed undo point for voice command")
            except Exception as e:
                print(f"[SpeechToBlender] Failed to push undo: {e}")
            _NEED_UNDO_PUSH = False
        
        while not _TASKQ.empty():
            msg = _TASKQ.get_nowait()
            kind = msg[0]
            if kind == "OP_SAFE":
                _, name, kwargs = msg
                # Special case: handle execute (Python code execution)
                if name == "execute" and "code" in kwargs:
                    code = kwargs.get("code", "")
                    print(f"[SpeechToBlender] Executing Python code...")
                    try:
                        exec(code, {"__builtins__": __builtins__, "bpy": bpy})
                        print(f"[SpeechToBlender] ✅ Python code executed successfully")
                    except Exception as e:
                        error_msg = str(e)
                        import traceback
                        tb = traceback.format_exc()
                        print(f"[SpeechToBlender] ❌ Python execution error: {error_msg}")
                        print(f"[SpeechToBlender] Traceback: {tb}")
                else:
                    print(f"[SpeechToBlender] Executing: {name} with kwargs: {kwargs}")
                    ok, reason = _safe_call_operator(name, kwargs)
                    if ok:
                        print(f"[SpeechToBlender] ✅ Success: {name}")
                    else:
                        print(f"[SpeechToBlender] ❌ Failed: {name} - {reason}")
                try:
                    bpy.ops.wm.redraw_timer(type="DRAW_WIN", iterations=1)
                except Exception:
                    pass
            elif kind == "EXEC_PYTHON":
                _, code = msg
                print(f"[SpeechToBlender] Executing Python code...")
                try:
                    # Execute Python code in Blender's context
                    exec(code, {"__builtins__": __builtins__, "bpy": bpy})
                    print(f"[SpeechToBlender] ✅ Python code executed successfully")
                except Exception as e:
                    error_msg = str(e)
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[SpeechToBlender] ❌ Python execution error: {error_msg}")
                    print(f"[SpeechToBlender] Traceback: {tb}")
                try:
                    bpy.ops.wm.redraw_timer(type="DRAW_WIN", iterations=1)
                except Exception:
                    pass
            elif kind == "CAPTURE_SCREENSHOT":
                _, capture_func = msg
                print(f"[SpeechToBlender] Executing screenshot capture on main thread...")
                try:
                    capture_func()
                    print(f"[SpeechToBlender] ✅ Screenshot capture completed")
                except Exception as e:
                    error_msg = str(e)
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[SpeechToBlender] ❌ Screenshot capture error: {error_msg}")
                    print(f"[SpeechToBlender] Traceback: {tb}")
    except queue.Empty:
        pass
    except Exception as e:
        print(f"[SpeechToBlender] Error draining task queue: {e}")
        import traceback
        traceback.print_exc()
    return 0.5 if _SERVER_RUNNING else None


# Phase 6: Safety - forbidden substrings in CODE (block before exec)
_LIVE_CODE_BLACKLIST = (
    "import os",
    "import sys",
    "os.",
    "sys.",
    "shutil",
    "sys.exit",
    "quit_blender",
    "quit(",
    "exit(",
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "subprocess",
    "remove(",
)


# Phase 5: Predictive God Mode - keyword -> Blender code (instant primitives)
_LIVE_PREDICT_MAP = {
    "cube": "bpy.ops.mesh.primitive_cube_add()",
    "sphere": "bpy.ops.mesh.primitive_uv_sphere_add()",
    "cylinder": "bpy.ops.mesh.primitive_cylinder_add()",
    "plane": "bpy.ops.mesh.primitive_plane_add()",
    "cone": "bpy.ops.mesh.primitive_cone_add()",
    "torus": "bpy.ops.mesh.primitive_torus_add()",
    "camera": "bpy.ops.object.camera_add()",
    "light": "bpy.ops.object.light_add(type='POINT')",
    "iphone": "bpy.ops.mesh.primitive_cube_add(); bpy.context.object.scale = (0.036, 0.074, 0.004)",
}


def _execute_predict(payload):
    """Execute PREDICT payload (Phase 5: instant primitive)."""
    payload = (payload or "").strip().lower()
    code = _LIVE_PREDICT_MAP.get(payload)
    if not code:
        return
    try:
        exec(code, {"__builtins__": __builtins__, "bpy": bpy})
        print(f"[SpeechToBlender Live] PREDICT executed: {payload}")
    except Exception as e:
        import traceback
        print(f"[SpeechToBlender Live] PREDICT failed ({payload}): {e}")
        traceback.print_exc()


def _drain_live_action_queue():
    """Drain Live Action Queue on main thread (Phase 4). Handle CODE, PREDICT, ERROR."""
    global _LIVE_SESSION_RUNNING
    if not _LIVE_SESSION_RUNNING:
        return None
    try:
        while not _LIVE_ACTION_QUEUE.empty():
            try:
                item = _LIVE_ACTION_QUEUE.get_nowait()
            except queue.Empty:
                break
            if not isinstance(item, dict):
                continue
            msg_type = item.get("type")
            payload = item.get("payload", "")
            if msg_type == "ERROR":
                print(f"[SpeechToBlender Live] ERROR: {payload}")
            elif msg_type == "CODE":
                blocked = False
                for forbidden in _LIVE_CODE_BLACKLIST:
                    if forbidden in payload:
                        print(f"[SpeechToBlender Live] BLOCKED (forbidden: {forbidden!r})")
                        blocked = True
                        break
                if not blocked:
                    try:
                        exec(payload, {"__builtins__": __builtins__, "bpy": bpy})
                        print("[SpeechToBlender Live] CODE executed")
                    except Exception as e:
                        import traceback
                        print(f"[SpeechToBlender Live] CODE exec failed: {e}")
                        traceback.print_exc()
            elif msg_type == "PREDICT":
                _execute_predict(payload)
    except Exception as e:
        print(f"[SpeechToBlender] Error draining live queue: {e}")
        import traceback
        traceback.print_exc()
    return 0.05 if _LIVE_SESSION_RUNNING else None


# ───────── Voice Process Management ─────────
def _get_voice_script_path():
    """Get the path to voice_to_blender.py script."""
    # Try to find it relative to the addon directory
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    voice_path = os.path.join(addon_dir, "voice_to_blender.py")
    if os.path.isfile(voice_path):
        return voice_path
    # Fallback to default
    return DEFAULT_VOICE_SCRIPT


def _bundled_python_exe():
    """Get path to bundled Python if available."""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(addon_dir, "stb_runtime", "python", "python.exe")
    if os.path.isfile(bundled):
        return bundled
    return None


def _ensure_openai_installed():
    """Ensure openai package is installed in bundled Python (if available)."""
    try:
        bundled_py = _bundled_python_exe()
        if not bundled_py:
            # No bundled Python, skip check
            return True
        
        # Check if openai is already installed
        result = subprocess.run(
            [bundled_py, "-c", "import openai"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
        
        # Install openai silently
        print("[SpeechToBlender] Installing openai package in bundled Python...")
        result = subprocess.run(
            [bundled_py, "-m", "pip", "install", "openai", "--quiet", "--disable-pip-version-check"],
            capture_output=True,
            timeout=120
        )
        if result.returncode == 0:
            print("[SpeechToBlender] ✅ openai package installed successfully")
            return True
        else:
            error_msg = result.stderr.decode("utf-8", errors="ignore") if result.stderr else "Unknown error"
            print(f"[SpeechToBlender] ⚠️ Failed to install openai: {error_msg[:200]}")
            return False
    except Exception as e:
        print(f"[SpeechToBlender] ⚠️ Error checking/installing openai: {e}")
        return False


def _resolve_python_exe():
    """Resolve a working Python executable to run the voice script."""
    def _ok(py):
        if not py:
            return False
        if os.path.isabs(py) and not os.path.isfile(py):
            return False
        try:
            p = subprocess.run([py, "--version"], capture_output=True, text=True, timeout=3)
            return p.returncode == 0
        except Exception:
            return False
    
    # Try in order: bundled, sys.executable, PATH python
    candidates = [
        _bundled_python_exe(),
        sys.executable,
        ("python.exe" if os.name == "nt" else "python"),
    ]
    
    for cand in candidates:
        if not cand:
            continue
        label = cand
        if not os.path.isabs(cand):
            w = shutil.which(cand)
            label = f"{cand} -> {w or 'NOT FOUND'}"
            cand = w or cand
        if _ok(cand):
            return cand
    
    return None


def _voice_is_running():
    """Check if voice process is still running."""
    global _VOICE_POPEN
    return _VOICE_POPEN is not None and _VOICE_POPEN.poll() is None


def _start_voice_process():
    """Start the voice_to_blender.py script."""
    global _VOICE_POPEN, _VOICE_RUNNING
    
    if _voice_is_running():
        print("[SpeechToBlender] Voice process already running")
        return True
    
    voice_path = _get_voice_script_path()
    if not os.path.isfile(voice_path):
        print(f"[SpeechToBlender] Voice script not found: {voice_path}")
        return False
    
    python_exe = _resolve_python_exe()
    if not python_exe:
        print("[SpeechToBlender] Could not resolve Python executable")
        return False
    
    try:
        print(f"[SpeechToBlender] Launching voice script: {voice_path}")
        print(f"[SpeechToBlender] Using Python: {python_exe}")
        
        env = os.environ.copy()
        # Forward OPENAI_API_KEY if present
        if "OPENAI_API_KEY" in os.environ:
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        
        # Try to set bundled whisper CLI if available
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        bundled_cli = os.path.join(addon_dir, "stb_runtime", "whisper", "whisper-cli.bat")
        if os.path.isfile(bundled_cli):
            env["WHISPER_CLI"] = bundled_cli
        
        # Let batch wrappers know which python to run
        env["STB_PYTHON_EXE"] = python_exe
        
        # Create a batch file wrapper that keeps console open
        if os.name == "nt":
            # Create temp batch file
            import tempfile
            batch_dir = tempfile.gettempdir()
            batch_file = os.path.join(batch_dir, "stb_voice_launcher.bat")
            
            # Write batch file that runs Python and pauses on error
            with open(batch_file, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{os.path.dirname(voice_path) or "."}"\n')
                f.write(f'"{python_exe}" "{voice_path}"\n')
                f.write(f'if errorlevel 1 (\n')
                f.write(f'    echo.\n')
                f.write(f'    echo Script exited with error code %errorlevel%\n')
                f.write(f'    pause\n')
                f.write(f')\n')
            
            # Launch batch file with new console
            _VOICE_POPEN = subprocess.Popen(
                ["cmd.exe", "/c", "start", "cmd.exe", "/k", batch_file],
                creationflags=subprocess.CREATE_NO_WINDOW,  # Don't create extra window for launcher
                env=env,
            )
        else:
            # Non-Windows: just run directly
            _VOICE_POPEN = subprocess.Popen(
                [python_exe, voice_path],
                cwd=os.path.dirname(voice_path) or None,
                env=env,
            )
        
        _VOICE_RUNNING = True
        print("[SpeechToBlender] Voice process started")
        print("[SpeechToBlender] A console window should have opened - check it for output/errors")
        
        # Wait a moment and check if it's still running
        time.sleep(1.0)  # Give it more time to start
        if _VOICE_POPEN.poll() is not None:
            # Process exited immediately
            print(f"[SpeechToBlender] ⚠️ Voice process launcher exited immediately with code: {_VOICE_POPEN.returncode}")
            print(f"[SpeechToBlender] Check the console window that should have opened")
            print(f"[SpeechToBlender] Or run manually to see errors:")
            print(f"[SpeechToBlender]   {python_exe} {voice_path}")
            _VOICE_POPEN = None
            _VOICE_RUNNING = False
            return False
        
        return True
        
    except FileNotFoundError as e:
        print(f"[SpeechToBlender] Python executable not found: {python_exe} - {e}")
        return False
    except Exception as e:
        print(f"[SpeechToBlender] Failed to start voice process: {e}")
        return False


def _stop_voice_process():
    """Stop the voice process."""
    global _VOICE_POPEN, _VOICE_RUNNING
    
    if not _voice_is_running():
        _VOICE_POPEN = None
        _VOICE_RUNNING = False
        return
    
    try:
        _VOICE_POPEN.terminate()
        _VOICE_POPEN.wait(timeout=3)
    except Exception:
        try:
            _VOICE_POPEN.kill()
        except Exception:
            pass
    
    _VOICE_POPEN = None
    _VOICE_RUNNING = False
    print("[SpeechToBlender] Voice process stopped")


# ───────── RPC Bridge Operators ─────────
class STB_OT_RPCStart(bpy.types.Operator):
    bl_idname = "stb.rpc_start"
    bl_label = "Start RPC"

    def execute(self, context):
        global _VOICE_LISTENING_ENABLED
        ok = _start_server_thread()
        if ok:
            context.window_manager.stb_rpc_server_running = True
            _VOICE_LISTENING_ENABLED = True  # Enable listening when RPC starts
            bpy.app.timers.register(_drain_task_queue, first_interval=0.3)
            # Automatically start voice script
            if _start_voice_process():
                self.report({'INFO'}, f"RPC server started on {HOST}:{PORT}, voice script launched")
            else:
                self.report({'INFO'}, f"RPC server started on {HOST}:{PORT} (voice script failed to start)")
        else:
            self.report({'ERROR'}, f"Couldn't start (port {PORT} in use?)")
        return {'FINISHED'}


class STB_OT_RPCStop(bpy.types.Operator):
    bl_idname = "stb.rpc_stop"
    bl_label = "Stop RPC"

    def execute(self, context):
        _stop_server_thread()
        context.window_manager.stb_rpc_server_running = False
        self.report({'INFO'}, "RPC server stopping…")
        return {'FINISHED'}


class STB_OT_ToggleVoiceListening(bpy.types.Operator):
    bl_idname = "stb.toggle_voice_listening"
    bl_label = "Toggle Voice Listening"
    bl_description = "Toggle voice listening on/off (Alt+F)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _VOICE_LISTENING_ENABLED
        _VOICE_LISTENING_ENABLED = not _VOICE_LISTENING_ENABLED
        status = "ON" if _VOICE_LISTENING_ENABLED else "OFF"
        self.report({'INFO'}, f"Voice listening: {status}")
        print(f"[SpeechToBlender] Voice listening toggled: {status}")
        
        # Force UI redraw to show status change immediately
        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        except Exception:
            pass
        
        return {'FINISHED'}


class STB_PT_RPCBridge(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STB"
    bl_label = "RPC Bridge"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        
        # Show status
        is_running = getattr(wm, "stb_rpc_server_running", False) or _SERVER_RUNNING
        voice_running = _voice_is_running()
        
        if is_running:
            layout.label(text=f"RPC: Running on {HOST}:{PORT}", icon="CHECKMARK")
            if voice_running:
                layout.label(text="Voice: Running", icon="CHECKMARK")
            else:
                layout.label(text="Voice: Stopped", icon="X")
            global _VOICE_LISTENING_ENABLED
            if _VOICE_LISTENING_ENABLED:
                layout.label(text="🟢 Listening: ON (Alt+F to toggle)", icon="SPEAKER")
            else:
                layout.label(text="🟡 Listening: OFF (Alt+F to toggle)", icon="SPEAKER")
            layout.operator("stb.rpc_stop", icon="PAUSE", text="Stop RPC")
        else:
            layout.label(text=f"RPC: Stopped", icon="X")
            layout.label(text="Voice: Stopped", icon="X")
            layout.operator("stb.rpc_start", icon="PLAY", text="Start RPC")


class STB_OT_MarkLastRunSuccess(bpy.types.Operator):
    """Mark the last run as a confirmed success"""
    bl_idname = "stb.mark_last_run_success"
    bl_label = "Mark Last Run as Success"
    bl_description = "Manually mark the most recent run as a confirmed success"
    
    def execute(self, context):
        try:
            import sys
            import os
            addon_path = os.path.dirname(os.path.abspath(__file__))
            if addon_path not in sys.path:
                sys.path.insert(0, addon_path)
            
            from lib.success_library import BlenderSuccessLibrary
            
            library = BlenderSuccessLibrary()
            last_run = library.get_last_run_info()
            
            if not last_run:
                self.report({'WARNING'}, "No recent runs found to mark as success")
                return {'CANCELLED'}
            
            # Promote to confirmed
            run_id = last_run.get("run_id")
            if library.promote_candidate_to_confirmed(run_id):
                self.report({'INFO'}, f"Marked run '{run_id}' as confirmed success")
            else:
                # If not a candidate, store as new confirmed
                screenshot_path = library.storage_path / last_run.get("screenshot_path", "")
                if screenshot_path.exists():
                    with open(screenshot_path, "rb") as f:
                        screenshot = f.read()
                    
                    library.store_run(
                        target_object=last_run.get("target_object", ""),
                        screenshot=screenshot,
                        commands=last_run.get("commands", []),
                        quality_score=last_run.get("quality_score", 0.0),
                        target_match_score=last_run.get("target_match_score", 0.0),
                        scene_analysis=last_run.get("scene_analysis", {}),
                        mesh_analysis=last_run.get("mesh_analysis", {}),
                        transcript=last_run.get("transcript", ""),
                        summary=last_run.get("summary", ""),
                        iterations=last_run.get("iterations", 0),
                        status="confirmed"
                    )
                    self.report({'INFO'}, f"Stored run as confirmed success")
                else:
                    self.report({'WARNING'}, "Could not find screenshot for last run")
                    return {'CANCELLED'}
            
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error marking success: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class STB_OT_DeleteRun(bpy.types.Operator):
    """Delete a run from the success library"""
    bl_idname = "stb.delete_run"
    bl_label = "Delete Run"
    bl_description = "Delete a run from the success library (for removing falsely marked successes)"
    bl_options = {'REGISTER', 'UNDO'}
    
    run_id: bpy.props.StringProperty(name="Run ID", description="ID of the run to delete")
    
    def execute(self, context):
        try:
            import sys
            import os
            addon_path = os.path.dirname(os.path.abspath(__file__))
            if addon_path not in sys.path:
                sys.path.insert(0, addon_path)
            
            from lib.success_library import BlenderSuccessLibrary
            
            library = BlenderSuccessLibrary()
            
            if library.delete_run(self.run_id):
                self.report({'INFO'}, f"Deleted run '{self.run_id}'")
            else:
                self.report({'WARNING'}, f"Run '{self.run_id}' not found")
            
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error deleting run: {e}")
            return {'CANCELLED'}


class STB_OT_DeleteLastRun(bpy.types.Operator):
    """Delete the last run from the success library"""
    bl_idname = "stb.delete_last_run"
    bl_label = "Delete Last Run"
    bl_description = "Delete the most recent run (for removing falsely marked successes)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        try:
            import sys
            import os
            addon_path = os.path.dirname(os.path.abspath(__file__))
            if addon_path not in sys.path:
                sys.path.insert(0, addon_path)
            
            from lib.success_library import BlenderSuccessLibrary
            
            library = BlenderSuccessLibrary()
            last_run = library.get_last_run_info()
            
            if not last_run:
                self.report({'WARNING'}, "No recent runs found to delete")
                return {'CANCELLED'}
            
            run_id = last_run.get("run_id")
            if library.delete_run(run_id):
                self.report({'INFO'}, f"Deleted run '{run_id}'")
            else:
                self.report({'WARNING'}, f"Could not delete run '{run_id}'")
            
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error deleting run: {e}")
            return {'CANCELLED'}


class STB_PT_SuccessLibrary(Panel):
    """Success Library Panel"""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STB"
    bl_label = "Success Library"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        try:
            import sys
            import os
            addon_path = os.path.dirname(os.path.abspath(__file__))
            if addon_path not in sys.path:
                sys.path.insert(0, addon_path)
            
            from lib.success_library import BlenderSuccessLibrary
            library = BlenderSuccessLibrary()
            
            # Show last run info
            last_run = library.get_last_run_info()
            if last_run:
                box = layout.box()
                box.label(text=f"Last Run: {last_run.get('target_object', 'Unknown')}", icon="MESH_DATA")
                box.label(text=f"Quality: {last_run.get('quality_score', 0):.2f}")
                box.label(text=f"Target Match: {last_run.get('target_match_score', 0):.2f}")
                box.label(text=f"Status: {last_run.get('status', 'unknown')}")
                box.label(text=f"Run ID: {last_run.get('run_id', 'unknown')[:20]}...")
                
                layout.separator()
                
                if last_run.get('status') != 'confirmed':
                    layout.operator("stb.mark_last_run_success", icon="CHECKMARK", text="Mark as Success")
                else:
                    layout.label(text="✓ Already confirmed", icon="CHECKMARK")
                
                layout.operator("stb.delete_last_run", icon="TRASH", text="Delete Last Run")
            else:
                layout.label(text="No recent runs", icon="INFO")
            
            # Show stats
            layout.separator()
            confirmed_count = len(list(library.confirmed_path.glob("*_metadata.json")))
            candidate_count = len(list(library.candidates_path.glob("*_metadata.json")))
            
            stats_box = layout.box()
            stats_box.label(text="Library Stats", icon="BOOKMARKS")
            stats_box.label(text=f"Confirmed: {confirmed_count}", icon="CHECKMARK")
            stats_box.label(text=f"Candidates: {candidate_count}", icon="OUTLINER_OB_MESH")
            
            # Show recent runs list
            if confirmed_count > 0 or candidate_count > 0:
                layout.separator()
                recent_box = layout.box()
                recent_box.label(text="Recent Runs:", icon="TIME")
                
                all_runs = library.get_all_runs()[:5]  # Show last 5
                for run in all_runs:
                    row = recent_box.row()
                    status_icon = "CHECKMARK" if run.get("status") == "confirmed" else "QUESTION"
                    row.label(text=f"{run.get('target_object', 'Unknown')[:20]}", icon=status_icon)
                    row.label(text=f"Q:{run.get('quality_score', 0):.2f}")
                    
                    # Delete button for each run
                    op = row.operator("stb.delete_run", text="", icon="TRASH")
                    op.run_id = run.get("run_id", "")
            
        except Exception as e:
            layout.label(text=f"Error: {e}", icon="ERROR")
            import traceback
            traceback.print_exc()


class STB_OT_SubmitTextCommand(bpy.types.Operator):
    """Submit typed text command to voice processing"""
    bl_idname = "stb.submit_text_command"
    bl_label = "Submit Text Command"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        wm = context.window_manager
        text = getattr(wm, "stb_text_input", "").strip()
        
        if not text:
            self.report({'WARNING'}, "Please enter a command")
            return {'CANCELLED'}
        
        # Write text to a file that the voice script can read
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            text_queue_file = os.path.join(temp_dir, "stb_text_queue.txt")
            
            # Append to queue file (voice script will read and clear)
            with open(text_queue_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            
            # Clear the input field
            wm.stb_text_input = ""
            
            self.report({'INFO'}, f"Command submitted: {text[:50]}...")
            print(f"[SpeechToBlender] Text command submitted: {text}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to submit: {str(e)}")
            print(f"[SpeechToBlender] Error submitting text command: {e}")
            return {'CANCELLED'}


class STB_PT_VoiceMode(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STB"
    bl_label = "Voice Mode"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        
        box = layout.box()
        col = box.column(align=True)
        col.label(text="⚡ Context-Aware Voice Modeling", icon="SPEAKER")
        
        # Target info
        try:
            addon_prefs = context.preferences.addons[ADDON_ROOT].preferences
            target = addon_prefs.super_mode_target_object
        except Exception:
            target = ""
        
        if target:
            col.label(text=f"Modeling Target: {target}", icon="OBJECT_DATA")
        else:
            col.label(text="Set a target in preferences for richer context", icon="INFO")
        
        react_enabled = False
        try:
            react_enabled = bool(addon_prefs.use_react_reasoning)
        except Exception:
            react_enabled = False
        icon = "LOOP_BACK" if react_enabled else "LOOP_FORWARDS"
        status = "ReAct reasoning: ON (iterative)" if react_enabled else "ReAct reasoning: OFF (single-shot)"
        col.label(text=status, icon=icon)
        
        col.separator()
        col.label(text="Capabilities:", icon="DOT")
        col.label(text="• Multi-step operations", icon="BLANK1")
        col.label(text="• Geometry analysis", icon="BLANK1")
        col.label(text="• Context-aware GPT", icon="BLANK1")
        col.label(text="• Any Blender operation", icon="BLANK1")
        col.label(text="• Smart operation selection", icon="BLANK1")
        col.separator()
        col.label(text="⚠ Uses API credits when GPT is needed", icon="INFO")
        
        # Text input section
        col.separator()
        col.separator()
        text_box = col.box()
        text_col = text_box.column(align=True)
        text_col.label(text="📝 Type Command (for users without microphone)", icon="TEXT")
        text_col.prop(wm, "stb_text_input", text="", icon="CONSOLE")
        text_col.operator("stb.submit_text_command", text="Submit Command", icon="PLAY")


# ───────── Safe stub: lazy import inside register, timers last ─────────
_CLASSES = (
    STB_AddonPreferences,
    STB_OT_MeshyGenerate,
    STB_PT_MeshyTools,
    STB_PT_MeshyStatus,
    STB_OT_RPCStart,
    STB_OT_RPCStop,
    STB_OT_ToggleVoiceListening,
    STB_PT_RPCBridge,
    STB_PT_VoiceMode,
    STB_OT_MarkLastRunSuccess,
    STB_OT_DeleteRun,
    STB_OT_DeleteLastRun,
    STB_PT_SuccessLibrary,
)

def register():
    # 1) prefs first
    bpy.utils.register_class(STB_AddonPreferences)
    bpy.utils.register_class(STB_OT_MarkLastRunSuccess)
    bpy.utils.register_class(STB_OT_DeleteRun)
    bpy.utils.register_class(STB_OT_DeleteLastRun)
    bpy.utils.register_class(STB_PT_SuccessLibrary)

    # 2) ensure WM props before any panel draw
    if not hasattr(bpy.types.WindowManager, "stb_meshy_prompt"):
        bpy.types.WindowManager.stb_meshy_prompt = StringProperty(
            name="Meshy Prompt",
            default="simple low-poly test object",
        )
    if not hasattr(bpy.types.WindowManager, "stb_rpc_server_running"):
        bpy.types.WindowManager.stb_rpc_server_running = bpy.props.BoolProperty(
            name="RPC Server Running",
            default=False,
            options={'HIDDEN'},
        )
    if not hasattr(bpy.types.WindowManager, "stb_text_input"):
        bpy.types.WindowManager.stb_text_input = StringProperty(
            name="Text Command",
            default="",
            description="Type your command here if you can't use the microphone",
        )
    # 3) operators and panels
    bpy.utils.register_class(STB_OT_MeshyGenerate)
    bpy.utils.register_class(STB_PT_MeshyTools)
    bpy.utils.register_class(STB_PT_MeshyStatus)
    bpy.utils.register_class(STB_OT_RPCStart)
    bpy.utils.register_class(STB_OT_RPCStop)
    bpy.utils.register_class(STB_OT_ToggleVoiceListening)
    bpy.utils.register_class(STB_OT_SubmitTextCommand)
    bpy.utils.register_class(STB_PT_RPCBridge)
    bpy.utils.register_class(STB_PT_VoiceMode)
    
    # 4) Register Alt+F keyboard shortcut for voice listening toggle
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Window', space_type='EMPTY')
        kmi = km.keymap_items.new(
            STB_OT_ToggleVoiceListening.bl_idname,
            'F',
            'PRESS',
            alt=True
        )
        print("[SpeechToBlender] Registered Alt+F shortcut for voice listening toggle")

    # 4) ensure openai is available in bundled Python
    _ensure_openai_installed()

    # 6) lazy import real module, never crash on error
    try:
        # Import your heavy modules late
        from . import stb_core  # noqa: F401
    except Exception as e:
        # do not raise, just log so Blender does not auto‑disable
        print("[SpeechToBlender] STARTUP ERROR:", e)

    # 7) timers last if you add them later


def unregister():
    # Unregister keyboard shortcut
    try:
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.addon
        if kc:
            km = kc.keymaps.get('Window')
            if km:
                for kmi in list(km.keymap_items):
                    if kmi.idname == STB_OT_ToggleVoiceListening.bl_idname:
                        km.keymap_items.remove(kmi)
    except Exception:
        pass
    
    # panels and ops
    for cls in reversed(_CLASSES[1:]):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    # prefs
    try:
        bpy.utils.unregister_class(STB_AddonPreferences)
    except Exception:
        pass

    # Stop server and voice process if running
    try:
        _stop_server_thread()
        _stop_voice_process()
    except Exception:
        pass
    
    # WM props
    try:
        if hasattr(bpy.types.WindowManager, "stb_meshy_prompt"):
            del bpy.types.WindowManager.stb_meshy_prompt
        if hasattr(bpy.types.WindowManager, "stb_rpc_server_running"):
            del bpy.types.WindowManager.stb_rpc_server_running
    except Exception:
        pass
