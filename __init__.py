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
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from bpy.props import StringProperty, EnumProperty
from bpy.types import AddonPreferences, Operator, Panel

ADDON_ROOT = (__package__ or __name__).split(".")[0]  # "SpeechToBlender"

# ───────── RPC Server State ─────────
HOST = "127.0.0.1"
PORT = 8765  # Match voice_to_blender.py expectation
_SERVER_THREAD = None
_SERVER = None
_SERVER_RUNNING = False
_TASKQ = queue.Queue()

# ───────── Voice Process State ─────────
_VOICE_POPEN = None
_VOICE_RUNNING = False
DEFAULT_VOICE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_to_blender.py")

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
        
        # Split key fields
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
        
        # Legacy field (hidden but kept for compatibility)
        # col.prop(self, "openai_api_key")  # Hidden - use parts instead
        
        col.separator()
        col.label(text="Used for natural language command understanding", icon="INFO")


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
                if op:
                    ok, msg = _safe_call_operator(op, kwargs)
                    return {"ok": ok, "message": msg}
        return {"ok": False, "error": "Invalid command format"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
            
            # Register RPC methods
            def ping():
                return "pong"
            
            def enqueue_op_safe(op_name="wm.redraw_timer", kwargs=None):
                kwargs = kwargs or {}
                print(f"[SpeechToBlender] RPC enqueue: {op_name} {kwargs}")
                _TASKQ.put(("OP_SAFE", op_name, kwargs))
                return "enqueued"
            
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
            
            server.register_function(ping, "ping")
            server.register_function(enqueue_op_safe, "enqueue_op_safe")
            server.register_function(enqueue_op_safe, "enqueue_op")  # Alias
            server.register_function(_rpc_execute, "execute")
            server.register_function(get_openai_api_key, "get_openai_api_key")
            
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
    try:
        while not _TASKQ.empty():
            msg = _TASKQ.get_nowait()
            kind, name, kwargs = msg
            if kind == "OP_SAFE":
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
    except queue.Empty:
        pass
    except Exception as e:
        print(f"[SpeechToBlender] Error draining task queue: {e}")
        import traceback
        traceback.print_exc()
    return 0.5 if _SERVER_RUNNING else None


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
    
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE
    
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
        
        _VOICE_POPEN = subprocess.Popen(
            [python_exe, voice_path],
            creationflags=creationflags,
            cwd=os.path.dirname(voice_path) or None,
            env=env,
        )
        _VOICE_RUNNING = True
        print("[SpeechToBlender] Voice process started")
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
        ok = _start_server_thread()
        if ok:
            context.window_manager.stb_rpc_server_running = True
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
            layout.operator("stb.rpc_stop", icon="PAUSE", text="Stop RPC")
        else:
            layout.label(text=f"RPC: Stopped", icon="X")
            layout.label(text="Voice: Stopped", icon="X")
            layout.operator("stb.rpc_start", icon="PLAY", text="Start RPC")


# ───────── Safe stub: lazy import inside register, timers last ─────────
_CLASSES = (
    STB_AddonPreferences,
    STB_OT_MeshyGenerate,
    STB_PT_MeshyTools,
    STB_PT_MeshyStatus,
    STB_OT_RPCStart,
    STB_OT_RPCStop,
    STB_PT_RPCBridge,
)

def register():
    # 1) prefs first
    bpy.utils.register_class(STB_AddonPreferences)

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

    # 3) operators and panels
    bpy.utils.register_class(STB_OT_MeshyGenerate)
    bpy.utils.register_class(STB_PT_MeshyTools)
    bpy.utils.register_class(STB_PT_MeshyStatus)
    bpy.utils.register_class(STB_OT_RPCStart)
    bpy.utils.register_class(STB_OT_RPCStop)
    bpy.utils.register_class(STB_PT_RPCBridge)

    # 4) ensure openai is available in bundled Python
    _ensure_openai_installed()

    # 5) lazy import real module, never crash on error
    try:
        # Import your heavy modules late
        from . import stb_core  # noqa: F401
    except Exception as e:
        # do not raise, just log so Blender does not auto‑disable
        print("[SpeechToBlender] STARTUP ERROR:", e)

    # 6) timers last if you add them later


def unregister():
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
