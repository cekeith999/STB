bl_info = {
    "name": "Blender RPC Bridge (RPC tab, voice controls) – with Safety Gate + Meshy",
    "author": "you",
    "version": (0, 7, 0),
    "blender": (3, 6, 0),
    "category": "System",
    "location": "3D View > N-panel > RPC",
    "description": "XML-RPC server + Voice launcher with diagnostics, operator Safety Gate, and Meshy text→3D import (API only)",
}

import bpy, threading, queue, time, os, subprocess, sys, socket, shutil, json, re, textwrap, tempfile
import urllib.request, urllib.error
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

# ====== CONFIG ======
DEFAULT_VOICE_SCRIPT = os.path.join(os.path.dirname(__file__), "voice_to_blender.py")
DEFAULT_PYTHON_EXE   = ""

HOST = "127.0.0.1"
PORT = 8765

API_BASE = "https://api.meshy.ai/openapi/v2"   # Meshy REST base

# ====== GLOBAL STATE ======
_SERVER_THREAD = None
_SERVER = None
_SERVER_RUNNING = False
_TASKQ = queue.Queue()

_VOICE_POPEN = None
_VOICE_RUNNING = False

# Meshy main-thread import queue (workers enqueue downloaded .glb paths here)
_MESHY_IMPORT_Q = queue.Queue()

# ====== UTIL ======
def _print(*a):
    print("[RPC Bridge]", *a)

def _set_error(msg: str):
    bpy.context.window_manager.rpc_last_error = msg or ""

def _port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0

def _get_addon_prefs():
    try:
        modname = __name__
        if modname in bpy.context.preferences.addons:
            return bpy.context.preferences.addons[modname].preferences, True
    except Exception:
        pass
    return None, False

def _get_effective_voice_path():
    prefs, ok = _get_addon_prefs()
    if ok and prefs and getattr(prefs, "voice_script", ""):
        return prefs.voice_script
    wm = bpy.context.window_manager
    p = (wm.rpc_voice_path or "").strip()
    return p if p else DEFAULT_VOICE_SCRIPT

def _bundled_python_exe():
    # self-contained portable interpreter
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "stb_runtime", "python", "python.exe")

def _bundled_cli_path():
    # self-contained whisper CLI next to the portable python
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "stb_runtime", "whisper", "whisper-cli.bat")

# --- resolve a Python to run the voice script ---
def _resolve_python_exe():
    """
    Choose a working Python:
      1) bundled ./stb_runtime/python/python.exe (must run)
      2) user override (UI field)
      3) current interpreter (sys.executable)
      4) PATH 'python'
      5) DEFAULT_PYTHON_EXE (if set)
    """
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

    wm = bpy.context.window_manager
    user = (wm.rpc_python_exe or "").strip()

    tried = []

    for cand in [
        _bundled_python_exe(),
        user,
        sys.executable,
        ("python.exe" if os.name == "nt" else "python"),
        DEFAULT_PYTHON_EXE,
    ]:
        if not cand:
            continue
        label = cand
        if not os.path.isabs(cand):
            w = shutil.which(cand)
            label = f"{cand} -> {w or 'NOT FOUND'}"
            cand = w or cand
        tried.append(label)
        if _ok(cand):
            return cand, tried

    return None, tried

# ====== WHISPER CLI FIRST-RUN SETUP (fallback only) ======
WHISPER_TOOL_REL = os.path.join("tools", "whisper")
WHISPER_FILES = {
    "whisper_cli.py": r'''# -*- coding: utf-8 -*-
# Minimal CLI wrapper around faster-whisper. Prints a one-line JSON: {"text": "..."}
import sys, os, json

def parse(args):
    cfg = {"model": "small", "language": "en", "input": None}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--model" and i+1 < len(args):
            i += 1; cfg["model"] = args[i]
        elif a == "--language" and i+1 < len(args):
            i += 1; cfg["language"] = args[i]
        elif a == "--input" and i+1 < len(args):
            i += 1; cfg["input"] = args[i]
        i += 1
    return cfg

def main():
    cfg = parse(sys.argv[1:])
    if not cfg["input"] or not os.path.exists(cfg["input"]):
        print(json.dumps({"error":"missing or invalid --input"})); sys.exit(2)
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(json.dumps({"error": f"faster-whisper not installed: {e}"})); sys.exit(3)

    device = os.environ.get("FW_DEVICE","cpu")  # cpu|cuda|auto
    compute_type = "int8" if device == "cpu" else "float16"

    try:
        model = WhisperModel(cfg["model"], device=device, compute_type=compute_type)
        segments, info = model.transcribe(cfg["input"], language=cfg["language"], vad_filter=True)
        out_text = "".join(s.text for s in segments)
        print(json.dumps({"text": out_text.strip()}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)})); sys.exit(4)

if __name__ == "__main__":
    main()
''',
    "whisper-cli.bat": r'''@echo off
setlocal
set PY=python
if not "%STB_PYTHON_EXE%"=="" set PY="%STB_PYTHON_EXE%"
set CLI=%~dp0whisper_cli.py
%PY% "%CLI%" %*
'''
}

def _user_scripts_dir():
    return bpy.utils.user_resource('SCRIPTS')

def _whisper_dir():
    return os.path.join(_user_scripts_dir(), WHISPER_TOOL_REL)

def _write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# robust Python for pip (rarely used now that we're self-contained)
def _blender_python():
    try:
        p = getattr(bpy.app, "binary_path_python", None)
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    if sys.executable and os.path.isfile(sys.executable):
        return sys.executable
    w = shutil.which("python.exe" if os.name == "nt" else "python")
    return w or ("python.exe" if os.name == "nt" else "python")

def _pip_install(packages):
    py = _blender_python()
    try:
        subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception:
        pass
    try:
        cmd = [py, "-m", "pip", "install"] + packages
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        _print("pip install failed:", e)
        return False

def ensure_whisper_cli_ready():
    """
    Prefer the bundled CLI if present. Otherwise, create a fallback CLI in the
    user-scripts folder and (if possible) install faster-whisper there.
    Returns a path to a runnable whisper-cli.*.
    """
    # 0) Bundled self-contained CLI wins
    bundled_cli = _bundled_cli_path()
    if os.path.isfile(bundled_cli):
        _print(f"whisper CLI (bundled) at: {bundled_cli}")
        return bundled_cli

    # 1) Respect environment override
    env_cli = os.environ.get("WHISPER_CLI")
    if env_cli and os.path.exists(env_cli):
        _print(f"Using WHISPER_CLI override: {env_cli}")
        return env_cli

    # 2) Lay down a fallback in user scripts
    base = _whisper_dir()
    try:
        os.makedirs(base, exist_ok=True)
        for name, content in WHISPER_FILES.items():
            _write_file(os.path.join(base, name), content)
    except Exception as e:
        _print(f"Failed to create whisper files: {e}")
        return None

    # 3) Try to ensure faster-whisper in whichever Python _blender_python() points to
    try:
        import faster_whisper  # noqa: F401
        dep_ok = True
    except Exception:
        _print("Installing faster-whisper into Blender’s Python (fallback only)…")
        dep_ok = _pip_install(["faster-whisper"])
    if not dep_ok:
        _print('⚠ Could not install faster-whisper. Manual fix:\n    "%s" -m pip install faster-whisper' % _blender_python())

    cli = os.path.join(base, "whisper-cli.bat")
    if os.path.exists(cli):
        _print(f"whisper CLI ready at: {cli}")
        return cli
    _print("whisper-cli.bat missing after setup")
    return None

# ====== SAFETY GATE ======
_DENY_PATTERNS = [
    r"^wm\.save_.*",
    r"^wm\.open_.*",
    r"^wm\.read_.*",
    r"^wm\.revert_.*",
    r"^wm\.quit_blender$",
    r"^wm\.link$",
    r"^wm\.append$",
    r"^wm\.path_open$",
    r"^wm\.url_.*",
    r"^script\.python_file_run$",
    r"^script\.reload$",
    r"^preferences\.addon_.*",
    r"^image\.save_.*",
    r"^outliner\.id_operation$",
    r"^file\..*",
    r"^wm\.call_menu.*$",
]

_DEFAULT_ALLOW_PREFIXES = [
    "object.","mesh.","curve.","surface.","lattice.","armature.",
    "grease_pencil.","gpencil.","annotation.","sculpt.","paint.","uv.",
    "shader.","node.","material.","texture.","brush.","view3d.","screen.",
    "wm.redraw_timer","transform.","render.","collection.","group.",
    "import_scene.","export_scene.","bpy_ops_proxy.",
]

def _compile_user_regex_list(text):
    lines = (text or "").splitlines()
    out = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(re.compile(s))
        except re.error:
            _print(f"Invalid regex ignored: {s}")
    return out

def _make_regexes():
    prefs, ok = _get_addon_prefs()
    user_denies, user_allows = [], []
    if ok and prefs:
        user_denies = _compile_user_regex_list(getattr(prefs, "safety_extra_deny", ""))
        user_allows = _compile_user_regex_list(getattr(prefs, "safety_extra_allow", ""))
    deny = [re.compile(p) for p in _DENY_PATTERNS] + user_denies
    allow_prefixes = list(_DEFAULT_ALLOW_PREFIXES)
    if ok and prefs:
        if prefs.allow_import_export is False:
            allow_prefixes = [p for p in allow_prefixes if not (p.startswith("import_") or p.startswith("export_"))]
        if prefs.allow_view3d_menus is True:
            deny = [rx for rx in deny if rx.pattern != r"^wm\.call_menu.*$"]
    allow = [re.compile(rf"^{re.escape(pref)}") for pref in allow_prefixes] + user_allows
    return deny, allow

def _is_safe_op(op_fullname: str):
    op = (op_fullname or "").strip()
    if not op or "." not in op:
        return False, "Malformed operator name"
    deny, allow = _make_regexes()
    for rx in deny:
        if rx.search(op):
            return False, f"Blocked by Safety Gate deny rule: {rx.pattern}"
    for rx in allow:
        if rx.search(op):
            return True, "Allowed by Safety Gate"
    prefs, ok = _get_addon_prefs()
    safe_mode = bool(getattr(prefs, "safety_strict_mode", True)) if ok else True
    if safe_mode:
        return False, "Not in allow-list while Safety Strict Mode is ON"
    else:
        return True, "Allowed (Strict Mode OFF)"

def _safe_call_operator(op_fullname: str, kwargs: dict):
    ok, reason = _is_safe_op(op_fullname)
    if not ok:
        _print(f"[SAFETY BLOCK] {op_fullname} -> {reason}")
        wm = bpy.context.window_manager
        wm.rpc_last_blocked = json.dumps({"operator": op_fullname, "reason": reason, "kwargs": kwargs or {}}, indent=2)
        _set_error(f"Blocked: {op_fullname}\n{reason}")
        return False, reason
    try:
        cat, name = op_fullname.split(".", 1)
        cat_obj = getattr(bpy.ops, cat)
        fn = getattr(cat_obj, name)
        fn(**(kwargs or {}))
        return True, "OK"
    except AttributeError as e:
        return False, f"Operator not found: {op_fullname} ({e})"
    except TypeError as e:
        return False, f"Bad arguments for {op_fullname}: {e}"
    except Exception as e:
        return False, f"Operator error {op_fullname}: {e}"

# ====== MESHY HELPERS ======
def _meshy_headers():
    prefs, ok = _get_addon_prefs()
    key = (getattr(prefs, "meshy_api_key", "") if ok and prefs else "").strip()
    if not key:
        raise RuntimeError("Set your Meshy API key in Add-on Preferences.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def _http_post_json(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=_meshy_headers(), method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def _http_get_json(url):
    req = urllib.request.Request(url, headers=_meshy_headers(), method="GET")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def _download_to(url, dst_path):
    urllib.request.urlretrieve(url, dst_path)

def _ensure_sandbox_collection(ctx):
    col = bpy.data.collections.get("Meshy_Imports")
    if not col:
        col = bpy.data.collections.new("Meshy_Imports")
        ctx.scene.collection.children.link(col)
    elif col.name not in ctx.scene.collection.children:
        ctx.scene.collection.children.link(col)
    return col

def _import_glb_on_main(glb_path):
    """Main-thread import of a GLB, then move to Meshy_Imports collection."""
    try:
        ok, _ = _safe_call_operator("import_scene.gltf", {"filepath": glb_path})
        if not ok:
            return
        col = _ensure_sandbox_collection(bpy.context)
        for obj in list(bpy.context.selected_objects):
            # unlink from any other collections and relink into sandbox
            for c in list(obj.users_collection):
                try:
                    c.objects.unlink(obj)
                except Exception:
                    pass
            if col not in obj.users_collection:
                col.objects.link(obj)
        _print(f"[Meshy] Imported: {glb_path}")
    except Exception as e:
        _print(f"[Meshy] Import error: {e}")

def _meshy_import_timer():
    """Timer that drains the Meshy import queue on the main thread."""
    drained = False
    while not _MESHY_IMPORT_Q.empty():
        drained = True
        path = _MESHY_IMPORT_Q.get()
        _import_glb_on_main(path)
    # Re-schedule based on preference
    prefs, ok = _get_addon_prefs()
    secs = max(1.0, float(getattr(prefs, "meshy_poll_seconds", 4))) if ok and prefs else 4.0
    return secs

def _meshy_worker(prompt, do_refine=True, should_remesh=True):
    """Background thread: Meshy preview -> optional refine -> download GLB -> enqueue path."""
    try:
        _print(f"[Meshy] start: {prompt}")
        preview = _http_post_json(f"{API_BASE}/text-to-3d", {
            "mode": "preview",
            "prompt": prompt,
            "should_remesh": bool(should_remesh)
        })
        task_id = preview["result"]

        # Poll preview
        while True:
            st = _http_get_json(f"{API_BASE}/text-to-3d/{task_id}")
            s = st.get("status")
            if s == "SUCCEEDED":
                url = st["model_urls"]["glb"]
                break
            if s in {"FAILED", "CANCELED"}:
                raise RuntimeError(f"Meshy preview failed: {st}")
            time.sleep(5)

        final_url = url
        if do_refine:
            ref = _http_post_json(f"{API_BASE}/text-to-3d", {"mode": "refine", "preview_task_id": task_id})
            ref_id = ref["result"]
            while True:
                st2 = _http_get_json(f"{API_BASE}/text-to-3d/{ref_id}")
                s2 = st2.get("status")
                if s2 == "SUCCEEDED":
                    final_url = st2["model_urls"]["glb"]
                    break
                if s2 in {"FAILED", "CANCELED"}:
                    raise RuntimeError(f"Meshy refine failed: {st2}")
                time.sleep(5)

        tmpdir = tempfile.mkdtemp(prefix="meshy_")
        glb_path = os.path.join(tmpdir, "meshy_model.glb")
        _download_to(final_url, glb_path)
        _MESHY_IMPORT_Q.put(glb_path)
        _print(f"[Meshy] done → {glb_path}")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "ignore")[:300]
        except Exception:
            body = "<no body>"
        _print(f"[Meshy] HTTP {e.code}: {body}")
        _set_error(f"Meshy HTTP {e.code}: {body}")
    except Exception as e:
        _print(f"[Meshy] worker error: {e}")
        _set_error(f"Meshy error: {e}")

# ====== SIMPLE VOICE COMMAND ROUTER ======
_PRIMS = {
    "cube":       lambda size: _safe_call_operator("mesh.primitive_cube_add", {"size": size}),
    "sphere":     lambda size: _safe_call_operator("mesh.primitive_uv_sphere_add", {"radius": size/2}),
    "ico sphere": lambda size: _safe_call_operator("mesh.primitive_ico_sphere_add", {"radius": size/2, "subdivisions": 3}),
    "cylinder":   lambda size: _safe_call_operator("mesh.primitive_cylinder_add", {"radius": size/2, "depth": size}),
    "cone":       lambda size: _safe_call_operator("mesh.primitive_cone_add", {"radius1": size/2, "depth": size}),
    "torus":      lambda size: _safe_call_operator("mesh.primitive_torus_add", {}),
    "plane":      lambda size: _safe_call_operator("mesh.primitive_plane_add", {"size": size}),
}

def _parse_size(text):
    """
    Extract a numeric size in Blender units from phrases like:
    '2', '2m', '2 meters', '0.5', '50 cm' (default 2.0)
    """
    m = re.search(r"(\d+(\.\d+)?)\s*(m|meter|meters|cm|centimeter|centimeters)?", text or "")
    if not m:
        return 2.0
    val = float(m.group(1))
    unit = (m.group(3) or "").lower()
    if unit in {"cm", "centimeter", "centimeters"}:
        return val / 100.0
    return val

def _ensure_link_to_sandbox_selected():
    col = _ensure_sandbox_collection(bpy.context)
    for obj in list(bpy.context.selected_objects):
        if col not in obj.users_collection:
            try:
                col.objects.link(obj)
            except Exception:
                pass

def handle_voice_command(text):
    """
    Routes English commands to safe primitives or Meshy.
    Blocks any file ops by keyword.
    """
    t = (text or "").strip().lower()
    if not t:
        return "Empty command."

    # Block obvious dangerous intents
    if any(w in t for w in ["delete", "erase", "save", "overwrite", "quit", "close file", "new file", "open file"]):
        return "Blocked: file operations are not allowed."

    # PRIMITIVE: "add cube 2 meters", "create plane 50 cm"
    m = re.match(r"(add|create|spawn)\s+([a-z ]+?)(?:\s+(\d.*))?$", t)
    if m:
        prim_name = m.group(2).strip().replace("icosphere", "ico sphere")
        size = _parse_size(m.group(3) or "")
        # fuzzy match
        pick = None
        for k in _PRIMS.keys():
            if prim_name == k or prim_name.startswith(k):
                pick = k; break
        if pick:
            ok, msg = _PRIMS[pick](size)
            if ok:
                _ensure_link_to_sandbox_selected()
                return f"Added {pick} (size ~{size:.2f})."
            return f"Blocked/failed: {pick} -> {msg}"

    # MESHY: "meshy a glossy red mask...", "create XXX with meshy"
    if t.startswith("meshy "):
        prompt = t[len("meshy "):].strip()
        if not prompt:
            return "Give me a prompt after 'meshy …'."
        threading.Thread(target=_meshy_worker, args=(prompt, True, True), daemon=True).start()
        return f"Meshy started: {prompt}. Will import into Meshy_Imports."

    mm = re.match(r"(create|generate|make)\s+(.+?)\s+(with|using)\s+meshy", t)
    if mm:
        prompt = mm.group(2).strip()
        threading.Thread(target=_meshy_worker, args=(prompt, True, True), daemon=True).start()
        return f"Meshy started: {prompt}. Will import into Meshy_Imports."

    return "Command not recognized. Try: 'add cube', 'add sphere 2 meters', or 'meshy a futuristic mask'."

# ====== XML-RPC SERVER THREAD ======
class _RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)

def _server_loop():
    global _SERVER, _SERVER_RUNNING
    try:
        with SimpleXMLRPCServer(
            (HOST, PORT),
            requestHandler=_RequestHandler,
            allow_none=True,
            logRequests=False,
        ) as server:
            _SERVER = server

            def ping():
                return "pong"

            def enqueue_op(op_name="wm.redraw_timer", kwargs=None):
                kwargs = kwargs or {}
                _TASKQ.put(("OP_SAFE", op_name, kwargs))
                return "enqueued"

            def enqueue_op_safe(op_name="wm.redraw_timer", kwargs=None):
                kwargs = kwargs or {}
                _TASKQ.put(("OP_SAFE", op_name, kwargs))
                return "enqueued"

            def safety_info():
                deny, allow = _make_regexes()
                prefs, ok = _get_addon_prefs()
                return {
                    "strict_mode": bool(getattr(prefs, "safety_strict_mode", True)) if ok else True,
                    "deny": [rx.pattern for rx in deny],
                    "allow": [rx.pattern for rx in allow],
                }

            def voice_handle(text):
                """Handle a natural-language command (primitive or Meshy)."""
                try:
                    msg = handle_voice_command(text)
                    return {"ok": True, "message": msg}
                except Exception as e:
                    return {"ok": False, "error": str(e)}

            server.register_function(ping, "ping")
            server.register_function(enqueue_op, "enqueue_op")
            server.register_function(enqueue_op_safe, "enqueue_op_safe")
            server.register_function(safety_info, "safety_info")
            server.register_function(voice_handle, "voice_handle")

            _SERVER_RUNNING = True
            _print(f"XML-RPC listening on http://{HOST}:{PORT}/RPC2")
            while _SERVER_RUNNING:
                server.handle_request()
    except Exception as e:
        _print("Server loop error:", e)
        _set_error(f"Server error: {e!r}")
    finally:
        _SERVER_RUNNING = False
        _print("Server loop ended")

def _start_server_thread():
    global _SERVER_THREAD, _SERVER_RUNNING
    if _SERVER_RUNNING:
        _print("Server already running")
        return True
    if _port_in_use(HOST, PORT):
        msg = f"Port {PORT} already in use; not starting."
        _print(msg); _set_error(msg)
        return False
    _SERVER_THREAD = threading.Thread(target=_server_loop, daemon=True)
    _SERVER_THREAD.start()
    for _ in range(20):
        if _SERVER_RUNNING:
            break
        time.sleep(0.05)
    return _SERVER_RUNNING

def _stop_server_thread():
    global _SERVER_RUNNING
    if not _SERVER_RUNNING:
        return
    _SERVER_RUNNING = False
    try:
        with socket.create_connection((HOST, PORT), timeout=0.2) as s:
            pass
    except Exception:
        pass
    _print("Requested server stop")

# ====== VOICE PROCESS CONTROL ======
def _voice_is_running():
    global _VOICE_POPEN
    return _VOICE_POPEN is not None and _VOICE_POPEN.poll() is None

def _start_voice_process(py_path):
    global _VOICE_POPEN, _VOICE_RUNNING
    if _voice_is_running():
        _print("Voice process already running")
        return True

    if not os.path.isfile(py_path):
        msg = f"Voice script not found: {py_path}"
        _print(msg); _set_error(msg)
        return False

    resolved, tried = _resolve_python_exe()
    if not resolved:
        msg = "Could not resolve a Python executable.\nTried:\n - " + "\n - ".join(tried)
        _print(msg); _set_error(msg)
        return False

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    try:
        _print("Launching voice script with:", resolved)
        _print("Voice script path:", py_path)

        env = os.environ.copy()

        # Prefer bundled whisper CLI
        bcli = _bundled_cli_path()
        if os.path.isfile(bcli):
            env["WHISPER_CLI"] = bcli

        # let batch wrappers know which python to run
        env["STB_PYTHON_EXE"] = resolved

        # Fallback: create/install a user-scripts CLI if bundled isn't present
        if "WHISPER_CLI" not in env:
            cli_path = ensure_whisper_cli_ready()
            if cli_path:
                env.setdefault("WHISPER_CLI", cli_path)
            else:
                _print("⚠ whisper CLI setup failed; transcription may be unavailable")

        # Forward API key if present
        if "OPENAI_API_KEY" in os.environ:
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

        _VOICE_POPEN = subprocess.Popen(
            [resolved, py_path],
            creationflags=creationflags,
            cwd=os.path.dirname(py_path) or None,
            env=env,
        )

    except FileNotFoundError as e:
        msg = f"Python executable not found: {resolved}\nDetails: {e}"
        _print(msg); _set_error(msg)
        return False
    except Exception as e:
        msg = f"Failed to start voice process with {resolved}\nDetails: {e}"
        _print(msg); _set_error(msg)
        return False

def _stop_voice_process():
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
    _print("Voice subprocess stopped")

# ====== MAIN THREAD TIMERS ======
def _drain_task_queue():
    try:
        while True:
            msg = _TASKQ.get_nowait()
            if not msg:
                break
            kind, name, kwargs = msg
            if kind == "OP_SAFE":
                ok, reason = _safe_call_operator(name, kwargs)
                if not ok:
                    _print(f"Queued op blocked: {name} ({reason})")
                try:
                    bpy.ops.wm.redraw_timer(type="DRAW_WIN", iterations=1)
                except Exception:
                    pass
    except queue.Empty:
        pass
    return 0.5 if _SERVER_RUNNING else None

# ====== ADD-ON PREFERENCES ======
class RPCBRIDGE_AddonPrefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    voice_script: bpy.props.StringProperty(name="Voice Script Path", subtype='FILE_PATH', default=DEFAULT_VOICE_SCRIPT)
    safety_strict_mode: bpy.props.BoolProperty(name="Safety Strict Mode", description="When ON, only operators matching the allow-list are permitted; OFF lets anything not explicitly denied through", default=True)
    allow_import_export: bpy.props.BoolProperty(name="Allow Import/Export Operators", default=True, description="If OFF, blocks import_scene.* / export_scene.*")
    allow_view3d_menus: bpy.props.BoolProperty(name="Allow Menu Calls", default=False, description="If ON, unblocks wm.call_menu style ops (still UI-only)")
    safety_extra_allow: bpy.props.StringProperty(name="Extra Allow Regex (one per line)", description="Advanced: add regex lines like ^pose\\. or ^sequencer\\.", subtype='NONE', default="")
    safety_extra_deny: bpy.props.StringProperty(name="Extra Deny Regex (one per line)", description="Advanced: add regex lines; deny always wins", subtype='NONE', default="")
    # Meshy prefs
    meshy_api_key: bpy.props.StringProperty(name="Meshy API Key", subtype='PASSWORD', description="Format: msy-RtHUDJezqJJG7KlQK0UNosTVemaIMGEmqh6C", default="")
    meshy_poll_seconds: bpy.props.IntProperty(name="Meshy importer poll (sec)", default=4, min=2, max=30)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Paths")
        layout.prop(self, "voice_script")
        layout.separator()
        layout.label(text="Safety Gate")
        layout.prop(self, "safety_strict_mode")
        layout.prop(self, "allow_import_export")
        layout.prop(self, "allow_view3d_menus")
        col = layout.column()
        col.prop(self, "safety_extra_allow")
        col.prop(self, "safety_extra_deny")
        layout.separator()
        layout.label(text="Meshy")
        layout.prop(self, "meshy_api_key")
        layout.prop(self, "meshy_poll_seconds")

# ====== PROPS ======
def ensure_props():
    wm = bpy.types.WindowManager
    if not hasattr(wm, "rpc_server_running"):
        wm.rpc_server_running = bpy.props.BoolProperty(name="Server Running", default=False, options={'HIDDEN'})
    if not hasattr(wm, "rpc_voice_running"):
        wm.rpc_voice_running = bpy.props.BoolProperty(name="Voice Running", default=False, options={'HIDDEN'})
    if not hasattr(wm, "rpc_voice_path"):
        wm.rpc_voice_path = bpy.props.StringProperty(name="Voice Script Path", subtype='FILE_PATH', default=DEFAULT_VOICE_SCRIPT)
    if not hasattr(wm, "rpc_python_exe"):
        wm.rpc_python_exe = bpy.props.StringProperty(name="Python Executable", subtype='FILE_PATH', default=DEFAULT_PYTHON_EXE)
    if not hasattr(wm, "rpc_last_error"):
        wm.rpc_last_error = bpy.props.StringProperty(name="Last Error", default="", options={'HIDDEN'})
    if not hasattr(wm, "rpc_last_blocked"):
        wm.rpc_last_blocked = bpy.props.StringProperty(name="Last Blocked (JSON)", default="", options={'HIDDEN'})

# ====== OPERATORS ======
class RPCBRIDGE_OT_server_toggle(bpy.types.Operator):
    bl_idname = "rpcbridge.server_toggle"
    bl_label = "Start/Stop RPC Server"
    bl_options = {'INTERNAL'}
    start: bpy.props.BoolProperty(default=True)
    def execute(self, context):
        if self.start:
            ok = _start_server_thread()
            if ok:
                context.window_manager.rpc_server_running = True
                bpy.app.timers.register(_drain_task_queue, first_interval=0.3)
                # ensure Meshy import timer is running
                bpy.app.timers.register(_meshy_import_timer, first_interval=3.0)
                self.report({'INFO'}, f"RPC server started on {HOST}:{PORT}")
            else:
                self.report({'ERROR'}, f"Couldn't start (port {PORT} in use?)")
        else:
            _stop_server_thread()
            context.window_manager.rpc_server_running = False
            self.report({'INFO'}, "RPC server stopping…")
        return {'FINISHED'}

class RPCBRIDGE_OT_voice_toggle(bpy.types.Operator):
    bl_idname = "rpcbridge.voice_toggle"
    bl_label = "Start/Stop Voice"
    bl_options = {'INTERNAL'}
    start: bpy.props.BoolProperty(default=True)
    def execute(self, context):
        voice_path = _get_effective_voice_path()
        if self.start:
            ok = _start_voice_process(voice_path)
            if ok:
                context.window_manager.rpc_voice_running = True
                self.report({'INFO'}, "Voice script started")
            else:
                self.report({'ERROR'}, "Failed to start voice script (see Last Error / console)")
        else:
            _stop_voice_process()
            context.window_manager.rpc_voice_running = False
            self.report({'INFO'}, "Voice script stopped")
        return {'FINISHED'}

class RPCBRIDGE_OT_console(bpy.types.Operator):
    bl_idname = "rpcbridge.toggle_console"
    bl_label = "Toggle System Console (Windows)"
    bl_options = {'INTERNAL'}
    def execute(self, context):
        try:
            bpy.ops.wm.console_toggle()
        except Exception:
            pass
        return {'FINISHED'}

class RPCBRIDGE_OT_validate(bpy.types.Operator):
    """Check Python, modules, voice script file, and RPC connectivity."""
    bl_idname = "rpcbridge.validate_env"
    bl_label = "Validate Environment"
    bl_options = {'INTERNAL'}
    def execute(self, context):
        voice_path = _get_effective_voice_path()
        pyexe, tried = _resolve_python_exe()
        report = {"python_exe": pyexe or "NOT RESOLVED", "tried": tried, "voice_script": voice_path, "steps": []}
        def step(name, ok, detail=""):
            report["steps"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not pyexe:
            _set_error("Python not resolved. See console for tried paths.")
            self.report({'ERROR'}, "Python not resolved")
            _print("Validate report:", json.dumps(report, indent=2))
            return {'CANCELLED'}
        # prefer bundled CLI if present; else fallback
        bcli = _bundled_cli_path()
        step("bundled whisper-cli", os.path.isfile(bcli), bcli)
        if not os.path.isfile(bcli):
            cli = ensure_whisper_cli_ready()
            step("fallback whisper-cli presence", bool(cli), cli or "not ready")
        # python --version
        try:
            out = subprocess.run([pyexe, "--version"], capture_output=True, text=True)
            step("python --version", out.returncode == 0, (out.stdout or out.stderr).strip())
        except Exception as e:
            step("python --version", False, repr(e))
        # module imports
        code = "import sounddevice, numpy, xmlrpc.client; print('OK')"
        try:
            out = subprocess.run([pyexe, "-c", code], capture_output=True, text=True)
            ok = (out.returncode == 0 and "OK" in (out.stdout + out.stderr))
            step("module imports", ok, (out.stdout or out.stderr).strip())
        except Exception as e:
            step("module imports", False, repr(e))
        # voice script exists
        step("voice script exists", os.path.isfile(voice_path), voice_path)
        # RPC ping
        try:
            import xmlrpc.client as _x
            s = _x.ServerProxy(f"http://{HOST}:{PORT}/RPC2")
            pong = s.ping()
            step("RPC ping", pong == "pong", f"ping -> {pong!r}")
        except Exception as e:
            step("RPC ping", False, repr(e))
        # summarize
        bad = [s for s in report["steps"] if not s["ok"]]
        if bad:
            msg = "Validation failed:\n" + "\n".join(f"- {b['name']}: {b['detail']}" for b in bad)
            _set_error(msg)
            self.report({'ERROR'}, "Validation failed. See Last Error / console.")
        else:
            _set_error("")
            self.report({'INFO'}, "Environment OK")
        _print("Validate report:", json.dumps(report, indent=2))
        return {'FINISHED'}

# ====== PANELS ======
class RPCBRIDGE_PT_panel(bpy.types.Panel):
    bl_label = "Blender RPC Bridge"
    bl_idname = "RPCBRIDGE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RPC"
    def draw(self, context):
        wm = context.window_manager
        layout = self.layout
        status = layout.column(align=True)
        status.label(text=f"Server: {'RUNNING' if wm.rpc_server_running else 'stopped'}")
        status.label(text=f"Voice:  {'RUNNING' if wm.rpc_voice_running else 'stopped'}")
        col = layout.column(align=True)
        row = col.row(align=True)
        row.operator("rpcbridge.server_toggle", text="Start Server", icon='PLAY').start = True
        row.operator("rpcbridge.server_toggle", text="Stop Server", icon='PAUSE').start = False
        row = col.row(align=True)
        row.operator("rpcbridge.voice_toggle", text="Start Voice", icon='PLAY').start = True
        row.operator("rpcbridge.voice_toggle", text="Stop Voice", icon='PAUSE').start = False
        layout.separator()
        prefs, using_prefs = _get_addon_prefs()
        if using_prefs:
            layout.label(text="Voice path is in Add-on Preferences.", icon='PREFERENCES')
        else:
            layout.prop(context.window_manager, "rpc_voice_path", text="Voice Script")
        layout.prop(context.window_manager, "rpc_python_exe", text="Python Executable")
        row = layout.row(align=True)
        row.operator("rpcbridge.validate_env", icon='CHECKMARK')
        row.operator("rpcbridge.toggle_console", icon='CONSOLE')
        box = layout.box()
        box.label(text="Safety Gate", icon='LOCKED')
        prefs, ok = _get_addon_prefs()
        strict = True
        if ok and prefs:
            strict = bool(prefs.safety_strict_mode)
        box.label(text=f"Strict Mode: {'ON' if strict else 'OFF'}")
        if wm.rpc_last_blocked:
            box2 = box.box()
            box2.label(text="Last Blocked")
            for line in wm.rpc_last_blocked.splitlines():
                box2.label(text=line)
        if wm.rpc_last_error:
            box = layout.box()
            box.label(text="Last Error", icon='ERROR')
            for line in wm.rpc_last_error.splitlines():
                box.label(text=line)



# ====== VOICE OPERATOR (routes to handle_voice_command) ======
class VOICE_OT_Handle(bpy.types.Operator):
    """Route a natural-language command to primitives/Meshy"""
    bl_idname = "voice.handle"
    bl_label  = "Handle Voice Command"
    bl_options = {'INTERNAL'}
    text: bpy.props.StringProperty(name="Text", description="Transcript to execute")

    def execute(self, context):
        try:
            msg = handle_voice_command(self.text)
            self.report({'INFO'}, msg)
            print(f"[VoiceMeshy] {msg}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            print(f"[VoiceMeshy] ERROR: {e}")
            return {'CANCELLED'}

# ====== REGISTER ======
_CLASSES = (
    RPCBRIDGE_AddonPrefs,
    RPCBRIDGE_OT_server_toggle,
    RPCBRIDGE_OT_voice_toggle,
    RPCBRIDGE_OT_console,
    RPCBRIDGE_OT_validate,
    RPCBRIDGE_PT_panel,,
    VOICE_OT_Handle,
)

def register():
    ensure_props()
    for cls in _CLASSES:
        try:
            bpy.utils.register_class(cls)
        except Exception:
            pass
    # kick off Meshy import timer so it runs even if server is off
    bpy.app.timers.register(_meshy_import_timer, first_interval=3.0)
    _print("REGISTER OK")

def unregister():
    _stop_voice_process()
    _stop_server_thread()
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _print("UNREGISTER OK")

if __name__ == "__main__":
    register()
