# voice_to_blender.py
#
# Pipeline:
# 1) Record from mic until silence
# 2) Transcribe with whisper-cli (Python wrapper, prints JSON)
# 3) Try local intent rules first (fast, offline)
# 4) If no local match, ask GPT for {"op": "...", "kwargs": {...}} (JSON only)
# 5) Send to Blender RPC: enqueue_op_safe(op, kwargs)

import os, sys, time, wave, tempfile, subprocess, json, re, math
import numpy as np
import sounddevice as sd
import xmlrpc.client
from datetime import datetime

# ========= CONFIG (portable paths) =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The add-on launcher passes STB_PYTHON_EXE and (if present) WHISPER_CLI.
# Defaults for fully self-contained bundle:
DEFAULT_BUNDLED_CLI = os.path.join(SCRIPT_DIR, "stb_runtime", "whisper", "whisper-cli.bat")
WHISPER_CLI = os.getenv("WHISPER_CLI", DEFAULT_BUNDLED_CLI)

# faster-whisper model name (not a file path). You can set FW_DEVICE=cpu|cuda|auto in env.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANG  = os.getenv("WHISPER_LANG", "en")

# Safety-Gate add-on listens here and exposes /RPC2
RPC_URL = "http://127.0.0.1:8765/RPC2"
# ==========================================

# Mic settings
SAMPLE_RATE  = 16000
BLOCK_SEC    = 0.2
SILENCE_RMS  = 500
SILENCE_HOLD = 0.8
MIN_SPOKEN   = 0.7

def _log_paths_once():
    print("[Voice] Python:", sys.executable)
    print("[Voice] WHISPER_CLI:", WHISPER_CLI, ("OK" if os.path.isfile(WHISPER_CLI) else "MISSING"))
    print("[Voice] Model:", WHISPER_MODEL, "Lang:", WHISPER_LANG)

_log_paths_once()

# Connect to Blender RPC
rpc = xmlrpc.client.ServerProxy(RPC_URL, allow_none=True)

# ------------------ Helpers ------------------
def rms_int16(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.int32)**2)))

def _parse_number(s):
    try:
        return float(s)
    except Exception:
        return None

def _extract_value_after(words, *keys, default=None):
    t = " ".join(words)
    for k in keys:
        m = re.search(rf"{re.escape(k)}\s+(-?\d+(?:\.\d+)?)", t)
        if m:
            v = _parse_number(m.group(1))
            if v is not None:
                return v
    return default

def _extract_triplet_after(words, *keys):
    t = " ".join(words)
    for k in keys:
        m = re.search(rf"{re.escape(k)}\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", t)
        if m:
            a = _parse_number(m.group(1))
            b = _parse_number(m.group(2))
            c = _parse_number(m.group(3))
            if None not in (a, b, c):
                return (a, b, c)
    return None

def _deg2rad(v):
    try:
        return math.radians(float(v))
    except Exception:
        return None

# ------------------ Recorder ------------------
def record_until_silence():
    print("🎙️ Speak command…")
    frames, spoken_sec, silence_sec = [], 0.0, 0.0
    block_samples = int(BLOCK_SEC * SAMPLE_RATE)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            block, _ = stream.read(block_samples)
            block = block.reshape(-1)
            frames.append(block)
            level = rms_int16(block)
            if level < SILENCE_RMS:
                silence_sec += BLOCK_SEC
            else:
                spoken_sec += BLOCK_SEC
                silence_sec = 0.0
            if silence_sec >= SILENCE_HOLD and spoken_sec >= MIN_SPOKEN:
                break
    audio = np.concatenate(frames, axis=0).astype(np.int16)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpdir = os.path.join(tempfile.gettempdir(), "voice_clips")
    os.makedirs(tmpdir, exist_ok=True)
    wav_path = os.path.join(tmpdir, f"clip_{ts}.wav")
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    print(f"💾 Saved: {wav_path}")
    return wav_path

# ------------------ Transcription (Python CLI wrapper) ------------------
def _extract_json(lines):
    """Scan backwards for the first JSON-looking line and parse it."""
    for ln in reversed(lines):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                continue
    return None

def transcribe(wav_path: str) -> str | None:
    if not os.path.isfile(WHISPER_CLI):
        print("⚠️ whisper-cli not found:", WHISPER_CLI)
        return None

    cmd = [WHISPER_CLI, "--model", WHISPER_MODEL, "--language", WHISPER_LANG, "--input", wav_path]

    # Silence noisy libs in the child process.
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("KMP_WARNINGS", "0")
    env.setdefault("CT2_VERBOSE", "0")
    env.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except Exception as e:
        print("⚠️ whisper-cli failed:", e)
        return None

    out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    lines = [ln for ln in out.splitlines() if ln.strip()]

    data = _extract_json(lines)
    if not data:
        tail = lines[-1] if lines else "<empty>"
        print("⚠️ Non-JSON whisper output tail:", tail[:300])
        return None

    if "error" in data:
        print("⚠️ whisper error:", data["error"])
        return None

    return (data.get("text") or "").strip()

# ------------------ Local Intent (offline) ------------------
MESH_PRIMS = {
    "cube":              ("mesh.primitive_cube_add",      {"size": "size"}),
    "uv sphere":         ("mesh.primitive_uv_sphere_add", {"radius": "size"}),
    "ico sphere":        ("mesh.primitive_ico_sphere_add",{"radius": "size"}),
    "cylinder":          ("mesh.primitive_cylinder_add",  {"radius": "size"}),
    "cone":              ("mesh.primitive_cone_add",      {"radius1": "size"}),
    "torus":             ("mesh.primitive_torus_add",     {"major_radius": "size"}),
    "plane":             ("mesh.primitive_plane_add",     {"size": "size"}),
    "grid":              ("mesh.primitive_grid_add",      {"size": "size"}),
    "monkey":            ("mesh.primitive_monkey_add",    {}),
    "circle":            ("mesh.primitive_circle_add",    {"radius": "size"}),
}

CURVE_PRIMS = {
    "bezier curve":      ("curve.primitive_bezier_curve_add", {}),
    "bezier circle":     ("curve.primitive_bezier_circle_add", {}),
    "nurbs curve":       ("curve.primitive_nurbs_curve_add", {}),
    "nurbs circle":      ("curve.primitive_nurbs_circle_add", {}),
    "path":              ("curve.primitive_nurbs_path_add", {}),
    "text":              ("object.text_add", {}),
}

LIGHTS = {
    "point light": ("object.light_add", {"type": "POINT"}),
    "sun":         ("object.light_add", {"type": "SUN"}),
    "spot":        ("object.light_add", {"type": "SPOT"}),
    "area light":  ("object.light_add", {"type": "AREA"}),
    "light":       ("object.light_add", {"type": "POINT"}),
}

CAMERA = {
    "camera": ("object.camera_add", {})
}

def _match_any_phrase(text, phrases):
    t = text.lower()
    for p in phrases:
        if p in t:
            return True
    return False

def _find_key_phrase(text, keys):
    t = text.lower()
    for k in keys:
        if k in t:
            return k
    return None

def _extract_common_kwargs(words):
    loc = _extract_triplet_after(words, "at", "location", "loc")
    rot_deg = _extract_triplet_after(words, "rotate", "rotation", "rot")
    rot = tuple(_deg2rad(v) for v in rot_deg) if rot_deg else None
    size = _extract_value_after(words, "size", "scale", default=None)
    kwargs = {}
    if loc: kwargs["location"] = loc
    if rot: kwargs["rotation"] = rot
    if size is not None: kwargs["size"] = float(size)
    return kwargs

def _maybe_quantity(text):
    m = re.search(r"\b(add|spawn|create)\s+(\d+)\b", text.lower())
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return 1
    return 1

def try_local_rules(text: str):
    t = (text or "").strip()
    if not t:
        return None
    tl = t.lower()
    words = tl.split()

    if _match_any_phrase(tl, ["select all"]):
        return {"op": "object.select_all", "kwargs": {"action": "SELECT"}}
    if _match_any_phrase(tl, ["deselect all", "clear selection"]):
        return {"op": "object.select_all", "kwargs": {"action": "DESELECT"}}
    if _match_any_phrase(tl, ["frame selected", "focus selected", "zoom to selected"]):
        return {"op": "view3d.view_selected", "kwargs": {}}

    if _match_any_phrase(tl, ["delete selected", "delete object", "remove object", "erase object"]):
        return {"op": "object.delete", "kwargs": {}}
    if _match_any_phrase(tl, ["duplicate", "duplicate object", "copy object", "make a copy"]):
        return {"op": "object.duplicate_move", "kwargs": {}}

    move = _extract_triplet_after(words, "move", "translate", "offset", "by")
    if move:
        return {"op": "transform.translate", "kwargs": {"value": move}}
    if _match_any_phrase(tl, ["scale up", "bigger", "increase size"]):
        return {"op": "transform.resize", "kwargs": {"value": (1.2, 1.2, 1.2)}}
    if _match_any_phrase(tl, ["scale down", "smaller", "decrease size"]):
        return {"op": "transform.resize", "kwargs": {"value": (0.8, 0.8, 0.8)}}
    m = re.search(r"(scale|resize)\s+(?:to\s+)?(\d+(\.\d+)?)", tl)
    if m:
        s = float(m.group(2))
        return {"op": "transform.resize", "kwargs": {"value": (s, s, s)}}
    m = re.search(r"rotate\s+(x|y|z)\s+(-?\d+(?:\.\d+)?)", tl)
    if m:
        axis = m.group(1).lower()
        deg = float(m.group(2))
        return {"op": "transform.rotate", "kwargs": {"value": math.radians(deg), "orient_axis": axis.upper(), "orient_type": "GLOBAL"}}

    if _match_any_phrase(tl, ["new collection", "create collection", "make collection"]):
        name_m = re.search(r"(?:called|named)\s+([a-z0-9 _-]+)", tl)
        nm = name_m.group(1).strip().title() if name_m else "Voice Collection"
        return {"op": "collection.create", "kwargs": {"name": nm}}

    if _match_any_phrase(tl, ["add", "make", "create", "spawn", "insert"]):
        qty = _maybe_quantity(tl)

        def make_add_cmd(template):
            op, param_map = template
            base_kwargs = _extract_common_kwargs(words)
            kwargs = dict(base_kwargs)
            if "size" in kwargs:
                for k, alias in param_map.items():
                    if alias == "size":
                        kwargs[k] = kwargs.pop("size")
                        break
            return {"op": op, "kwargs": kwargs}

        for key, template in MESH_PRIMS.items():
            if key in tl:
                return [make_add_cmd(template) for _ in range(qty)] if qty > 1 else make_add_cmd(template)
        for key, template in CURVE_PRIMS.items():
            if key in tl or (key == "text" and "text" in tl):
                return [make_add_cmd(template) for _ in range(qty)] if qty > 1 else make_add_cmd(template)
        for key, template in LIGHTS.items():
            if key in tl:
                return [make_add_cmd(template) for _ in range(qty)] if qty > 1 else make_add_cmd(template)
        for key, template in CAMERA.items():
            if key in tl:
                return [make_add_cmd(template) for _ in range(qty)] if qty > 1 else make_add_cmd(template)

    return None

# ------------------ GPT Fallback ------------------
def gpt_to_json(transcript: str):
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY not set; skipping GPT fallback.")
        return None
    try:
        from openai import OpenAI
    except Exception:
        print("⚠️ openai package not available; pip install openai")
        return None

    system = (
        "You are a Blender automation agent.\n"
        "Output ONLY raw JSON (no prose, no code fences).\n"
        "Each command must be of the form:{\"op\":\"<module.op>\",\"kwargs\":{}}.\n"
        "If multiple steps are implied, output a JSON array of such dicts.\n"
        "Prefer creative operators (object/mesh/curve/transform/material/node/render).\n"
        "Never use file/quit/addon/script/image.save operators."
    )
    user = f"Instruction: {transcript}\nReturn JSON only."

    print("🧠 GPT mapping…")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip()
        out = out.replace("```json", "").replace("```", "").strip()
        if out.lower().startswith("json"):
            out = out.split("\n", 1)[-1].strip()
        try:
            cmd = json.loads(out)
        except Exception as e:
            print("⚠️ JSON parse error:", e, "RAW:", out[:2000])
            return None
        if isinstance(cmd, dict) and "op" in cmd:
            return cmd
        if isinstance(cmd, list) and all(isinstance(x, dict) and "op" in x for x in cmd):
            return cmd
        print("⚠️ GPT returned JSON but missing 'op':", str(out)[:500])
        return None
    except Exception as e:
        print("⚠️ GPT error:", e)
        return None

# ------------------ RPC send ------------------
def send_to_blender(cmd: dict):
    if not isinstance(cmd, dict):
        print("⚠️ command is not a dict:", cmd); return
    op = cmd.get("op"); kwargs = cmd.get("kwargs", {}) or {}
    if not op or not isinstance(op, str):
        print("⚠️ Missing/invalid 'op' in command:", cmd); return
    try:
        print(f"➡️ enqueue {op} {kwargs}")
        res = rpc.enqueue_op_safe(op, kwargs)
        print("✅ RPC:", res)
    except Exception as e:
        print("❌ RPC failed:", e)

# ------------------ Main loop ------------------
if __name__ == "__main__":
    print("Voice → Whisper → (Local rules / GPT) → Blender via Safety Gate (Ctrl+C to exit)")
    try:
        print("ping:", rpc.ping())
    except Exception as e:
        print("⚠️ Could not reach Blender RPC at", RPC_URL, ":", e)

    while True:
        try:
            wav = record_until_silence()
            text = transcribe(wav)
            if not text:
                print("⚠️ No transcription.")
                continue
            print("📝 Transcript ->", text)

            cmd = try_local_rules(text) or gpt_to_json(text)

            if cmd:
                if isinstance(cmd, list):
                    for single in cmd:
                        send_to_blender(single)
                        time.sleep(0.05)
                else:
                    send_to_blender(cmd)
            else:
                print("🤷 No command derived for:", text)

            time.sleep(0.2)

        except KeyboardInterrupt:
            print("\nBye.")
            sys.exit(0)
