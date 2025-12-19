# voice_to_blender.py
#
# Pipeline:
# 1) Record from mic until silence
# 2) Transcribe with whisper-cli (Python wrapper, prints JSON)
# 3) Try local intent rules first (fast, offline)
# 4) If no local match, ask GPT for {"op": "...", "kwargs": {...}} (JSON only)
# 5) Send to Blender RPC: enqueue_op_safe(op, kwargs)

import os, sys, time, wave, tempfile, subprocess, json, re, math, pathlib
import numpy as np
import sounddevice as sd
import xmlrpc.client
from datetime import datetime
import traceback  # keep tracebacks visible instead of killing process
import webrtcvad

# ========= CONFIG (portable paths) =========
def _get_openai_api_key():
    """Get OpenAI API key from cache, RPC (preferences), or environment variable (fallback)."""
    global _CACHED_API_KEY, _CACHE_VALID
    
    # Use cached key if available and valid
    if _CACHE_VALID and _CACHED_API_KEY:
        if VERBOSE_DEBUG:
            print(f"[DEBUG] Using cached API key: length={len(_CACHED_API_KEY)}, preview={_CACHED_API_KEY[:15]}...")
        return _CACHED_API_KEY
    
    # Try RPC (from Blender preferences) first - this is the preferred source
    if rpc is not None:
        try:
            key = rpc.get_openai_api_key()
            if VERBOSE_DEBUG:
                print(f"[DEBUG] RPC returned key: length={len(key) if key else 0}, preview={key[:15] if key and len(key) > 15 else (key if key else 'None')}...")
            
            if key and key.strip():
                key = key.strip()
                # Remove any hidden whitespace characters (newlines, tabs, etc.)
                key = ''.join(key.split())
                # Remove non-printable characters
                key = ''.join(c for c in key if c.isprintable())
                if key:
                    # Cache the key
                    _CACHED_API_KEY = key
                    _CACHE_VALID = True
                    if VERBOSE_DEBUG:
                        print(f"[DEBUG] ✅ Cached RPC key from preferences: length={len(key)}, preview={key[:15]}...")
                    return key
            elif VERBOSE_DEBUG:
                print(f"[DEBUG] RPC returned empty or invalid key")
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[DEBUG] RPC call failed: {e}")
                import traceback
                traceback.print_exc()
            pass  # RPC not available or method doesn't exist
    
    # Fall back to environment variable only if RPC/preferences key is not available
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        # Clean environment variable key
        env_key = ''.join(env_key.split())
        env_key = ''.join(c for c in env_key if c.isprintable())
        if env_key:
            # Cache the env key too
            _CACHED_API_KEY = env_key
            _CACHE_VALID = True
            if VERBOSE_DEBUG:
                print(f"[DEBUG] ✅ Cached environment variable key (fallback): length={len(env_key)}, preview={env_key[:15]}...")
            return env_key
        elif VERBOSE_DEBUG:
            print(f"[DEBUG] Environment variable key is empty after cleaning")
    
    return ""


def _clear_api_key_cache():
    """Clear the cached API key (call when RPC stops)."""
    global _CACHED_API_KEY, _CACHE_VALID
    _CACHED_API_KEY = None
    _CACHE_VALID = False
    if VERBOSE_DEBUG:
        print("[DEBUG] Cleared API key cache")

# Cached API key (set when RPC is available, cleared when RPC stops)
_CACHED_API_KEY = None
_CACHE_VALID = False

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

# Feature flags / verbosity
ENABLE_GPT_FALLBACK = True   # GPT-4o fallback for natural language understanding
VERBOSE_DEBUG = False        # debug prints for import matcher (set True for debugging)

# Modeling target / reasoning state (from Blender)
SUPER_MODE_TARGET = ""
USE_REACT_REASONING = False

# Mic settings
SAMPLE_RATE  = 16000
BLOCK_SEC    = 0.2
SILENCE_HOLD = 0.6   # Silence duration required to stop (using consecutive blocks)
MIN_SPOKEN   = 0.3   # Minimum speech time before silence can trigger stop
MAX_RECORD_SEC = 8.0  # Maximum recording time to prevent infinite loops
INITIAL_SILENCE_ABORT = 2.0  # Stop early if no speech detected after this many seconds
VAD_FRAME_MS = 30  # Frame size used for WebRTC VAD (10, 20, or 30 ms only)
VAD_AGGRESSIVENESS = 2  # 0=least aggressive, 3=most aggressive
VAD = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def _log_paths_once():
    print("[Voice] Python:", sys.executable)
    print("[Voice] WHISPER_CLI:", WHISPER_CLI, ("OK" if os.path.isfile(WHISPER_CLI) else "MISSING"))
    print("[Voice] Model:", WHISPER_MODEL, "Lang:", WHISPER_LANG)

_log_paths_once()

# Connect to Blender RPC
rpc = xmlrpc.client.ServerProxy(RPC_URL, allow_none=True)

# ------------------ Helpers ------------------

def _dbg(*args):
    if VERBOSE_DEBUG:
        print("[VOICE-IO]", *args)
        sys.stdout.flush()

def rms_int16(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.int32)**2)))


def _block_has_voice(block: np.ndarray) -> bool:
    """Return True if any sub-frame in this block is classified as speech by VAD."""
    if VAD is None:
        return False
    frame_samples = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
    if frame_samples <= 0:
        return False
    frame_bytes = frame_samples * 2  # int16 -> 2 bytes
    raw = block.astype(np.int16).tobytes()
    total = len(raw)
    if total < frame_bytes:
        return False
    for start in range(0, total - frame_bytes + 1, frame_bytes):
        chunk = raw[start:start + frame_bytes]
        if len(chunk) < frame_bytes:
            break
        try:
            if VAD.is_speech(chunk, SAMPLE_RATE):
                return True
        except Exception:
            # Ignore badly sized frames
            continue
    return False

def _parse_number(s):
    try:
        return float(s)
    except Exception:
        return None

def _word_to_number(word):
    """Convert word numbers to integers."""
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20
    }
    return word_map.get(word.lower(), None)

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


def _should_use_react_command(text: str) -> bool:
    """Heuristic to decide whether to invoke ReAct reasoning."""
    if not text:
        return False
    lowered = text.lower()
    complex_markers = [
        "then ",
        "after ",
        "before ",
        "all of them",
        "each of them",
        "together",
        "loop cut",
        "loopcut",
        "subdivide",
        "modifier",
        "bevel",
        "boolean",
        "material",
        "glass",
        "extrude",
        "bridge edge",
        "connect",
    ]
    if any(marker in lowered for marker in complex_markers):
        return True
    if " and " in lowered and len(text.split()) >= 10:
        return True
    if len(text) > 60:
        return True
    return False


def _capture_screen_local():
    """Capture screen using local Python libraries (PIL/mss) - runs in voice script, not Blender."""
    try:
        import base64
        import tempfile
        import os
        
        # Method 1: Try PIL/Pillow
        try:
            from PIL import ImageGrab
            print("[Screenshot] Trying PIL.ImageGrab...")
            screenshot = ImageGrab.grab()
            temp_path = tempfile.mktemp(suffix='.png')
            screenshot.save(temp_path, 'PNG')
            
            with open(temp_path, 'rb') as f:
                image_data = f.read()
                screenshot_data = base64.b64encode(image_data).decode('utf-8')
            
            os.remove(temp_path)
            print(f"[Screenshot] ✅ PIL.ImageGrab succeeded: {len(image_data)} bytes raw, {len(screenshot_data)} chars base64")
            return screenshot_data
        except ImportError as e:
            print(f"[Screenshot] ⚠️ PIL not available: {e}")
        except Exception as e:
            print(f"[Screenshot] ⚠️ PIL.ImageGrab failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Method 2: Try mss
        try:
            import mss
            print("[Screenshot] Trying mss...")
            with mss.mss() as sct:
                screenshot = sct.grab(sct.monitors[0])
                temp_path = tempfile.mktemp(suffix='.png')
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=temp_path)
                
                with open(temp_path, 'rb') as f:
                    image_data = f.read()
                    screenshot_data = base64.b64encode(image_data).decode('utf-8')
                
                os.remove(temp_path)
                print(f"[Screenshot] ✅ mss succeeded: {len(image_data)} bytes raw, {len(screenshot_data)} chars base64")
                return screenshot_data
        except ImportError as e:
            print(f"[Screenshot] ⚠️ mss not available: {e}")
        except Exception as e:
            print(f"[Screenshot] ⚠️ mss failed: {e}")
            import traceback
            traceback.print_exc()
        
        # If both fail, return None
        print("[Screenshot] ❌ Both PIL and mss unavailable or failed for screen capture")
        return None
    except Exception as e:
        print(f"[Screenshot] ❌ Screen capture error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _fetch_context_for_gpt(include_screenshot: bool = True):
    """Helper to fetch modeling context + mesh analysis + scene analysis + screenshot via RPC."""
    modeling_context = None
    mesh_analysis = None
    scene_analysis = None
    screenshot_data = None
    if rpc is None:
        return modeling_context, mesh_analysis, scene_analysis, screenshot_data
    try:
        modeling_context = rpc.get_modeling_context()
        if modeling_context and modeling_context.get("active_object"):
            ao = modeling_context["active_object"]
            if ao.get("type") == "MESH":
                mesh_analysis = rpc.analyze_current_mesh()
        # Always fetch scene analysis for context
        try:
            scene_analysis = rpc.analyze_scene()
        except Exception as e:
            print(f"[Context] ⚠️ Error analyzing scene: {e}")
            scene_analysis = None
        
        # Capture screenshot if requested (use local screen capture, not Blender RPC)
        # Do this even if context fetching failed
        if include_screenshot:
            try:
                print("📸 Capturing viewport screenshot...")
                screenshot_data = _capture_screen_local()
                if screenshot_data:
                    print(f"✅ Screenshot captured: {len(screenshot_data)} bytes (base64)")
                else:
                    print("⚠️ Screenshot capture returned None - check [Screenshot] messages above for details")
            except Exception as e:
                print(f"⚠️ Screenshot capture exception: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"[Context] ⚠️ Error fetching context: {e}")
        import traceback
        traceback.print_exc()
        # Still try screenshot even if context fetch failed
        if include_screenshot and not screenshot_data:
            try:
                print("📸 Attempting screenshot capture after context error...")
                screenshot_data = _capture_screen_local()
                if screenshot_data:
                    print(f"✅ Screenshot captured after error: {len(screenshot_data)} bytes (base64)")
            except Exception as e2:
                print(f"⚠️ Screenshot capture also failed: {e2}")
    return modeling_context, mesh_analysis, scene_analysis, screenshot_data


def _resolve_local_sequence(command_parts):
    """Try to satisfy every command part via local rule handlers."""
    sequence = []
    for part in command_parts:
        cmd = try_io_rules(part)
        cmd = cmd or try_local_rules(part)
        if not cmd:
            return None
        if isinstance(cmd, list):
            sequence.extend(cmd)
        else:
            sequence.append(cmd)
    return sequence


def _react_observe(request: str) -> str:
    """Perform observation requests for ReAct loop with actionable error messages."""
    if rpc is None:
        return "RPC unavailable; cannot observe. You should EXECUTE actions instead."
    req = (request or "").strip().lower()
    try:
        if req in ("selected_objects", "selection", "selected"):
            ctx = rpc.get_modeling_context()
            selected = ctx.get("selected_objects", [])
            if not selected:
                return "No objects are currently selected. You may need to create objects first using EXECUTE actions."
            return json.dumps(selected, indent=2)
        if req in ("active_object", "active", "current_object"):
            ctx = rpc.get_modeling_context()
            active = ctx.get("active_object")
            if not active:
                return "No active object. You may need to create an object first using EXECUTE actions like mesh.primitive_cube_add."
            return json.dumps(active, indent=2)
        if req in ("mesh_analysis", "mesh", "geometry"):
            try:
                analysis = rpc.analyze_current_mesh()
                if analysis and analysis.get("error"):
                    return f"Mesh analysis error: {analysis.get('error')}. There may be no active mesh object. Create one first using EXECUTE actions."
                if analysis:
                    return json.dumps(analysis, indent=2)
                return "Mesh analysis returned empty result. There may be no active mesh object."
            except Exception as e:
                error_msg = str(e)
                if "TypeError" in error_msg or "dictionary key" in error_msg:
                    return f"Mesh analysis error (serialization issue): {error_msg}. Try executing an action to create or select a mesh object first."
                return f"Mesh analysis error: {error_msg}. There may be no active mesh object or the object is in an invalid state."
        if req in ("modifiers", "mods"):
            ctx = rpc.get_modeling_context()
            mods = ctx.get("modifiers", [])
            if not mods:
                return "No modifiers on active object. You can add modifiers using EXECUTE with object.modifier_add."
            return json.dumps(mods, indent=2)
        if req in ("scene", "scene_info", "scene_analysis"):
            analysis = rpc.analyze_scene()
            return json.dumps(analysis, indent=2)
        if req in ("parts", "categorized_parts", "object_parts"):
            analysis = rpc.analyze_scene()
            parts = analysis.get("categorized_parts", {})
            if not any(parts.values()):
                return "No categorized parts found. You may need to create objects first using EXECUTE actions."
            return json.dumps(parts, indent=2)
        return f"Unknown observation '{request}'. Try selected_objects, active_object, mesh_analysis, modifiers, scene, scene_analysis, parts. If you need to create objects, use EXECUTE actions instead."
    except Exception as e:
        return f"Observation error: {e}. Consider using EXECUTE actions to create or modify objects instead of observing."


def _validate_object_after_execution(target_object: str, mesh_analysis) -> str:
    """Post-execution validation for key objects. Returns validation message or empty string if OK."""
    if not target_object or not mesh_analysis or mesh_analysis.get("error"):
        return ""
    
    target_lower = target_object.lower()
    bounds = mesh_analysis.get("bounds")
    if not bounds:
        return ""
    
    size = bounds.get("size", (0, 0, 0))
    shape_class = bounds.get("shape_class", "")
    
    # Validate Echo Dot
    if "echo" in target_lower or "dot" in target_lower:
        # Should be roughly cylindrical (balanced or slightly tall)
        if shape_class == "flat":
            return "⚠️ Validation: Echo Dot appears too flat. Should be roughly cylindrical (height ≈ diameter)."
        # Check if dimensions are roughly equal (cylindrical)
        dims = sorted(size, reverse=True)
        if len(dims) >= 2:
            ratio = dims[0] / dims[1] if dims[1] > 0.001 else 0
            if ratio > 2.0:
                return f"⚠️ Validation: Echo Dot proportions seem off (largest dimension is {ratio:.1f}x the second). Should be roughly 1:1:1."
    
    # Validate smartphone
    if "phone" in target_lower or "smartphone" in target_lower:
        # Should be very flat (depth << width/height)
        if shape_class != "flat":
            return "⚠️ Validation: Smartphone should be very flat (thin depth). Current shape classification: " + shape_class
        # Check depth is much smaller than width/height
        width, depth, height = size[0], size[1], size[2]
        if depth > min(width, height) * 0.3:
            return f"⚠️ Validation: Smartphone depth ({depth:.2f}) is too large. Should be much thinner (depth << width/height)."
        # Check aspect ratio (should be roughly 16:9 or similar)
        if width > 0.001 and height > 0.001:
            aspect = max(width, height) / min(width, height)
            if aspect < 1.5:
                return f"⚠️ Validation: Smartphone aspect ratio ({aspect:.2f}) seems too square. Should be roughly 16:9 (≈1.78)."
    
    return ""  # Validation passed


# Global state memory for ReAct (persists between commands)
_REACT_STATE_MEMORY = {
    "last_summary": None,
    "last_target": None,
    "executed_ops": [],  # Track executed operations to avoid duplicates
}

# Global conversation history (persists across voice commands)
_CONVERSATION_HISTORY = []
_MAX_HISTORY_LENGTH = 20  # Keep last 20 messages


def add_to_conversation_history(role: str, content: str, images: list = None):
    """Add message to persistent conversation history."""
    global _CONVERSATION_HISTORY
    has_images = bool(images and len(images) > 0)
    _CONVERSATION_HISTORY.append({
        "role": role,
        "content": content,
        "images": images or [],
        "timestamp": datetime.now().isoformat()
    })
    # Keep last N messages
    if len(_CONVERSATION_HISTORY) > _MAX_HISTORY_LENGTH:
        _CONVERSATION_HISTORY.pop(0)
    
    # Debug logging
    img_info = f" (with {len(images)} image(s))" if has_images else ""
    print(f"💾 Saved to history: {role.upper()}{img_info} - {content[:50]}{'...' if len(content) > 50 else ''}")
    print(f"📚 Total history: {len(_CONVERSATION_HISTORY)} messages")


def get_conversation_context(max_messages: int = 10):
    """Get recent conversation history for GPT (last N messages)."""
    history = _CONVERSATION_HISTORY[-max_messages:] if _CONVERSATION_HISTORY else []
    if history:
        total_images = sum(1 for msg in history if msg.get("images"))
        print(f"📖 Loading {len(history)} messages from history ({total_images} with images)")
    return history


def clear_conversation_history():
    """Clear conversation history (useful for resetting context)."""
    global _CONVERSATION_HISTORY
    _CONVERSATION_HISTORY = []

def _react_execute(action_input: str, executed_commands: list) -> str:
    """Execute a command emitted by ReAct loop with error handling."""
    if not action_input:
        return "❌ Execute called with empty input. You must provide valid JSON. Example: {\"op\":\"mesh.primitive_cube_add\",\"kwargs\":{}}"
    
    cleaned = action_input.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("` \n")
    
    # Check if it looks like JSON
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        return f"❌ Execute input doesn't look like JSON. Expected JSON object or array. Got: {cleaned[:100]}. Example: {{\"op\":\"mesh.primitive_cube_add\",\"kwargs\":{{}}}}"
    
    try:
        cmd = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return f"❌ JSON parse error in execute: {e}. Input: {cleaned[:120]}. Make sure your JSON is valid. Example: {{\"op\":\"mesh.primitive_cube_add\",\"kwargs\":{{}}}}"
    
    def _exec_single(single_cmd):
        if not isinstance(single_cmd, dict) or "op" not in single_cmd:
            return "Execute input must be a JSON object with 'op' and optional 'kwargs'."
        
        op = single_cmd.get("op", "")
        kwargs = single_cmd.get("kwargs", {})
        
        # Operations that should be allowed to run multiple times (state-changing operations)
        # These operations modify selection/state and may need to be repeated
        allow_repeat_ops = {
            "object.select_all",
            "object.select_by_type",
            "object.join",
            "object.delete",
            "mesh.select_all",
            "mesh.select_mode",
            "object.mode_set",
        }
        
        # Check for duplicate operations (same op + same kwargs)
        # But allow certain operations to repeat
        if op not in allow_repeat_ops:
            cmd_signature = (op, json.dumps(kwargs, sort_keys=True))
            if cmd_signature in _REACT_STATE_MEMORY["executed_ops"]:
                return f"⚠️ Duplicate operation skipped: {op} (already executed with same parameters)"
        
        # Try to execute
        try:
            executed_commands.append(single_cmd)
            # Only track non-repeatable operations for duplicate detection
            if op not in allow_repeat_ops:
                cmd_signature = (op, json.dumps(kwargs, sort_keys=True))
                _REACT_STATE_MEMORY["executed_ops"].append(cmd_signature)
            result = send_to_blender(single_cmd)
            time.sleep(0.15)  # Give Blender a moment to process before next observation
            
            # Check if result indicates failure
            if result and isinstance(result, str):
                if "error" in result.lower() or "failed" in result.lower() or "❌" in result:
                    return f"❌ Operation failed: {op} - {result}. Try a different approach."
            
            return f"✅ Queued {op}: {result}"
        except Exception as e:
            error_msg = str(e)
            print(f"[ReAct] Execution error for {op}: {error_msg}")
            return f"❌ Operation error: {op} - {error_msg}. Try a different approach or check if prerequisites are met."
    
    if isinstance(cmd, list):
        observations = []
        for single in cmd:
            obs = _exec_single(single)
            observations.append(obs)
        return " | ".join(observations)
    else:
        return _exec_single(cmd)


def gpt_to_json_react(transcript: str, modeling_context=None, mesh_analysis=None, scene_analysis=None, target_object="", screenshot_data=None):
    """Use ReAct loop (Thought/Action/Observation) to execute complex commands."""
    if not ENABLE_GPT_FALLBACK:
        return None
    api_key = _get_openai_api_key()
    if not api_key:
        print("⚠️ ReAct: Missing OpenAI API key.")
        return None
    try:
        from openai import OpenAI
    except Exception as e:
        print(f"⚠️ ReAct: Failed to import OpenAI: {e}")
        return None
    
    # Get reference knowledge with step-by-step templates
    ref_knowledge = _get_reference_knowledge(target_object) if target_object else None
    
    system_prompt = [
        "You are a Blender automation agent that must follow the ReAct (Reason + Action) format.",
        "Respond ONLY with blocks containing:",
        "Thought: <reasoning>",
        "Action: <execute|observe|finish>",
        "Action Input: <JSON or observation request>",
        "",
        "Action rules:",
        "- execute: run a Blender operator via JSON, e.g. {\"op\":\"mesh.primitive_cube_add\",\"kwargs\":{}}.",
        "- observe: ask for data (selected_objects, active_object, mesh_analysis, modifiers, scene).",
        "- finish: task done, provide short summary.",
        "",
        "CRITICAL RULES:",
        "1. DO NOT observe more than 2 times in a row. After observing, you MUST execute an action.",
        "2. If an operation fails, try a DIFFERENT approach, don't just observe again.",
        "3. If you're stuck, simplify your approach or finish with what you have.",
        "4. Always provide valid JSON in Action Input for execute actions.",
        "5. If you need to create objects, use EXECUTE with mesh.primitive_*_add operators.",
        "",
        "Never output prose outside the required fields. Combine multiple Blender steps into sequential execute actions.",
        "",
        "IMPORTANT: If an operation fails, observe the current state ONCE, then try a different approach or finish.",
        "IMPORTANT: Avoid repeating the same operation with identical parameters (duplicates are skipped).",
    ]
    
    # Add state memory context
    if _REACT_STATE_MEMORY["last_summary"] and _REACT_STATE_MEMORY["last_target"] == target_object:
        system_prompt.append(f"\nPrevious command summary: {_REACT_STATE_MEMORY['last_summary']}")
        system_prompt.append("Continue building on the previous work.")
    
    if target_object:
        system_prompt.append(f"\nTarget Object: {target_object}")
        
        # Add detailed reference template if available
        if ref_knowledge:
            system_prompt.append(f"\n--- Reference Template: {target_object} ---")
            system_prompt.append(f"Description: {ref_knowledge.get('description', '')}")
            
            if ref_knowledge.get("typical_geometry"):
                geom = ref_knowledge["typical_geometry"]
                system_prompt.append(f"Shape: {geom.get('shape', '')}")
                system_prompt.append(f"Proportions: {geom.get('proportions', '')}")
                if geom.get("features"):
                    system_prompt.append("Key Features:")
                    for feature in geom["features"]:
                        system_prompt.append(f"  - {feature}")
            
            if ref_knowledge.get("step_by_step_template"):
                template = ref_knowledge["step_by_step_template"]
                system_prompt.append("\nStep-by-Step Build Template:")
                for step_name, step_info in template.items():
                    if isinstance(step_info, dict) and "op" in step_info:
                        system_prompt.append(f"  {step_name}: {step_info.get('description', '')}")
                        system_prompt.append(f"    Operation: {step_info['op']} with kwargs: {json.dumps(step_info.get('kwargs', {}))}")
                        if step_info.get("requires_edit_mode"):
                            system_prompt.append(f"    NOTE: Requires EDIT mode (enter edit mode first)")
                        if step_info.get("requires_object_mode"):
                            system_prompt.append(f"    NOTE: Requires OBJECT mode")
                    elif isinstance(step_info, dict) and "steps" in step_info:
                        system_prompt.append(f"  {step_name}: {step_info.get('description', '')}")
                        for substep in step_info["steps"]:
                            system_prompt.append(f"    - {substep}")
            
            if ref_knowledge.get("modeling_approach"):
                system_prompt.append("\nRecommended Modeling Approach:")
                for approach in ref_knowledge["modeling_approach"]:
                    system_prompt.append(f"  - {approach}")
    
    if modeling_context:
        system_prompt.append(f"\n--- Current Modeling Context ---")
        system_prompt.append(json.dumps(modeling_context, indent=2))
    
    if mesh_analysis and not mesh_analysis.get("error"):
        system_prompt.append(f"\n--- Current Mesh Analysis ---")
        ma = mesh_analysis
        system_prompt.append(f"Object: {ma.get('object_name')}")
        system_prompt.append(f"Vertices: {ma.get('vertex_count')}, Edges: {ma.get('edge_count')}, Faces: {ma.get('face_count')}")
        
        if ma.get("bounds"):
            bounds = ma["bounds"]
            size = bounds.get("size", (0, 0, 0))
            system_prompt.append(f"Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
            
            # Enhanced geometry hints
            if bounds.get("avg_thickness"):
                system_prompt.append(f"Average Thickness: {bounds['avg_thickness']:.3f}")
            if bounds.get("shape_class"):
                system_prompt.append(f"Shape Classification: {bounds['shape_class']} (flat/tall/balanced)")
            if bounds.get("aspect_ratios"):
                ratios = bounds["aspect_ratios"]
                system_prompt.append(f"Aspect Ratios: width/depth={ratios.get('width_depth', 0):.2f}, width/height={ratios.get('width_height', 0):.2f}")
        
        if ma.get("face_topology", {}).get("face_types"):
            face_types = ma["face_topology"]["face_types"]
            # Keys are strings from XML-RPC, convert back to int for comparison
            face_desc = ", ".join([f"{count} {int(v)}-gons" if int(v) > 4 else f"{count} {'triangles' if int(v) == 3 else 'quads'}" 
                                  for v, count in sorted(face_types.items(), key=lambda x: int(x[0]))])
            system_prompt.append(f"Face Types: {face_desc}")
    
    system_prompt_text = "\n".join(system_prompt)
    
    client = OpenAI(api_key=api_key)
    
    # Build initial conversation with history and optional screenshot
    conversation = []
    
    # Add conversation history (last 10 messages)
    history = get_conversation_context(max_messages=10)
    for hist_msg in history:
        msg_content = hist_msg["content"]
        # If history has images, include them
        if hist_msg.get("images") and hist_msg["images"]:
            content_list = [{"type": "text", "text": msg_content}]
            for img_data in hist_msg["images"]:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_data}"
                    }
                })
            conversation.append({
                "role": hist_msg["role"],
                "content": content_list
            })
        else:
            conversation.append({
                "role": hist_msg["role"],
                "content": msg_content
            })
    
    # Add current task with optional screenshot
    task_content = f"Task: {transcript}\nUse ReAct format. Begin reasoning."
    if screenshot_data:
        user_content = [
            {"type": "text", "text": task_content},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_data}"
                }
            }
        ]
        print(f"📸 Including viewport screenshot in ReAct GPT call ({len(screenshot_data)} bytes)")
    else:
        user_content = task_content
        print("⚠️ No screenshot available for ReAct GPT call")
    
    conversation.append({"role": "user", "content": user_content})
    
    executed_commands = []
    max_iterations = 25
    action_history = []  # Track action history for loop detection
    last_execute_iteration = -1  # Track when we last executed something
    
    for iteration in range(max_iterations):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt_text}, *conversation],
                temperature=0,
            )
        except Exception as e:
            print(f"⚠️ ReAct GPT error: {e}")
            return None
        
        output = (resp.choices[0].message.content or "").strip()
        conversation.append({"role": "assistant", "content": output})
        
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", output, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip().lower() if action_match else ""
        action_input = action_input_match.group(1).strip() if action_input_match else ""
        
        # Track action history for loop detection
        action_history.append(action)
        
        # Detect observation loops (3+ consecutive OBSERVE actions)
        if len(action_history) >= 3 and all(a == "observe" for a in action_history[-3:]):
            print(f"[ReAct] ⚠️ Detection: Observation loop detected (3+ consecutive OBSERVE actions)")
            observation = f"⚠️ You have been observing repeatedly. You must either EXECUTE an action or FINISH the task. Stop observing and take action. If you're stuck, try a simpler approach or finish with what you have."
            conversation.append({"role": "user", "content": f"Observation: {observation}"})
            continue
        
        # Detect execute-without-action loops (empty or repeated execute attempts)
        if action == "execute" and (not action_input or action_input.strip() == ""):
            print(f"[ReAct] ⚠️ Detection: Empty execute action")
            observation = "⚠️ Execute action called but no action input provided. You must provide valid JSON with 'op' and optional 'kwargs'. Example: {\"op\":\"mesh.primitive_cube_add\",\"kwargs\":{}}"
            conversation.append({"role": "user", "content": f"Observation: {observation}"})
            continue
        
        # Log reasoning (Thought content) to console
        if thought:
            print(f"[ReAct] Iteration {iteration+1} - Thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")
        print(f"[ReAct] Iteration {iteration+1} - Action: {action.upper()}"); sys.stdout.flush()
        
        if action == "execute":
            observation = _react_execute(action_input, executed_commands)
            if "✅" in observation:  # Success
                last_execute_iteration = iteration
        elif action == "observe":
            observation = _react_observe(action_input)
            # If we've been observing for 5+ iterations without executing, warn
            if iteration - last_execute_iteration >= 5 and last_execute_iteration >= 0:
                observation += " ⚠️ You've been observing for 5+ iterations. You must EXECUTE an action soon or FINISH."
        elif action == "finish":
            summary = action_input
            
            # Post-execution validation (before finishing)
            validation_failed = False
            if target_object and executed_commands:
                try:
                    # Get fresh mesh analysis for validation
                    fresh_analysis = rpc.analyze_current_mesh() if rpc else None
                    validation_msg = _validate_object_after_execution(target_object, fresh_analysis)
                    if validation_msg:
                        print(f"[ReAct] ⚠️ Validation warning: {validation_msg}")
                        # If validation fails, add it as an observation and continue loop
                        conversation.append({"role": "user", "content": f"Observation: {validation_msg}. Please fix this issue before finishing."})
                        validation_failed = True
                except Exception as e:
                    if VERBOSE_DEBUG:
                        print(f"[ReAct] Validation error: {e}")
            
            # If validation failed, continue the loop instead of finishing
            if validation_failed:
                print(f"[ReAct] Continuing due to validation issue...")
                continue
            
            # Validation passed or no validation needed - finish
            print(f"[ReAct] Finished: {summary}"); sys.stdout.flush()
            
            # Update state memory
            _REACT_STATE_MEMORY["last_summary"] = summary
            _REACT_STATE_MEMORY["last_target"] = target_object
            # Keep executed_ops for duplicate detection, but clear if starting new target
            if target_object != _REACT_STATE_MEMORY.get("last_target"):
                _REACT_STATE_MEMORY["executed_ops"] = []
            
            # Save to conversation history
            add_to_conversation_history("user", transcript, [screenshot_data] if screenshot_data else None)
            add_to_conversation_history("assistant", f"ReAct completed: {summary}")
            
            return {"commands": executed_commands, "summary": summary, "iterations": iteration+1}
        else:
            observation = f"Unknown action '{action}'. Expected execute/observe/finish."
        
        conversation.append({"role": "user", "content": f"Observation: {observation}"})
    
    print(f"⚠️ ReAct: Max iterations ({max_iterations}) reached without finish.")
    return {"commands": executed_commands, "summary": "Max iterations reached"}

# ------------------ Recorder ------------------

def record_until_silence():
    print("🎙️ Speak command…")
    frames = []
    spoken_sec = 0.0
    silence_sec = 0.0
    block_samples = int(BLOCK_SEC * SAMPLE_RATE)
    consecutive_silence_blocks = 0
    max_silence_blocks = int(SILENCE_HOLD / BLOCK_SEC)
    detected_speech = False
    
    # Get the current default input device (respects system settings when headphones are plugged in)
    # Query each time to pick up device changes
    default_input = None
    try:
        # sd.default.device[0] is the default input device index
        # Query each time to ensure we use the current system default
        default_input = sd.default.device[0]
        if VERBOSE_DEBUG and default_input is not None:
            device_info = sd.query_devices(default_input)
            _dbg(f"Using input device: {device_info.get('name', 'unknown')} (index {default_input})")
    except Exception:
        # If query fails, None will use the system default automatically
        if VERBOSE_DEBUG:
            _dbg("Using system default input device (device query failed)")
    
    with sd.InputStream(device=default_input, samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            block, _ = stream.read(block_samples)
            block = block.reshape(-1)
            frames.append(block)
            total_time = len(frames) * BLOCK_SEC
            voiced = _block_has_voice(block)
            
            if voiced:
                detected_speech = True
                spoken_sec += BLOCK_SEC
                silence_sec = 0.0
                consecutive_silence_blocks = 0
            else:
                silence_sec += BLOCK_SEC
                consecutive_silence_blocks += 1
            
            # Stop if we have enough consecutive silence blocks after sufficient speech
            if detected_speech and consecutive_silence_blocks >= max_silence_blocks and spoken_sec >= MIN_SPOKEN:
                if VERBOSE_DEBUG:
                    print(f"✅ Stopping: {spoken_sec:.2f}s speech, {silence_sec:.2f}s silence, {consecutive_silence_blocks} consecutive silent blocks")
                break
            
            if not detected_speech and total_time >= INITIAL_SILENCE_ABORT:
                if VERBOSE_DEBUG:
                    print(f"[DEBUG] Aborting capture after {total_time:.1f}s (no speech detected)")
                break

            # If no speech was detected yet, allow early abort after a short timeout
            if total_time > MAX_RECORD_SEC:
                print(f"⚠️ Recording timeout ({MAX_RECORD_SEC}s) - stopping anyway (had {spoken_sec:.2f}s speech)")
                break
            
            # Debug output every 2 seconds if verbose
            if VERBOSE_DEBUG and len(frames) % 10 == 0:  # Every 2 seconds (10 blocks * 0.2s)
                print(f"[DEBUG] Time: {total_time:.1f}s, Speech: {spoken_sec:.2f}s, Silence: {silence_sec:.2f}s, Silent blocks: {consecutive_silence_blocks}")

    if not detected_speech:
        if VERBOSE_DEBUG:
            print("🤫 Silence/noise only clip discarded")
        return None
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

# ================== Voice I/O (robust import) ==================

# Paths like C:\foo\bar.fbx or /some/path/file.obj
PATH_RE = re.compile(r'([A-Za-z]:\\[^"\n]+|\b[A-Za-z]:/[^"\n]+|\/(?:[^"\'\n]+))', re.IGNORECASE)

# “from my downloads folder”, “from downloads”, “in desktop”, “at documents”
KNOWN_FOLDER_RE = re.compile(
    r'\b(?:from|in|into|at)\s+(?:the\s+)?(?:my\s+)?'
    r'(desktop|download|downloads|documents|docs|pictures|models)'
    r'(?:\s+folder)?\b',
    re.IGNORECASE
)

# Literal filename with dot ext (strongest): "etherealGuardian.fbx"
FILENAME_DOT_RE = re.compile(
    r'\b([A-Za-z0-9_\-\. ]+?)\.(fbx|obj|gltf|glb|stl|ply|usd|dae|abc|svg)\b',
    re.IGNORECASE
)

# Spoken ext: "ethereal guardian fbx", "chair dot obj", "showroom glb file"
FILENAME_WORD_RE = re.compile(
    r'\b([A-Za-z0-9_\- ]+?)\s*(?:dot\s*)?'
    r'(fbx|obj|gltf|glb|stl|ply|usd|dae|abc|svg)(?:\s+file)?\b',
    re.IGNORECASE
)

# Names like: called ethereal guardian / named "ethereal guardian"
CALLED_NAME_RE = re.compile(
    r"\b(?:called|named|titled)\s+(?:the\s+)?(.+?)(?=\s+(?:it(?:'s)?|that|which|who|in|on|at|from|into|to|and|but)\b|$)",
    re.IGNORECASE
)
QUOTED_NAME_RE = re.compile(r'"([^"]{2,})\'|\'([^\']]{2,})\'')  # will be corrected below if needed
# Fix QUOTED_NAME_RE to a safe version (some editors mangle backslashes)
QUOTED_NAME_RE = re.compile(r'"([^"]{2,})"|\'([^\']{2,})\'')

# Base-only: "import ethereal guardian from downloads"
BASE_ONLY_RE = re.compile(
    r'\bimport|^port\b',  # we’ll handle extraction separately; this just flags intent
    re.IGNORECASE
)

SUPPORTED_EXTS = ["fbx","obj","gltf","glb","stl","ply","usd","dae","abc","svg"]

# Minimum similarity required when fuzzy-matching filenames in a folder
MIN_MATCH_SCORE = 0.72

# Simple import map (operator, keyword)
IMPORT_MAP = {
    "fbx":  ("import_scene.fbx",  "filepath"),
    "obj":  ("import_scene.obj",  "filepath"),
    "gltf": ("import_scene.gltf", "filepath"),
    "glb":  ("import_scene.gltf", "filepath"),
    "stl":  ("import_mesh.stl",   "filepath"),
    "ply":  ("import_mesh.ply",   "filepath"),
    "usd":  ("wm.usd_import",     "filepath"),
    "dae":  ("import_scene.dae",  "filepath"),
    "abc":  ("import_scene.abc",  "filepath"),
    "svg":  ("import_curve.svg",  "filepath"),
}

def _known_folder(name: str) -> str:
    n = (name or "").strip().lower()
    home = pathlib.Path.home()
    table = {
        "desktop":  home / "Desktop",
        "download": home / "Downloads",
        "downloads":home / "Downloads",
        "documents":home / "Documents",
        "docs":     home / "Documents",
        "pictures": home / "Pictures",
        "models":   home / "Documents" / "3D Models",
    }
    return str(table.get(n, ""))

def _normalize_path(raw: str) -> str:
    raw = raw.strip().strip('"').strip("'").replace(" / ", "/")
    return os.path.normpath(os.path.expandvars(os.path.expanduser(raw)))

def _norm_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def _score_names(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()

def _find_best_match(folder: str, spoken_base: str, ext: str):
    if not folder or not os.path.isdir(folder):
        return (None, 0.0)
    try:
        candidates = [f for f in os.listdir(folder) if f.lower().endswith("." + ext.lower())]
    except Exception:
        return (None, 0.0)
    if not candidates:
        return (None, 0.0)

    spoken_norm = _norm_name(spoken_base)
    best, best_score = None, 0.0

    exact_name = f"{spoken_base}.{ext}".lower()
    for f in candidates:
        if f.lower() == exact_name:
            return (os.path.join(folder, f), 1.0)

    for f in candidates:
        base_norm = _norm_name(os.path.splitext(f)[0])
        score = _score_names(base_norm, spoken_norm)
        if spoken_norm and (spoken_norm in base_norm or base_norm in spoken_norm):
            score += 0.15
        if score > best_score:
            best, best_score = f, score
    if best and best_score >= MIN_MATCH_SCORE:
        return (os.path.join(folder, best), best_score)
    _dbg(f"reject: best fuzzy match score {best_score:.2f} < {MIN_MATCH_SCORE}")
    return (None, best_score)

def _find_best_match_any_ext(folder: str, spoken_base: str, allowed_exts=SUPPORTED_EXTS):
    best_path, best_ext, best_score = None, None, 0.0
    for ext in allowed_exts:
        p, s = _find_best_match(folder, spoken_base, ext)
        if s > best_score:
            best_path, best_ext, best_score = p, ext, s
    if best_path and best_score >= MIN_MATCH_SCORE:
        return (best_path, best_ext, best_score)
    _dbg(f"reject-any: best fuzzy match score {best_score:.2f} < {MIN_MATCH_SCORE}")
    return (None, None, best_score)

def _extract_base_only(text: str) -> str:
    """
    Extracts the base name between 'import'/'port' and 'from/in/into/at'.
    e.g., 'import the ethereal guardian from my downloads folder'
          -> 'the ethereal guardian' -> 'ethereal guardian'
    """
    m = re.search(
        r'(?:\bimport\b|^port\b)\s+(?:the\s+)?(.+?)\s+(?:file\s+)?(?:from|in|into|at)\b',
        text, re.IGNORECASE)
    if not m:
        return ""
    base = m.group(1).strip()
    # Trim trailing filler like "folder"
    base = re.sub(r'\bfolder\b$', '', base, flags=re.IGNORECASE).strip()
    return base

def _pick_format_from_text(text: str, fallback_ext: str = "") -> str:
    t = text.lower()
    for ext in SUPPORTED_EXTS:
        if re.search(rf'\b{ext}\b', t):
            return ext
    return fallback_ext.lower()

def _io_cmd_import(utterance: str):
    t = utterance.strip()
    _dbg(f"import: raw='{t}'")

    # Normalize common filler to avoid leading fragments from contractions
    t = re.sub("[’]", "'", t)  # smart quotes → straight
    t = re.sub(r"\b(?:so\s+)?there(?:'|’)s\s+a\s+file\s+(?:called|named|titled)\b",
               "called", t, flags=re.IGNORECASE)
    t = re.sub(r"\bit's\b", "it is", t, flags=re.IGNORECASE)
    t = re.sub(r"\bits\b",  "it is", t, flags=re.IGNORECASE)
    _dbg(f"import: normalized='{t}'")

    # Treat transcripts that start with 'port ' as 'import '
    if t.lower().startswith("port "):
        t = "import " + t[5:]
        _dbg("import: normalized leading 'port' -> 'import'")

    # 0) Try to capture explicit names via quotes or called/named
    name = None
    q = QUOTED_NAME_RE.search(t)
    if q:
        name = (q.group(1) or q.group(2) or "").strip()
    else:
        m_called = CALLED_NAME_RE.search(t)
        if m_called:
            name = (m_called.group(1) or "").strip()
    if name:
        # "ethereal something" -> keep "ethereal"
        name = re.split(r"\bsomething\b", name, flags=re.IGNORECASE)[0].strip()
        # Remove contraction fragments like leading "s " from "there's"
        name = re.sub(r"^(?:s|t)\s+", "", name)
        # Stop at common delimiters (safety)
        name = re.split(r"\b(?:it|it is|that's|that|which|who|in|on|at|from|into|to|and|but)\b", name, 1)[0].strip()
        _dbg(f"NAME_CAPTURE -> '{name}'")

    # 1) Full explicit path
    m_path = PATH_RE.search(t)
    _dbg(f"PATH_RE -> {m_path.group(0) if m_path else None}")
    if m_path:
        p = _normalize_path(m_path.group(0))
        ext = pathlib.Path(p).suffix.lower().lstrip(".")
        fmt = _pick_format_from_text(t, ext)
        _dbg(f"explicit path p='{p}', ext='{ext}', fmt='{fmt}'")
        if fmt in IMPORT_MAP:
            op, file_kw = IMPORT_MAP[fmt]
            return {"op": op, "kwargs": {file_kw: p}}

    # 2) Known folder
    folder_match = KNOWN_FOLDER_RE.search(t)
    _dbg(f"KNOWN_FOLDER_RE -> {folder_match.group(0) if folder_match else None}")
    folder = _known_folder(folder_match.group(1)) if folder_match else ""

    # If we have a captured name and folder, try that first across any ext
    if folder and name and len(_norm_name(name)) >= 3:
        final_path, best_ext, score = _find_best_match_any_ext(folder, name, SUPPORTED_EXTS)
        _dbg(f"folder+captured-name -> chosen='{final_path}' ext='{best_ext}' score={score:.2f}")
        if final_path and best_ext in IMPORT_MAP:
            op, file_kw = IMPORT_MAP[best_ext]
            return {"op": op, "kwargs": {file_kw: _normalize_path(final_path)}}

    # 3) Filename with literal dot
    m_dot = FILENAME_DOT_RE.search(t)
    if m_dot:
        spoken_base = m_dot.group(1).strip()
        ext = m_dot.group(2).lower()
        _dbg(f"FILENAME_DOT_RE -> base='{spoken_base}' ext='{ext}'")
        if folder:
            final_path, score = _find_best_match(folder, spoken_base, ext)
            _dbg(f"folder+file (dot ext) -> chosen='{final_path}' score={score:.2f}")
            if final_path and ext in IMPORT_MAP:
                op, file_kw = IMPORT_MAP[ext]
                return {"op": op, "kwargs": {file_kw: _normalize_path(final_path)}}

    # 4) Filename with spoken ext (e.g., "ethereal guardian fbx")
    m_word = FILENAME_WORD_RE.search(t)
    if m_word:
        spoken_base = m_word.group(1).strip()
        ext = m_word.group(2).lower()
        # Filter out obviously garbage bases (common filler)
        garbage_patterns = [
            r"^but it is an$", r"^it is an$", r"^it is a$", r"^its an$", r"^it's an$", r"^an$",
            r"^s an$", r"^s a$", r"^a file$", r"^file$", r"^the file$", r"^an fbx$", r"^fbx$"
        ]
        garbage = any(re.match(pat, spoken_base.lower()) for pat in garbage_patterns)
        if garbage or len(_norm_name(spoken_base)) < 3:
            _dbg(f"FILENAME_WORD_RE rejected base='{spoken_base}' (garbage or too short)")
        else:
            _dbg(f"FILENAME_WORD_RE -> base='{spoken_base}' ext='{ext}'")
            if folder:
                final_path, score = _find_best_match(folder, spoken_base, ext)
                _dbg(f"folder+file (word ext) -> chosen='{final_path}' score={score:.2f}")
                if final_path and ext in IMPORT_MAP:
                    op, file_kw = IMPORT_MAP[ext]
                    return {"op": op, "kwargs": {file_kw: _normalize_path(final_path)}}

    # 5) Base-only → best across ALL extensions
    if folder:
        base_only = _extract_base_only(t) or (name or "")
        _dbg(f"BASE_ONLY -> '{base_only}'")
        if base_only and len(_norm_name(base_only)) >= 3:
            final_path, best_ext, score = _find_best_match_any_ext(folder, base_only, SUPPORTED_EXTS)
            _dbg(f"folder+file (any ext) -> chosen='{final_path}' ext='{best_ext}' score={score:.2f}")
            if final_path and best_ext in IMPORT_MAP:
                op, file_kw = IMPORT_MAP[best_ext]
                return {"op": op, "kwargs": {file_kw: _normalize_path(final_path)}}

    _dbg("import: no match")
    return None

def try_io_rules(text: str):
    """High-priority import/export style commands before local geometry rules."""
    if not text:
        return None
    # Import intent keywords
    if re.search(r"\b(import|port)\b", text, re.IGNORECASE) or \
       re.search(r"\bcalled\b|\bnamed\b|\btitled\b|\"|'", text):
        return _io_cmd_import(text)
    # (Future) export intents could be added here.
    return None

# ================== /Voice I/O (robust import) ==================

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
    """Extract quantity from text, handling both digits and word numbers."""
    # Try digits first
    m = re.search(r"\b(add|spawn|create)\s+(\d+)\b", text.lower())
    if m:
        try:
            return int(m.group(2))
        except Exception:
            pass
    
    # Try word numbers
    words = text.lower().split()
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20
    }
    
    for i, word in enumerate(words):
        if word in ("add", "spawn", "create") and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in word_map:
                return word_map[next_word]
    
    return 1

def _split_multiple_commands(text: str):
    """Split text into multiple commands on 'and', 'then', commas, etc."""
    # Split on common conjunctions
    separators = [r"\s+and\s+", r"\s+then\s+", r",\s+", r"\.\s+"]
    parts = [text]
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(re.split(sep, part, flags=re.IGNORECASE))
        parts = [p.strip() for p in new_parts if p.strip()]
        if len(parts) > 1:
            break  # Use first separator that splits
    return parts if len(parts) > 1 else [text]


def try_local_rules(text: str):
    t = (text or "").strip()
    if not t:
        return None
    tl = t.lower()
    words = tl.split()

    # Deselect commands - check BEFORE select all (more specific)
    deselect_patterns = [
        "deselect all", "clear selection", "deselect everything",
        "unselect all", "clear all selection"
    ]
    if _match_any_phrase(tl, deselect_patterns):
        return {"op": "object.select_all", "kwargs": {"action": "DESELECT"}}
    
    # Select all - check after deselect
    if _match_any_phrase(tl, ["select all"]):
        return {"op": "object.select_all", "kwargs": {"action": "SELECT"}}
    if _match_any_phrase(tl, ["frame selected", "focus selected", "zoom to selected"]):
        return {"op": "view3d.view_selected", "kwargs": {}}

    # Delete commands - more natural language variants
    delete_patterns = [
        "delete selected", "delete object", "remove object", "erase object",
        "delete all", "delete everything", "remove all", "erase all",
        "delete all objects", "delete all selected", "remove all objects",
        "delete all of the objects", "delete all the objects", "remove all selected"
    ]
    if _match_any_phrase(tl, delete_patterns):
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
    m = re.search(r"(scale|resize)\s+(?:to\s+)?(\d+(?:\.\d+)?)", tl)
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
        # Handle multiple object types in one command (e.g., "add 10 cubes and 10 spheres")
        commands = []
        
        def make_add_cmd(template, qty=1):
            op, param_map = template
            base_kwargs = _extract_common_kwargs(words)
            kwargs = dict(base_kwargs)
            if "size" in kwargs:
                for k, alias in param_map.items():
                    if alias == "size":
                        kwargs[k] = kwargs.pop("size")
                        break
            return [{"op": op, "kwargs": kwargs} for _ in range(qty)]
        
        # Check for multiple object types with quantities
        # Pattern: "add X cubes and Y spheres"
        multi_pattern = re.search(r"add\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(\w+)(?:\s+and\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(\w+))?", tl)
        if multi_pattern:
            # Extract first object type and quantity
            qty1_word = multi_pattern.group(1)
            obj1 = multi_pattern.group(2)
            qty1 = _word_to_number(qty1_word) if _word_to_number(qty1_word) is not None else int(qty1_word) if qty1_word.isdigit() else 1
            
            # Helper to match object names (handles plurals)
            def match_obj_name(obj_word, key):
                obj_lower = obj_word.lower()
                key_lower = key.lower()
                if obj_lower in key_lower or key_lower in obj_lower:
                    return True
                # Handle plurals
                if obj_lower.endswith("s") and obj_lower[:-1] in key_lower:
                    return True
                if not obj_lower.endswith("s") and (obj_lower + "s") in key_lower:
                    return True
                # Handle "sphere" vs "uv sphere"
                if "sphere" in obj_lower and "sphere" in key_lower:
                    return True
                return False
            
            # Check all primitives for matches
            all_prims = {**MESH_PRIMS, **CURVE_PRIMS, **LIGHTS, **CAMERA}
            for key, template in all_prims.items():
                if match_obj_name(obj1, key):
                    commands.extend(make_add_cmd(template, qty1))
                    break
            
            # Extract second object type and quantity if present
            if multi_pattern.group(3) and multi_pattern.group(4):
                qty2_word = multi_pattern.group(3)
                obj2 = multi_pattern.group(4)
                qty2 = _word_to_number(qty2_word) if _word_to_number(qty2_word) is not None else int(qty2_word) if qty2_word.isdigit() else 1
                
                for key, template in all_prims.items():
                    if match_obj_name(obj2, key):
                        commands.extend(make_add_cmd(template, qty2))
                        break
            
            if commands:
                return commands if len(commands) > 1 else commands[0]
        
        # Single object type (original logic)
        qty = _maybe_quantity(tl)

        def make_add_cmd_single(template):
            op, param_map = template
            base_kwargs = _extract_common_kwargs(words)
            kwargs = dict(base_kwargs)
            if "size" in kwargs:
                for k, alias in param_map.items():
                    if alias == "size":
                        kwargs[k] = kwargs.pop("size")
                        break
            return {"op": op, "kwargs": kwargs}

        # Helper to check if key matches (handles plurals)
        def key_matches(key, text):
            if key in text:
                return True
            # Check plural/singular variants
            if key.endswith("s") and key[:-1] in text:
                return True
            if not key.endswith("s") and (key + "s") in text:
                return True
            # Handle multi-word keys (e.g., "uv sphere" matches "uv spheres")
            if " " in key:
                base = key.rsplit(" ", 1)[0]
                if base in text and ("sphere" in text or "spheres" in text):
                    return True
            return False

        for key, template in MESH_PRIMS.items():
            if key_matches(key, tl):
                return [make_add_cmd_single(template) for _ in range(qty)] if qty > 1 else make_add_cmd_single(template)
        for key, template in CURVE_PRIMS.items():
            if key_matches(key, tl) or (key == "text" and "text" in tl):
                return [make_add_cmd_single(template) for _ in range(qty)] if qty > 1 else make_add_cmd_single(template)
        for key, template in LIGHTS.items():
            if key_matches(key, tl):
                return [make_add_cmd_single(template) for _ in range(qty)] if qty > 1 else make_add_cmd_single(template)
        for key, template in CAMERA.items():
            if key_matches(key, tl):
                return [make_add_cmd_single(template) for _ in range(qty)] if qty > 1 else make_add_cmd_single(template)

    return None

# ------------------ GPT Fallback ------------------

# Cache for generated reference knowledge
_REFERENCE_CACHE = {}

def _get_hardcoded_reference(target_lower: str):
    """Fallback to hardcoded templates for common objects."""
    knowledge = {
        "echo dot": {
            "description": "Amazon Echo Dot - a small cylindrical smart speaker",
            "typical_geometry": {
                "shape": "cylinder with rounded top and bottom",
                "proportions": "roughly 1:1:1 (height ≈ diameter), approximately 3-4 units in Blender scale",
                "features": [
                    "Circular top face with button/light array (4 small circular indentations)",
                    "Cylindrical body with fabric mesh texture",
                    "Rounded edges (beveled)",
                    "Bottom with rubber base",
                    "Ring detail around the top edge",
                ],
            },
            "modeling_approach": [
                "Step 1: Create base cylinder (mesh.primitive_cylinder_add) with radius ~1.5-2.0, depth ~1.5-2.0",
                "Step 2: Enter EDIT mode and add loop cuts near top and bottom for detail",
                "Step 3: Bevel top and bottom edges for rounded appearance",
                "Step 4: On top face, create 4 small circular indentations (inset faces, then extrude inward)",
                "Step 5: Add subdivision surface modifier for smoothness",
                "Step 6: Scale/position as needed",
            ],
            "step_by_step_template": {
                "base_primitive": {
                    "op": "mesh.primitive_cylinder_add",
                    "kwargs": {"radius": 1.8, "depth": 1.8, "vertices": 32},
                    "description": "Create main cylindrical body"
                },
                "loop_cuts": {
                    "op": "mesh.loopcut",
                    "kwargs": {"number_cuts": 2},
                    "description": "Add detail loops for beveling",
                    "requires_edit_mode": True
                },
                "bevel_edges": {
                    "op": "mesh.bevel",
                    "kwargs": {"offset": 0.05, "segments": 3},
                    "description": "Round the top and bottom edges",
                    "requires_edit_mode": True
                },
                "top_details": {
                    "description": "Create 4 button indentations on top face",
                    "steps": [
                        "Select top face",
                        "Inset faces (mesh.inset_faces) with small amount",
                        "Extrude region (mesh.extrude_region) inward slightly",
                        "Repeat for 4 positions in square pattern"
                    ]
                },
                "subdivision": {
                    "op": "object.modifier_add",
                    "kwargs": {"type": "SUBSURF"},
                    "description": "Add smoothness",
                    "requires_object_mode": True
                }
            },
            "common_operations": [
                "mesh.primitive_cylinder_add",
                "object.modifier_add(type='SUBSURF')",
                "mesh.bevel",
                "mesh.loopcut",
                "mesh.inset_faces",
                "mesh.extrude_region",
                "transform.resize",
            ],
        },
        "phone": {
            "description": "Smartphone - rectangular device with rounded corners",
            "typical_geometry": {
                "shape": "rectangular prism with rounded corners",
                "proportions": "roughly 16:9 aspect ratio (width:height), very thin depth (~0.1-0.15 units)",
                "features": [
                    "Large flat screen face (front)",
                    "Rounded corners (fillet/bevel)",
                    "Thin bezel around screen (inset on front face)",
                    "Camera bump on back (small extruded rectangle)",
                    "Button cutouts on sides",
                    "Very thin profile (depth << width/height)",
                ],
            },
            "modeling_approach": [
                "Step 1: Create base cube (mesh.primitive_cube_add)",
                "Step 2: Scale to phone proportions: width ~6-8, height ~10-14, depth ~0.1-0.15 (make it very thin)",
                "Step 3: Enter EDIT mode and bevel all edges for rounded corners",
                "Step 4: Select front face and inset for screen bezel",
                "Step 5: Select back face, inset slightly, then extrude outward for camera bump",
                "Step 6: Add small details (button cutouts on sides if needed)",
            ],
            "step_by_step_template": {
                "base_primitive": {
                    "op": "mesh.primitive_cube_add",
                    "kwargs": {"size": 1.0},
                    "description": "Create base cube"
                },
                "scale_to_proportions": {
                    "op": "transform.resize",
                    "kwargs": {"value": (7.0, 12.0, 0.12)},
                    "description": "Scale to phone proportions (width, height, very thin depth)",
                    "note": "Depth should be much smaller than width/height (phone is very thin)"
                },
                "bevel_edges": {
                    "op": "mesh.bevel",
                    "kwargs": {"offset": 0.2, "segments": 4},
                    "description": "Round all edges for phone-like appearance",
                    "requires_edit_mode": True
                },
                "screen_bezel": {
                    "description": "Create screen bezel on front face",
                    "steps": [
                        "Select front face (largest face)",
                        "Inset faces (mesh.inset_faces) with small amount (~0.1-0.2)",
                        "This creates the bezel around the screen"
                    ]
                },
                "camera_bump": {
                    "description": "Create camera bump on back",
                    "steps": [
                        "Select back face (opposite of front)",
                        "Inset faces slightly",
                        "Extrude region (mesh.extrude_region) outward by small amount (~0.05-0.1)",
                        "This creates the camera bump"
                    ]
                }
            },
            "common_operations": [
                "mesh.primitive_cube_add",
                "mesh.inset_faces",
                "mesh.extrude_faces",
                "mesh.extrude_region",
                "mesh.bevel",
                "transform.resize",
            ],
        },
        "sphere": {
            "description": "Sphere - perfectly round object",
            "typical_geometry": {
                "shape": "perfect sphere",
                "features": ["Uniform curvature", "No edges"],
            },
            "modeling_approach": [
                "Use UV sphere or icosphere primitive",
                "Subdivision surface for smoothness",
            ],
            "common_operations": [
                "mesh.primitive_uv_sphere_add",
                "mesh.primitive_ico_sphere_add",
                "object.modifier_add(type='SUBSURF')",
            ],
        },
    }
    
    # Fuzzy match target object
    for key, info in knowledge.items():
        if key in target_lower or target_lower in key:
            return info
    
    return None

def _get_reference_knowledge(target_object: str, use_gpt=True):
    """Get reference knowledge about any 3D object - dynamically generated via GPT or cached."""
    if not target_object:
        return None
    
    target_lower = target_object.lower()
    
    # Check cache first
    if target_lower in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[target_lower]
    
    # Try hardcoded templates first (for common objects - keep existing ones)
    hardcoded = _get_hardcoded_reference(target_lower)
    if hardcoded:
        _REFERENCE_CACHE[target_lower] = hardcoded
        return hardcoded
    
    # Generate dynamically using GPT if enabled
    if use_gpt and ENABLE_GPT_FALLBACK:
        api_key = _get_openai_api_key()
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                prompt = f"""You are a 3D modeling expert specializing in Blender. Generate comprehensive reference knowledge for modeling a "{target_object}" in Blender.

Provide a JSON response with this exact structure:
{{
    "description": "Brief 1-2 sentence description of the object",
    "typical_geometry": {{
        "shape": "Overall shape description (e.g., 'cylindrical', 'rectangular prism', 'organic curved')",
        "proportions": "Typical size ratios and dimensions (e.g., 'roughly 1:1:1 (height ≈ diameter), approximately 3-4 units in Blender scale')",
        "features": ["Key visual feature 1", "Key visual feature 2", "Key visual feature 3"]
    }},
    "modeling_approach": [
        "Step 1: Create base primitive (specify which and dimensions)",
        "Step 2: Enter EDIT mode and...",
        "Step 3: Add details by...",
        "Step 4: Apply modifiers for...",
        "Step 5: Final adjustments..."
    ],
    "step_by_step_template": {{
        "base_primitive": {{
            "op": "mesh.primitive_cube_add",
            "kwargs": {{"size": 1.0}},
            "description": "Create base shape"
        }},
        "detail_step_1": {{
            "op": "mesh.bevel",
            "kwargs": {{"offset": 0.1, "segments": 3}},
            "description": "Round edges",
            "requires_edit_mode": true
        }}
    }},
    "common_operations": ["mesh.primitive_cube_add", "mesh.bevel", "transform.resize"]
}}

Guidelines:
- Be specific about Blender operations and operators
- Include realistic proportions and dimensions
- Focus on key visual features that make the object recognizable
- Provide step-by-step workflow that's actually achievable in Blender
- Use proper Blender operator names (mesh.*, object.*, transform.*)
- Consider both OBJECT and EDIT modes where appropriate
"""
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a 3D modeling expert. Return only valid JSON matching the exact structure requested. Do not include markdown code fences."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                # Remove any markdown code fences if present
                if content.startswith("```"):
                    content = content.split("```", 2)[-1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
                
                generated = json.loads(content)
                
                # Validate structure
                if isinstance(generated, dict) and "description" in generated:
                    _REFERENCE_CACHE[target_lower] = generated
                    print(f"[ReAct] Generated reference template for: {target_object}")
                    return generated
                else:
                    if VERBOSE_DEBUG:
                        print(f"[DEBUG] Generated reference has invalid structure")
                    return None
                
            except json.JSONDecodeError as e:
                if VERBOSE_DEBUG:
                    print(f"[DEBUG] JSON parse error for {target_object}: {e}")
                    print(f"[DEBUG] Response was: {content[:200]}")
                return None
            except Exception as e:
                if VERBOSE_DEBUG:
                    print(f"[DEBUG] Failed to generate reference for {target_object}: {e}")
                return None
    
    return None

def _get_mode_aware_operations(current_mode: str, has_mesh: bool):
    """Get recommended operations based on current Blender mode (Phase 3)."""
    operations = {
        "OBJECT": [
            "object.select_all",
            "object.delete",
            "object.duplicate_move",
            "transform.translate",
            "transform.rotate",
            "transform.resize",
            "object.modifier_add",
            "mesh.primitive_*_add",
        ],
        "EDIT": [
            "mesh.select_all",
            "mesh.delete",
            "mesh.extrude_region",
            "mesh.inset_faces",
            "mesh.bevel",
            "mesh.loopcut",
            "mesh.subdivide",
            "mesh.merge",
            "mesh.separate",
            "mesh.select_by_type",
        ],
        "SCULPT": [
            "sculpt.sculptmode_toggle",
            "paint.brush_select",
            # Sculpt operations are typically brush-based, not operator-based
        ],
    }
    
    mode_ops = operations.get(current_mode, operations["OBJECT"])
    
    if not has_mesh:
        # Filter out mesh-specific operations if no mesh
        mode_ops = [op for op in mode_ops if not op.startswith("mesh.")]
    
    return mode_ops

def _get_best_practices_guidance(modeling_context, mesh_analysis, target_object):
    """Generate best practices guidance based on context (Phase 3)."""
    practices = []
    
    if modeling_context:
        mode = modeling_context.get("current_mode", "OBJECT")
        
        if mode == "EDIT":
            practices.append("In EDIT mode: Use mesh operations (extrude, bevel, loopcut) for topology changes")
            practices.append("Maintain quad topology when possible for better subdivision")
            practices.append("Use loop cuts to add detail without changing overall shape")
        
        if modeling_context.get("modifiers"):
            mods = modeling_context["modifiers"]
            if any(m.get("type") == "SUBSURF" for m in mods):
                practices.append("Subdivision Surface active: Ensure clean topology with quads")
                practices.append("Avoid n-gons (faces with >4 vertices) when using subdivision")
    
    if mesh_analysis and not mesh_analysis.get("error"):
        ma = mesh_analysis
        face_types = ma.get("face_topology", {}).get("face_types", {})
        
        n_gon_count = sum(count for v, count in face_types.items() if v > 4)
        if n_gon_count > 0:
            practices.append(f"Warning: {n_gon_count} n-gon(s) detected - consider converting to quads/triangles")
        
        if face_types.get(3, 0) > face_types.get(4, 0) * 2:
            practices.append("Many triangles detected - consider retopology for better subdivision")
    
    if target_object:
        ref_knowledge = _get_reference_knowledge(target_object)
        if ref_knowledge:
            practices.append(f"Reference: {ref_knowledge.get('description', '')}")
            if ref_knowledge.get("modeling_approach"):
                practices.append("Recommended approach: " + "; ".join(ref_knowledge["modeling_approach"][:3]))
    
    return practices

def gpt_to_json(transcript: str, modeling_context=None, mesh_analysis=None, target_object="", screenshot_data=None):
    """
    Convert natural language transcript to Blender operation JSON.
    
    Args:
        transcript: Voice command text
        modeling_context: Current modeling context (from get_modeling_context RPC)
        mesh_analysis: Current mesh analysis (from analyze_current_mesh RPC)
        target_object: Modeling target provided by Blender preferences
        screenshot_data: Base64-encoded screenshot image (optional)
    """
    # allow disabling while debugging
    if not ENABLE_GPT_FALLBACK:
        return None

    # Get API key (may have been updated via RPC)
    api_key = _get_openai_api_key()
    if not api_key:
        # Only print warning once per session to avoid spam
        if not hasattr(gpt_to_json, "_warned"):
            print("⚠️ OpenAI API key not set; skipping GPT fallback.")
            gpt_to_json._warned = True
        return None
    
    # Validate key format
    api_key = api_key.strip()
    if not api_key.startswith("sk-") and not api_key.startswith("sk-proj-"):
        if not hasattr(gpt_to_json, "_format_warned"):
            print(f"⚠️ OpenAI API key format looks invalid (got: {api_key[:10]}...). Keys should start with 'sk-' or 'sk-proj-'")
            gpt_to_json._format_warned = True
    
    try:
        from openai import OpenAI
    except Exception:
        print("⚠️ openai package not available; pip install openai")
        return None

    # Build system prompt with context if available (Phase 2 + Phase 3)
    system_parts = [
        "You are a Blender automation agent with expert knowledge of 3D modeling workflows.",
        "Output ONLY raw JSON (no prose, no code fences).",
        "Each command must be of the form: {\"op\":\"<module.op>\",\"kwargs\":{}}.",
        "If multiple steps are implied, output a JSON array of such dicts.",
        "Prefer creative operators (object/mesh/curve/transform/material/node/render).",
        "Never use file/quit/addon/script/image.save operators.",
    ]
    
    # Phase 3: Get reference knowledge and best practices
    ref_knowledge = _get_reference_knowledge(target_object) if target_object else None
    current_mode = modeling_context.get("current_mode", "OBJECT") if modeling_context else "OBJECT"
    has_mesh = mesh_analysis is not None and not mesh_analysis.get("error")
    mode_ops = _get_mode_aware_operations(current_mode, has_mesh)
    best_practices = _get_best_practices_guidance(modeling_context, mesh_analysis, target_object)
    
    # Add context information if available
    if modeling_context or mesh_analysis or target_object:
        system_parts.append("\n--- Current Scene Context ---")
        
        if target_object:
            system_parts.append(f"Target Object: {target_object} (this is what we're building/modifying)")
            if ref_knowledge:
                system_parts.append(f"Reference Knowledge: {ref_knowledge.get('description', '')}")
        
        if modeling_context and not modeling_context.get("error"):
            ctx = modeling_context
            system_parts.append(f"Current Mode: {ctx.get('current_mode', 'OBJECT')}")
            
            if ctx.get("active_object"):
                ao = ctx["active_object"]
                system_parts.append(f"Active Object: {ao.get('name')} (type: {ao.get('type')}, mode: {ao.get('mode')})")
            
            if ctx.get("selected_objects"):
                sel_count = len(ctx["selected_objects"])
                system_parts.append(f"Selected Objects: {sel_count} object(s)")
                for obj in ctx["selected_objects"][:3]:  # Show first 3
                    system_parts.append(f"  - {obj.get('name')} ({obj.get('type')})")
            
            if ctx.get("modifiers"):
                mods = ctx["modifiers"]
                system_parts.append(f"Modifiers on active object: {len(mods)}")
                for mod in mods[:5]:  # Show first 5
                    system_parts.append(f"  - {mod.get('name')} ({mod.get('type')})")
        
        if mesh_analysis and not mesh_analysis.get("error"):
            ma = mesh_analysis
            system_parts.append(f"\n--- Mesh Analysis ---")
            system_parts.append(f"Object: {ma.get('object_name')}")
            system_parts.append(f"Vertices: {ma.get('vertex_count')}, Edges: {ma.get('edge_count')}, Faces: {ma.get('face_count')}")
            
            if ma.get("bounds"):
                bounds = ma["bounds"]
                size = bounds.get("size", (0, 0, 0))
                system_parts.append(f"Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
            
            if ma.get("face_topology", {}).get("face_types"):
                face_types = ma["face_topology"]["face_types"]
                face_desc = ", ".join([f"{count} {v}-gons" if v > 4 else f"{count} {'triangles' if v == 3 else 'quads'}" 
                                      for v, count in sorted(face_types.items())])
                system_parts.append(f"Face Types: {face_desc}")
            
            if ma.get("selection"):
                sel = ma["selection"]
                if sel.get("vertices") or sel.get("edges") or sel.get("faces"):
                    sel_desc = []
                    if sel.get("vertices"):
                        sel_desc.append(f"{len(sel['vertices'])} vertices")
                    if sel.get("edges"):
                        sel_desc.append(f"{len(sel['edges'])} edges")
                    if sel.get("faces"):
                        sel_desc.append(f"{len(sel['faces'])} faces")
                    if sel_desc:
                        system_parts.append(f"Selected: {', '.join(sel_desc)}")
            
            if ma.get("edge_loops"):
                loops = ma["edge_loops"]
                system_parts.append(f"Edge Loops: {len(loops)} potential loop(s) detected")
        
        # Phase 3: Add mode-aware operation suggestions
        system_parts.append(f"\n--- Recommended Operations (Mode: {current_mode}) ---")
        system_parts.append("Available operations for current mode:")
        for op in mode_ops[:10]:  # Show first 10
            system_parts.append(f"  - {op}")
        
        # Phase 3: Add reference knowledge details
        if ref_knowledge:
            system_parts.append(f"\n--- Reference Knowledge: {target_object} ---")
            if ref_knowledge.get("typical_geometry"):
                geom = ref_knowledge["typical_geometry"]
                system_parts.append(f"Typical Shape: {geom.get('shape', '')}")
                if geom.get("proportions"):
                    system_parts.append(f"Proportions: {geom.get('proportions', '')}")
                if geom.get("features"):
                    system_parts.append("Key Features:")
                    for feature in geom["features"][:5]:
                        system_parts.append(f"  - {feature}")
            
            if ref_knowledge.get("common_operations"):
                system_parts.append("Common Operations for this object type:")
                for op in ref_knowledge["common_operations"][:8]:
                    system_parts.append(f"  - {op}")
        
        # Phase 3: Add best practices guidance
        if best_practices:
            system_parts.append(f"\n--- Best Practices & Guidance ---")
            for practice in best_practices:
                system_parts.append(f"• {practice}")
        
        # Phase 3: Operation selection guidance
        system_parts.append(f"\n--- Operation Selection Guidelines ---")
        system_parts.append("Choose operations based on:")
        system_parts.append("1. Current mode and available operations")
        system_parts.append("2. Desired outcome and best practices")
        system_parts.append("3. Current mesh topology and geometry state")
        if ref_knowledge:
            system_parts.append("4. Reference knowledge for target object type")
        system_parts.append("5. Maintain clean topology (prefer quads, avoid n-gons when using subdivision)")
        if current_mode == "EDIT":
            system_parts.append("6. In EDIT mode: Use mesh operations for topology changes")
        system_parts.append("7. Chain multiple operations if needed to achieve the desired effect")
        
        # Material operations guidance
        system_parts.append(f"\n--- Material Operations ---")
        system_parts.append("CRITICAL: Materials must be CREATED with proper name AND properties set. Use execute RPC with Python code.")
        system_parts.append("IMPORTANT: Each execute call runs in separate scope. Combine ALL operations in ONE execute call.")
        system_parts.append("MANDATORY: When creating glass material, you MUST:")
        system_parts.append("  1. Name it exactly 'Glass' (not 'Material' or anything else)")
        system_parts.append("  2. Enable use_nodes = True")
        system_parts.append("  3. Get the Principled BSDF node")
        system_parts.append("  4. Set Transmission = 1.0 (fully transparent)")
        system_parts.append("  5. Set Roughness = 0.0 (smooth)")
        system_parts.append("  6. Set IOR = 1.45 (glass index of refraction)")
        system_parts.append("  7. Assign to object(s)")
        system_parts.append("")
        system_parts.append("EXACT CODE for glass material (copy this pattern):")
        system_parts.append("import bpy")
        system_parts.append("mat = bpy.data.materials.new(name='Glass')")
        system_parts.append("mat.use_nodes = True")
        system_parts.append("bsdf = mat.node_tree.nodes.get('Principled BSDF')")
        system_parts.append("if bsdf:")
        system_parts.append("    # Set transmission (glass transparency)")
        system_parts.append("    if 'Transmission Weight' in bsdf.inputs:")
        system_parts.append("        bsdf.inputs['Transmission Weight'].default_value = 1.0")
        system_parts.append("    elif 'Transmission' in bsdf.inputs:")
        system_parts.append("        bsdf.inputs['Transmission'].default_value = 1.0")
        system_parts.append("    # Set roughness (smooth)")
        system_parts.append("    bsdf.inputs['Roughness'].default_value = 0.0")
        system_parts.append("    # Set IOR (glass index of refraction)")
        system_parts.append("    if 'IOR' in bsdf.inputs:")
        system_parts.append("        bsdf.inputs['IOR'].default_value = 1.45")
        system_parts.append("    # Make base color white/light")
        system_parts.append("    if 'Base Color' in bsdf.inputs:")
        system_parts.append("        bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)")
        system_parts.append("for obj in bpy.context.selected_objects:")
        system_parts.append("    if obj.type == 'MESH':")
        system_parts.append("        if not obj.data.materials:")
        system_parts.append("            obj.data.materials.append(mat)")
        system_parts.append("        else:")
        system_parts.append("            obj.data.materials[0] = mat")
        system_parts.append("")
        system_parts.append("JSON format for execute call (use \\n for newlines):")
        system_parts.append("{\"op\":\"execute\",\"kwargs\":{\"code\":\"import bpy\\nmat = bpy.data.materials.new(name='Glass')\\nmat.use_nodes = True\\nbsdf = mat.node_tree.nodes.get('Principled BSDF')\\nif bsdf:\\n    if 'Transmission Weight' in bsdf.inputs:\\n        bsdf.inputs['Transmission Weight'].default_value = 1.0\\n    elif 'Transmission' in bsdf.inputs:\\n        bsdf.inputs['Transmission'].default_value = 1.0\\n    bsdf.inputs['Roughness'].default_value = 0.0\\n    if 'IOR' in bsdf.inputs:\\n        bsdf.inputs['IOR'].default_value = 1.45\\n    if 'Base Color' in bsdf.inputs:\\n        bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)\\nfor obj in bpy.context.selected_objects:\\n    if obj.type == 'MESH':\\n        if not obj.data.materials:\\n            obj.data.materials.append(mat)\\n        else:\\n            obj.data.materials[0] = mat\"}}")
        system_parts.append("")
        system_parts.append("VERIFY: The material MUST be named 'Glass' and have all properties set")
        system_parts.append("CRITICAL: Always check if input exists before setting: if 'InputName' in bsdf.inputs:")
        system_parts.append("DO NOT: Create material without setting properties")
        system_parts.append("DO NOT: Use generic names like 'Material'")
        system_parts.append("DO NOT: Skip any of the BSDF property settings")
        system_parts.append("DO NOT: Access inputs without checking if they exist first")
        
        # Reference resolution guidance
        system_parts.append(f"\n--- Reference Resolution ---")
        system_parts.append("When command contains references (them, it, these, those, they, all of them):")
        system_parts.append("1. 'them'/'they'/'all of them' refers to objects created/mentioned earlier in the command")
        system_parts.append("2. 'it' refers to the active/selected object or last created object")
        system_parts.append("3. Resolve references by tracking what was created/modified in previous steps")
        system_parts.append("4. Example: 'add 3 cubes and give them all glass materials' means:")
        system_parts.append("   - Step 1: Create 3 cubes (mesh.primitive_cube_add, 3 times)")
        system_parts.append("   - Step 2: Create glass material (material.new)")
        system_parts.append("   - Step 3: Assign material to all 3 cubes (loop through objects, material.assign)")
        system_parts.append("5. Always output ALL steps needed, including material creation AND assignment")
    
    system = "\n".join(system_parts)
    user = f"Instruction: {transcript}\nReturn JSON only."

    print("🧠 GPT mapping…")
    try:
        api_key = _get_openai_api_key().strip()
        
        # Validate key is present
        if not api_key:
            print("⚠️ GPT error: API key is empty")
            return None
        
        # Debug: show key length and first/last few chars (for troubleshooting)
        if VERBOSE_DEBUG:
            key_len = len(api_key)
            key_start = api_key[:10] if key_len > 10 else api_key
            key_end = api_key[-4:] if key_len > 4 else ""
            print(f"[DEBUG] Using API key: length={key_len}, starts with: {key_start}..., ends with: ...{key_end}")
            
            # Check for hidden/non-printable characters
            non_printable = [c for c in api_key if not c.isprintable()]
            if non_printable:
                print(f"[DEBUG] ⚠️ Key contains {len(non_printable)} non-printable characters: {[ord(c) for c in non_printable[:5]]}")
                # Clean the key
                api_key = ''.join(c for c in api_key if c.isprintable())
                print(f"[DEBUG] Cleaned key length: {len(api_key)}")
            
            # Check for non-ASCII
            non_ascii = [c for c in api_key if ord(c) > 127]
            if non_ascii:
                print(f"[DEBUG] ⚠️ Key contains non-ASCII characters: {[ord(c) for c in non_ascii[:5]]}")
            
            # Show raw representation
            print(f"[DEBUG] Key repr (first 30): {repr(api_key[:30])}")
            print(f"[DEBUG] Key repr (last 30): {repr(api_key[-30:])}")
        
        # Key validation test removed for performance - the actual GPT call will validate the key
        # if VERBOSE_DEBUG:
        #     try:
        #         test_client = OpenAI(api_key=api_key)
        #         # Try a minimal test call
        #         test_models = test_client.models.list()
        #         print(f"[DEBUG] ✅ Key validation test passed - {len(list(test_models))} models accessible")
        #     except Exception as test_e:
        #         print(f"[DEBUG] ❌ Key validation test failed: {test_e}")
        #         # Continue anyway to see the actual error
        
        # Create client with explicit key
        client = OpenAI(api_key=api_key)
        
        # Build messages with conversation history and image
        messages = [{"role": "system", "content": system}]
        
        # Add conversation history (last 10 messages)
        history = get_conversation_context(max_messages=10)
        for hist_msg in history:
            # Reconstruct message from history
            msg_content = hist_msg["content"]
            # If history has images, include them
            if hist_msg.get("images") and hist_msg["images"]:
                # Format as vision API message
                content_list = [{"type": "text", "text": msg_content}]
                for img_data in hist_msg["images"]:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_data}"
                        }
                    })
                messages.append({
                    "role": hist_msg["role"],
                    "content": content_list
                })
            else:
                messages.append({
                    "role": hist_msg["role"],
                    "content": msg_content
                })
        
        # Add current user message with optional screenshot
        if screenshot_data:
            user_content = [
                {"type": "text", "text": user},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_data}"
                    }
                }
            ]
            print(f"📸 Including viewport screenshot in GPT call ({len(screenshot_data)} bytes)")
        else:
            user_content = user
            print("⚠️ No screenshot available for this GPT call")
        
        messages.append({"role": "user", "content": user_content})
        
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
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
            # Save to conversation history
            add_to_conversation_history("user", transcript, [screenshot_data] if screenshot_data else None)
            add_to_conversation_history("assistant", out)
            return cmd
        if isinstance(cmd, list) and all(isinstance(x, dict) and "op" in x for x in cmd):
            # Save to conversation history
            add_to_conversation_history("user", transcript, [screenshot_data] if screenshot_data else None)
            add_to_conversation_history("assistant", out)
            return cmd
        print("⚠️ GPT returned JSON but missing 'op':", str(out)[:500])
        return None
    except Exception as e:
        error_msg = str(e)
        # Check for common API key errors
        if "Invalid API key" in error_msg or "401" in error_msg or "unauthorized" in error_msg.lower():
            print(f"⚠️ GPT error: Invalid API key. Please check your OpenAI API key in Blender preferences.")
            print(f"   Key preview: {api_key[:10]}..." if len(api_key) > 10 else f"   Key: {api_key}")
            print(f"   Key length: {len(api_key)} characters")
            print(f"   Troubleshooting:")
            print(f"   1. Verify the key is correct in OpenAI dashboard: https://platform.openai.com/api-keys")
            print(f"   2. Check if the key has been revoked or expired")
            print(f"   3. Ensure the key has 'gpt-4o' model access")
            print(f"   4. For project keys (sk-proj-), verify project billing is active")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            print(f"⚠️ GPT error: Rate limit exceeded. Please try again later.")
        else:
            print(f"⚠️ GPT error: {error_msg}")
        return None

# ------------------ RPC send ------------------

def send_to_blender(cmd: dict):
    if not isinstance(cmd, dict):
        print("⚠️ command is not a dict:", cmd); sys.stdout.flush(); return
    op = cmd.get("op"); kwargs = cmd.get("kwargs", {}) or {}
    if not op or not isinstance(op, str):
        print("⚠️ Missing/invalid 'op' in command:", cmd); sys.stdout.flush(); return
    if rpc is None:
        print("❌ RPC not available - cannot send command"); sys.stdout.flush(); return
    try:
        print(f"➡️ enqueue {op} {kwargs}"); sys.stdout.flush()
        # NOTE: If your bridge exposes a different method name, swap below accordingly,
        # e.g. rpc.enqueue_op or rpc.enqueue
        res = rpc.enqueue_op_safe(op, kwargs)
        print("✅ RPC:", res); sys.stdout.flush()
        return res
    except Exception as e:
        print("❌ RPC failed:", e)
        traceback.print_exc()
        # Do not raise; keep loop alive

# ------------------ Main loop ------------------

if __name__ == "__main__":
    try:
        print("Voice → Whisper → (IO rules / Local rules / GPT) → Blender via Safety Gate (Ctrl+C to exit)")
        print("=" * 70)
        
        # Test imports first
        try:
            import numpy as np
            import sounddevice as sd
            print("✅ Imports OK")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please install: pip install numpy sounddevice")
            input("\nPress Enter to exit...")
            sys.exit(1)
        
        # Test RPC connection
        if rpc is None:
            print(f"❌ RPC proxy not created")
            print(f"   Make sure Blender RPC server is running on {RPC_URL}")
        else:
            try:
                ping_result = rpc.ping()
                print(f"✅ RPC connection OK: {ping_result}")
            except Exception as e:
                print(f"❌ RPC connection failed: {e}")
                print(f"   Make sure Blender RPC server is running on {RPC_URL}")
                print("   The script will continue but commands won't work until RPC is available.")
                traceback.print_exc()
        
        # Get OpenAI API key from RPC (Blender preferences) - this will cache it
        try:
            api_key = _get_openai_api_key()
            if api_key:
                key_preview = api_key[:7] + "..." if len(api_key) > 7 else api_key
                print(f"✅ OpenAI API key loaded and cached ({key_preview})")
                if not api_key.startswith("sk-") and not api_key.startswith("sk-proj-"):
                    print(f"⚠️ Warning: API key format looks unusual (should start with 'sk-' or 'sk-proj-')")
            elif ENABLE_GPT_FALLBACK:
                print("⚠️ OpenAI API key not set in preferences or environment; GPT fallback disabled")
        except Exception as e:
            print("⚠️ Could not get OpenAI API key:", e)
            try:
                api_key = _get_openai_api_key()
                if api_key and ENABLE_GPT_FALLBACK:
                    print("✅ Using OpenAI API key from environment variable (cached)")
            except Exception:
                pass

        while True:
            try:
                listening_enabled = True
                if rpc is not None:
                    try:
                        state = rpc.get_voice_listening_state()
                        listening_enabled = state.get("enabled", True)
                    except Exception:
                        listening_enabled = True
                
                if not listening_enabled:
                    time.sleep(0.1)
                    continue
                
                wav = record_until_silence()
                if not wav:
                    continue
                
                listening_enabled = True
                if rpc is not None:
                    try:
                        state = rpc.get_voice_listening_state()
                        listening_enabled = state.get("enabled", True)
                    except Exception:
                        listening_enabled = True
                
                if not listening_enabled:
                    print("⏸️ Listening disabled - skipping audio processing"); sys.stdout.flush()
                    continue
                
                text = transcribe(wav)
                if not text:
                    print("⚠️ No transcription."); sys.stdout.flush()
                    continue
                print("📝 Transcript ->", text); sys.stdout.flush()

                if rpc is not None:
                    try:
                        rpc.start_voice_command()
                    except Exception:
                        pass
                    
                    try:
                        super_state = rpc.get_super_mode_state()
                        SUPER_MODE_TARGET = super_state.get("target_object", "")
                        USE_REACT_REASONING = bool(super_state.get("use_react", False))
                        if SUPER_MODE_TARGET:
                            print(f"🎯 Modeling target: {SUPER_MODE_TARGET}")
                        if USE_REACT_REASONING:
                            print("[ReAct] Iterative reasoning enabled")
                    except Exception:
                        SUPER_MODE_TARGET = ""
                        USE_REACT_REASONING = False
                else:
                    SUPER_MODE_TARGET = ""
                    USE_REACT_REASONING = False

                has_references = bool(re.search(r'\b(them|it|these|those|they|all of them|each of them)\b', text, re.IGNORECASE))
                command_parts = _split_multiple_commands(text)

                context_cache = {"modeling": None, "mesh": None, "scene": None, "screenshot": None}

                def ensure_context():
                    # Always fetch fresh screenshot (don't cache it) since scene changes between commands
                    # But cache modeling/mesh/scene context to avoid repeated RPC calls
                    if context_cache["modeling"] is None:
                        print("[Context] Fetching fresh context (first time)...")
                        modeling_context_cache, mesh_analysis_cache, scene_analysis_cache, screenshot_cache = _fetch_context_for_gpt(include_screenshot=True)
                        context_cache["modeling"] = modeling_context_cache
                        context_cache["mesh"] = mesh_analysis_cache
                        context_cache["scene"] = scene_analysis_cache
                        context_cache["screenshot"] = screenshot_cache
                        print(f"[Context] Cached context, screenshot: {'✅' if screenshot_cache else '❌ None'}")
                    else:
                        # Context is cached, but always get fresh screenshot
                        print("[Context] Context cached, fetching fresh screenshot...")
                        _, _, _, screenshot_cache = _fetch_context_for_gpt(include_screenshot=True)
                        context_cache["screenshot"] = screenshot_cache
                        print(f"[Context] Fresh screenshot: {'✅' if screenshot_cache else '❌ None'}")
                    return context_cache["modeling"], context_cache["mesh"], context_cache.get("scene"), context_cache.get("screenshot")

                all_commands = []

                local_sequence = None
                if not has_references:
                    local_sequence = _resolve_local_sequence(command_parts)

                if local_sequence:
                    all_commands = local_sequence
                else:
                    react_attempted = False
                    if USE_REACT_REASONING and ENABLE_GPT_FALLBACK:
                        ensure_context()
                        react_result = gpt_to_json_react(
                            text,
                            modeling_context=context_cache["modeling"],
                            mesh_analysis=context_cache["mesh"],
                            scene_analysis=context_cache.get("scene"),
                            target_object=SUPER_MODE_TARGET,
                            screenshot_data=context_cache.get("screenshot"),
                        )
                        react_attempted = True
                        if react_result:
                            executed = len(react_result.get("commands", []))
                            iterations = react_result.get("iterations", "?")
                            print(f"[ReAct] Executed {executed} commands in {iterations} iterations")
                            continue
                        else:
                            print("[ReAct] No plan returned, falling back to standard GPT")

                    if has_references and ENABLE_GPT_FALLBACK:
                        local_cmds = []
                        for part in command_parts:
                            cmd = try_io_rules(part)
                            cmd = cmd or try_local_rules(part)
                            if cmd:
                                if isinstance(cmd, list):
                                    local_cmds.extend(cmd)
                                else:
                                    local_cmds.append(cmd)
                        
                        ensure_context()
                        gpt_cmd = gpt_to_json(
                            text,
                            modeling_context=context_cache["modeling"],
                            mesh_analysis=context_cache["mesh"],
                            target_object=SUPER_MODE_TARGET,
                            screenshot_data=context_cache.get("screenshot"),
                        )
                        
                        if gpt_cmd:
                            if isinstance(gpt_cmd, list):
                                all_commands.extend(gpt_cmd)
                            else:
                                all_commands.append(gpt_cmd)
                        else:
                            all_commands.extend(local_cmds)
                    else:
                        for part in command_parts:
                            cmd = try_io_rules(part)
                            cmd = cmd or try_local_rules(part)
                            
                            if not cmd and ENABLE_GPT_FALLBACK:
                                ensure_context()
                                cmd = gpt_to_json(
                                    part,
                                    modeling_context=context_cache["modeling"],
                                    mesh_analysis=context_cache["mesh"],
                                    target_object=SUPER_MODE_TARGET,
                                    screenshot_data=context_cache.get("screenshot"),
                                )
                            
                            if cmd:
                                if isinstance(cmd, list):
                                    all_commands.extend(cmd)
                                else:
                                    all_commands.append(cmd)
                
                if all_commands:
                    listening_enabled = True
                    if rpc is not None:
                        try:
                            state = rpc.get_voice_listening_state()
                            listening_enabled = state.get("enabled", True)
                        except Exception:
                            listening_enabled = True
                    
                    if not listening_enabled:
                        print("⏸️ Listening disabled - skipping command execution"); sys.stdout.flush()
                    else:
                        for single in all_commands:
                            send_to_blender(single)
                            time.sleep(0.01)
                else:
                    print("🤷 No command derived for:", text); sys.stdout.flush()

                    time.sleep(0.05)

            except KeyboardInterrupt:
                print("\nBye.")
                sys.exit(0)
            except Exception as e:
                print("💥 Loop error:", e)
                traceback.print_exc()
                time.sleep(0.1)
                continue
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ FATAL ERROR - Script crashed!")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("This window will close in 10 seconds...")
        print("(Check the error above to diagnose the issue)")
        time.sleep(10)
        sys.exit(1)
