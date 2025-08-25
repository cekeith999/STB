# Speech To Blender/stb_core/providers/meshy.py
# MeshyProvider — submit jobs, poll status, fetch results, and async import into Blender.

from typing import Dict, Any, List, Optional, Tuple
import os, json, urllib.request, urllib.error, urllib.parse, tempfile, pathlib, shutil
import threading, time, collections, re
import bpy

Json = Dict[str, Any]

# ---------------- Meshy collection helpers ----------------

def ensure_meshy_collection(name: str = "Meshy_Imports"):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        # link to scene root
        bpy.context.scene.collection.children.link(coll)
    # make sure it is linked (covers edge case where coll exists but unlinked)
    names = {c.name for c in bpy.context.scene.collection.children}
    if coll.name not in names:
        bpy.context.scene.collection.children.link(coll)
    return coll

def route_new_imports_to_meshy(before_objs: set) -> List[bpy.types.Object]:
    dest = ensure_meshy_collection()
    after_objs = set(bpy.data.objects)
    new_objs = [o for o in (after_objs - before_objs) if o.users_collection]
    for ob in new_objs:
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        dest.objects.link(ob)
    return new_objs

def _normalize_import(objects: List[bpy.types.Object]) -> None:
    if not objects:
        return
    bpy.ops.object.select_all(action='DESELECT')
    for o in objects:
        try:
            o.select_set(True)
        except Exception:
            pass
    try:
        bpy.context.view_layer.objects.active = objects[0]
    except Exception:
        pass
    # apply rotation then scale; origin to bounds
    try:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    except Exception:
        pass
    try:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass
    try:
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    except Exception:
        pass
    for o in objects:
        try:
            o.select_set(False)
        except Exception:
            pass

# ---------------- Naming de-dupe helpers ----------------

def _unique_name(base: str, used: set[str]) -> str:
    """Return a unique name by appending _2, _3, ... (avoids Blender's .001 style)."""
    if base not in used:
        return base
    m = re.match(r"^(.*)_(\d+)$", base)
    root = m.group(1) if m else base
    n = int(m.group(2)) if m else 1
    while True:
        n += 1
        cand = f"{root}_{n}"
        if cand not in used:
            return cand

def _dedupe_object_names(objs: List[bpy.types.Object]) -> None:
    """Ensure imported objects don't collide with existing names."""
    used = {o.name for o in bpy.data.objects}
    for o in objs:
        new_name = _unique_name(o.name, used)
        if new_name != o.name:
            o.name = new_name
        used.add(o.name)

# ---------------- HTTP + provider ----------------

class MeshyProvider:
    name = "meshy"

    def __init__(self, cfg: Dict[str, Any]):
        """
        cfg example:
        {
          "api_key_env": "MESHY_API_KEY",
          "base_url": "https://api.meshy.ai",
          "model": "v5",
          "endpoints": {
            "text2mesh": [...],
            "img2mesh":  [...],
            "task": [...]
          },
          "dl_format_preference": ["glb","fbx","obj","usdz"]
        }
        """
        self._last_request = ""

        self._cfg: Dict[str, Any] = cfg or {}

        # API key: prefs override env
        self._key_env: str = self._cfg.get("api_key_env", "MESHY_API_KEY")
        self._key: Optional[str] = self._cfg.get("api_key") or os.environ.get(self._key_env)
        if not self._key:
            raise RuntimeError(f"No Meshy API key provided (prefs) and env var {self._key_env} is not set.")

        self._base: str = (self._cfg.get("base_url") or "https://api.meshy.ai").rstrip("/")
        self._model: str = self._cfg.get("model", "v5")

        # Endpoint candidates (robust across Meshy versions/tenants)
        self._eps_text2mesh: List[str] = list((self._cfg.get("endpoints") or {}).get("text2mesh") or [
            # Common current
            "/openapi/v3/text-to-3d",
            "/openapi/v2/text-to-3d",
            "/openapi/v1/text-to-3d",
            # Older/simple
            "/v3/text-to-3d",
            "/v2/text-to-3d",
            "/v1/text-to-3d",
            "/text-to-3d",
            # Some tenants expose under /api
            "/api/v3/text-to-3d",
            "/api/v2/text-to-3d",
            "/api/v1/text-to-3d",
        ])

        self._eps_img2mesh: List[str] = list((self._cfg.get("endpoints") or {}).get("img2mesh") or [
            "/openapi/v3/image-to-3d",
            "/openapi/v2/image-to-3d",
            "/openapi/v1/image-to-3d",
            "/v3/image-to-3d",
            "/v2/image-to-3d",
            "/v1/image-to-3d",
            "/image-to-3d",
            "/api/v3/image-to-3d",
            "/api/v2/image-to-3d",
            "/api/v1/image-to-3d",
        ])

        self._eps_task: List[str] = list((self._cfg.get("endpoints") or {}).get("task") or [
            # v3/v2 task/status shapes first
            "/openapi/v3/text-to-3d/{job_id}",
            "/openapi/v3/image-to-3d/{job_id}",
            "/openapi/v2/text-to-3d/{job_id}",
            "/openapi/v2/image-to-3d/{job_id}",
            "/openapi/v1/text-to-3d/{job_id}",
            "/openapi/v1/image-to-3d/{job_id}",
            "/v3/tasks/{job_id}",
            "/v2/tasks/{job_id}",
            "/v1/tasks/{job_id}",
            "/tasks/{job_id}",
            "/api/v3/tasks/{job_id}",
            "/api/v2/tasks/{job_id}",
            "/api/v1/tasks/{job_id}",
        ])

        self._dl_pref: List[str] = list(self._cfg.get("dl_format_preference") or ["glb", "fbx", "obj", "usdz"])

    # ---- HTTP helpers ----
    def _build_url(self, path: str) -> str:
        return f"{self._base}/{path.lstrip('/')}"

    def _req(self, path: str, method: str = "GET", data: Optional[Json] = None) -> Json:
        url = self._build_url(path)
        self._last_request = f"{method} {url}"
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        body = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode("utf-8")
            except Exception:
                err = str(e)
            raise RuntimeError(f"HTTP {e.code} {method} {path}: {err}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error {method} {path}: {e}") from None

    def friendly_error(self, e: Exception) -> str:
        msg = str(e); low = msg.lower()
        if "http 401" in low or "http 403" in low:
            return f"Authorization failed — check MESHY_API_KEY (or addon prefs). [{self._last_request}]"
        if "http 404" in low:
            return f"Endpoint not found — try /openapi/v3 or /openapi/v2 family. [{self._last_request}]"
        if "timeout" in low or "timed out" in low or "network error" in low:
            return f"Network timeout contacting Meshy. [{self._last_request}]"
        if "no downloadable files" in low:
            return "Job finished but returned no downloadable files."
        if "unsupported extension" in low:
            return "Downloaded format isn’t supported by this Blender build."
        return f"{msg} [{self._last_request}]"

    def _try_post(self, candidates: List[str], payload: Json) -> Json:
        tried, last_err = [], None
        for path in candidates:
            tried.append(path)
            try:
                return self._req(path, "POST", payload)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"All submit endpoints failed: {last_err} | tried={tried}")

    def _try_get(self, candidates: List[str]) -> Json:
        tried, last_err = [], None
        for path in candidates:
            tried.append(path)
            try:
                return self._req(path, "GET", None)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"All status endpoints failed: {last_err} | tried={tried}")

    # ---- Provider interface ----
    def supports(self, capability: str) -> bool:
        return (capability or "").lower() in {"text2mesh", "img2mesh"}

    def submit(self, payload: Json) -> str:
        cap = (payload.get("capability") or "").lower()
        if cap in ("text2mesh", "text-to-3d", "text_to_3d"):
            candidates = self._eps_text2mesh
            body = {
                "mode": payload.get("mode", "preview"),
                "title": payload.get("title") or "STB Text2Mesh",
                "prompt": payload.get("prompt", "")
            }
        elif cap in ("img2mesh", "image-to-3d", "image_to_3d"):
            candidates = self._eps_img2mesh
            body = {
                "mode": payload.get("mode", "preview"),
                "title": payload.get("title") or "STB Img2Mesh",
                "image_url": payload.get("image_url") or payload.get("url", ""),
                "prompt": payload.get("prompt", "")
            }
        else:
            raise ValueError(f"Unsupported capability for Meshy: {cap}")

        data = self._try_post(candidates, body)

        # Extract job id from multiple possible shapes
        job_id: Optional[str] = None
        possible = [
            data.get("id"), data.get("result"), data.get("task_id"), data.get("job_id"),
            data.get("taskId"), data.get("jobId"),
        ]
        if isinstance(data.get("result"), dict):
            res = data["result"]
            possible.extend([res.get(k) for k in ("id", "task_id", "job_id", "taskId", "jobId")])
        for cand in possible:
            if isinstance(cand, str) and len(cand) >= 8:
                job_id = cand
                break
        if not job_id:
            raise RuntimeError(f"submit() returned no job id. Raw: {data}")
        return str(job_id)

    def status(self, job_id: str) -> Json:
        paths = [p.format(job_id=job_id) for p in self._eps_task]
        raw = self._try_get(paths)

        state = (raw.get("state") or raw.get("status") or raw.get("task_status") or raw.get("phase") or "unknown")
        state_l = str(state).lower()

        prog = raw.get("progress")
        if prog is None:
            for parent in ("data", "result", "output"):
                sub = raw.get(parent)
                if isinstance(sub, dict) and "progress" in sub:
                    prog = sub["progress"]
                    break

        if isinstance(prog, float) and 0.0 <= prog <= 1.0:
            progress = int(round(prog * 100))
        elif isinstance(prog, (int, float)):
            progress = int(round(prog))
        else:
            progress = 100 if "succeed" in state_l or "complete" in state_l else 0

        if any(k in state_l for k in ("succeed", "complete", "done", "success")):
            norm_state = "succeeded"
        elif any(k in state_l for k in ("fail", "error", "cancel")):
            norm_state = "failed"
        elif any(k in state_l for k in ("queue", "pend", "start", "run", "proc")):
            norm_state = "running"
        else:
            norm_state = state_l or "unknown"

        return {"state": norm_state, "progress": progress, "raw": raw}

    def fetch_result(self, job_id: str) -> Json:
        paths = [p.format(job_id=job_id) for p in self._eps_task]
        raw = self._try_get(paths)

        urls: List[str] = []
        for k in ("model_url", "mesh_url", "output_url", "url"):
            u = raw.get(k)
            if isinstance(u, str):
                urls.append(u)

        model_urls = raw.get("model_urls")
        if isinstance(model_urls, dict):
            for _, v in model_urls.items():
                if isinstance(v, str):
                    urls.append(v)

        files_list = raw.get("files")
        if isinstance(files_list, list):
            for item in files_list:
                if isinstance(item, dict):
                    u = item.get("url") or item.get("href") or item.get("download_url")
                    if isinstance(u, str):
                        urls.append(u)

        for parent in ("result", "data", "output", "asset", "assets"):
            sub = raw.get(parent)
            if isinstance(sub, dict):
                for k in ("model_url", "mesh_url", "output_url", "url"):
                    u = sub.get(k)
                    if isinstance(u, str):
                        urls.append(u)
                for _, val in sub.items():
                    if isinstance(val, list):
                        for it in val:
                            if isinstance(it, dict) and isinstance(it.get("url"), str):
                                urls.append(it["url"])

        # de-dup
        seen, dedup = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u)
                dedup.append(u)

        files: List[Json] = []
        for u in dedup:
            fmt = None
            low = u.lower()
            for ext in ("fbx", "obj", "glb", "gltf", "zip", "usdz"):
                if low.endswith("." + ext) or f".{ext}?" in low:
                    fmt = ext
                    break
            files.append({"url": u, "format": fmt})

        return {"files": files, "meta": raw}

    # ---------------- Download + Import (sync) ----------------

    def _pick_best_file(self, files: List[Json]) -> Optional[Json]:
        if not files:
            return None
        bucket = {}
        for f in files:
            fmt = (f.get("format") or "").lower()
            bucket.setdefault(fmt, []).append(f)
        for ext in self._dl_pref:
            if ext in bucket and bucket[ext]:
                return bucket[ext][0]
        return files[0]

    def _download_to_temp(self, url: str) -> pathlib.Path:
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="stb_meshy_"))
        parsed = urllib.parse.urlparse(url)
        name = os.path.basename(parsed.path) or "meshy_model"
        dest = tmpdir / name
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest

    def _import_path(self, path: pathlib.Path) -> Tuple[List[bpy.types.Object], str]:
        ext = path.suffix.lower().lstrip(".")
        before = set(bpy.data.objects)

        try:
            if ext in {"glb", "gltf"}:
                try:
                    bpy.ops.import_scene.gltf(filepath=str(path)); used = "import_scene.gltf"
                except Exception:
                    bpy.ops.wm.gltf_import(filepath=str(path)); used = "wm.gltf_import"
            elif ext == "fbx":
                bpy.ops.import_scene.fbx(filepath=str(path)); used = "import_scene.fbx"
            elif ext == "obj":
                try:
                    bpy.ops.wm.obj_import(filepath=str(path)); used = "wm.obj_import"
                except Exception:
                    bpy.ops.import_scene.obj(filepath=str(path)); used = "import_scene.obj"
            elif ext in {"usd", "usda", "usdc", "usdz"}:
                bpy.ops.wm.usd_import(filepath=str(path)); used = "wm.usd_import"
            else:
                raise RuntimeError(f"Unsupported extension: .{ext}")
        except Exception as e:
            raise RuntimeError(f"Import failed for {path.name} via {ext}: {e}")

        new_objs = route_new_imports_to_meshy(before)
        _dedupe_object_names(new_objs)
        _normalize_import(new_objs)
        return new_objs, used

    def import_job_result(self, job_id: str) -> Dict[str, Any]:
        res = self.fetch_result(job_id)
        best = self._pick_best_file(res.get("files") or [])
        if not best:
            raise RuntimeError("No downloadable files in job result.")
        url = best["url"]
        local = self._download_to_temp(url)
        new_objs, used = self._import_path(local)
        return {
            "imported_count": len(new_objs),
            "importer": used,
            "local_path": str(local),
            "chosen_format": best.get("format"),
            "collection": ensure_meshy_collection().name,
        }


def generate_from_prompt(context, prompt: str, cfg: dict | None = None):
    """
    Thin convenience wrapper used by the top-level operator.
    Builds a MeshyProvider from prefs/env and submits an async text2mesh job,
    then returns a small dict for UI logging.
    """
    cfg = dict(cfg or {})

    # Normalize formats into provider's download preference order
    fmts = cfg.get("formats") or "glb,fbx,obj"
    if isinstance(fmts, str):
        dl_pref = [f.strip().lower() for f in fmts.split(",") if f.strip()]
    else:
        dl_pref = [str(f).strip().lower() for f in (fmts or [])]

    provider_cfg = {
        # direct API key from prefs takes priority over env
        "api_key": cfg.get("api_key", ""),
        "api_key_env": cfg.get("api_key_env", "MESHY_API_KEY"),
        "base_url": cfg.get("base_url", "https://api.meshy.ai"),
        "model": cfg.get("model", "v5"),

        # Keep explicit endpoint candidates here too for belt-and-suspenders robustness.
        "endpoints": {
            "text2mesh": [
                "/openapi/v3/text-to-3d",
                "/openapi/v2/text-to-3d",
                "/openapi/v1/text-to-3d",
                "/v3/text-to-3d",
                "/v2/text-to-3d",
                "/v1/text-to-3d",
                "/text-to-3d",
                "/api/v3/text-to-3d",
                "/api/v2/text-to-3d",
                "/api/v1/text-to-3d",
            ],
            "img2mesh": [
                "/openapi/v3/image-to-3d",
                "/openapi/v2/image-to-3d",
                "/openapi/v1/image-to-3d",
                "/v3/image-to-3d",
                "/v2/image-to-3d",
                "/v1/image-to-3d",
                "/image-to-3d",
                "/api/v3/image-to-3d",
                "/api/v2/image-to-3d",
                "/api/v1/image-to-3d",
            ],
            "task": [
                "/openapi/v3/text-to-3d/{job_id}",
                "/openapi/v3/image-to-3d/{job_id}",
                "/openapi/v2/text-to-3d/{job_id}",
                "/openapi/v2/image-to-3d/{job_id}",
                "/openapi/v1/text-to-3d/{job_id}",
                "/openapi/v1/image-to-3d/{job_id}",
                "/v3/tasks/{job_id}",
                "/v2/tasks/{job_id}",
                "/v1/tasks/{job_id}",
                "/tasks/{job_id}",
                "/api/v3/tasks/{job_id}",
                "/api/v2/tasks/{job_id}",
                "/api/v1/tasks/{job_id}",
            ],
        },

        "dl_format_preference": dl_pref or ["glb", "fbx", "obj", "usdz"],
    }

    provider = MeshyProvider(provider_cfg)

    # Mode is sent in the payload that submit() builds
    mode = cfg.get("mode", "standard")
    title = cfg.get("title", "STB Text2Mesh")

    # Non-blocking end-to-end: submit → poll → download → import
    meshy_submit_and_import_async(
        provider,
        capability="text2mesh",
        mode=mode,
        title=title,
        prompt=prompt,
    )

    # Status panel will update via the timer pump already running
    return {"submitted": True, "mode": mode, "model": provider._model}


# ---------------- Async (non-blocking) support ----------------

_MAIN_QUEUE = collections.deque()
_PUMP_RUNNING = False

def _schedule_on_main(func, *args, **kwargs):
    _MAIN_QUEUE.append((func, args, kwargs))

def _pump_main_queue():
    try:
        while _MAIN_QUEUE:
            func, args, kwargs = _MAIN_QUEUE.popleft()
            try:
                func(*args, **kwargs)
            except Exception as e:
                print("[Meshy Async] main-thread task error:", e)
    finally:
        return 0.2  # run again in ~0.2s

def _ensure_pump_running():
    global _PUMP_RUNNING
    if not _PUMP_RUNNING:
        bpy.app.timers.register(_pump_main_queue, first_interval=0.2)
        _PUMP_RUNNING = True

def _set_status(txt: str):
    try:
        bpy.context.window_manager["meshy_status"] = txt
    except Exception:
        pass

def get_meshy_status() -> str:
    try:
        return bpy.context.window_manager.get("meshy_status", "")
    except Exception:
        return ""

def _safe_import_path(provider: MeshyProvider, local_path: pathlib.Path):
    new_objs, used = provider._import_path(local_path)
    _set_status(f"Imported via {used} → {len(new_objs)} object(s)")
    print("[Meshy Async] Import complete:", {"used": used, "count": len(new_objs), "path": str(local_path)})

def _bg_worker_submit_and_import(provider: MeshyProvider, payload: Json):
    try:
        _set_status("Queued")
        job_id = provider.submit(payload)
        _set_status(f"Submitted: {job_id}")

        last_p = -1
        for _ in range(600):  # up to ~20 minutes at 2s
            st = provider.status(job_id)
            p = int(st.get("progress", 0))
            state = st.get("state", "running")
            if p != last_p:
                _set_status(f"Generating ({p}%)")
                last_p = p
            if state in ("succeeded", "failed"):
                break
            time.sleep(2)

        if state != "succeeded":
            _set_status(f"Failed ({state})")
            print("[Meshy Async] Job failed:", st)
            return

        _set_status("Downloading")
        res = provider.fetch_result(job_id)
        best = provider._pick_best_file(res.get("files") or [])
        if not best:
            _set_status("No downloadable files in result")
            return
        url = best["url"]
        local = provider._download_to_temp(url)

        _set_status("Importing")
        _schedule_on_main(_safe_import_path, provider, local)
        _set_status("Done")
    except Exception as e:
        _set_status(f"Error: {provider.friendly_error(e)}")
        print("[Meshy Async] Error:", e)

def meshy_submit_and_import_async(provider: MeshyProvider, capability="text2mesh", **kwargs):
    """
    Non-blocking path:
      - submits a job
      - polls Meshy in a background thread
      - downloads the asset
      - schedules main-thread import (no UI freeze)
    kwargs are merged into the provider.submit() payload.
    """
    payload = {"capability": capability}
    payload.update(kwargs)
    _ensure_pump_running()
    t = threading.Thread(target=_bg_worker_submit_and_import, args=(provider, payload), daemon=True)
    t.start()
    return t

# Start the main-thread pump automatically so the status panel updates without “forcing” it.
_ensure_pump_running()
