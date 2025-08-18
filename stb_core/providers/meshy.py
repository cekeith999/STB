# Speech To Blender/stb_core/providers/meshy.py
# MeshyProvider — submit jobs, poll status, fetch results.
# Works with /openapi/v2/... endpoints (configurable) and falls back to older /v2/... shapes.

from typing import Dict, Any, List, Optional
import os, json, urllib.request, urllib.error

Json = Dict[str, Any]

class MeshyProvider:
    name = "meshy"

    def __init__(self, cfg: Dict[str, Any]):
        """
        cfg is the providers['meshy'] block from config/config.json, e.g.:
        {
          "api_key_env": "MESHY_API_KEY",
          "base_url": "https://api.meshy.ai",
          "model": "v5",
          "endpoints": {
            "text2mesh": ["/openapi/v2/text-to-3d", ...],
            "img2mesh":  ["/openapi/v2/image-to-3d", ...],
            "task": [
              "/openapi/v2/text-to-3d/{job_id}",
              "/openapi/v2/image-to-3d/{job_id}",
              ...
            ],
            "stream": [...]
          },
          "dl_format_preference": ["glb","fbx","usdz","obj"]
        }
        """
        self._cfg: Dict[str, Any] = cfg or {}
        self._key_env: str = self._cfg.get("api_key_env", "MESHY_API_KEY")
        self._key: Optional[str] = os.environ.get(self._key_env)
        if not self._key:
            raise RuntimeError(f"Environment variable {self._key_env} is not set.")

        self._base: str = (self._cfg.get("base_url") or "https://api.meshy.ai").rstrip("/")
        self._model: str = self._cfg.get("model", "v5")

        eps = self._cfg.get("endpoints") or {}

        # Preferred: /openapi/v2/... ; Fallbacks: /openapi/v1/... then older /v2, /v1.
        self._eps_text2mesh: List[str] = list(eps.get("text2mesh") or [
            "/openapi/v2/text-to-3d",
            "/openapi/v1/text-to-3d",
            "/v2/text-to-3d",
            "/v1/text-to-3d",
            "/text-to-3d",
        ])
        self._eps_img2mesh: List[str] = list(eps.get("img2mesh") or [
            "/openapi/v2/image-to-3d",
            "/openapi/v1/image-to-3d",
            "/v2/image-to-3d",
            "/v1/image-to-3d",
            "/image-to-3d",
        ])
        # Task lookups: allow both per-capability routes and generic /tasks/{job_id}
        self._eps_task: List[str] = list(eps.get("task") or [
            "/openapi/v2/text-to-3d/{job_id}",
            "/openapi/v2/image-to-3d/{job_id}",
            "/openapi/v1/text-to-3d/{job_id}",
            "/openapi/v1/image-to-3d/{job_id}",
            "/v2/tasks/{job_id}",
            "/v1/tasks/{job_id}",
            "/tasks/{job_id}",
        ])

    # ---------------------------- HTTP helpers ----------------------------

    def _build_url(self, path: str) -> str:
        return f"{self._base}/{path.lstrip('/')}"

    def _req(self, path: str, method: str = "GET", data: Optional[Json] = None) -> Json:
        url = self._build_url(path)
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
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

    def _try_post(self, candidates: List[str], payload: Json) -> Json:
        last_err = None
        for path in candidates:
            try:
                return self._req(path, "POST", payload)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"All submit endpoints failed: {last_err}")

    def _try_get(self, candidates: List[str]) -> Json:
        last_err = None
        for path in candidates:
            try:
                return self._req(path, "GET", None)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"All status endpoints failed: {last_err}")

    # ------------------------ Provider interface -------------------------

    def supports(self, capability: str) -> bool:
        return (capability or "").lower() in {"text2mesh", "img2mesh"}

    def submit(self, payload: Json) -> str:
        """
        Start a Meshy job and return the job_id string.
        Meshy may return the id under 'id', 'result', 'task_id', or 'job_id'
        (and sometimes camelCase variants). We accept any of those; if 'result'
        is an object, we also look inside it.
        """
        cap = (payload.get("capability") or "").lower()
        if cap in ("text2mesh", "text-to-3d", "text_to_3d"):
            candidates = self._eps_text2mesh
            body = {
                "mode": payload.get("mode", "preview"),
                "title": payload.get("title") or "STB Text2Mesh",
                "prompt": payload.get("prompt", ""),
            }
        elif cap in ("img2mesh", "image-to-3d", "image_to_3d"):
            candidates = self._eps_img2mesh
            body = {
                "mode": payload.get("mode", "preview"),
                "title": payload.get("title") or "STB Img2Mesh",
                "image_url": payload.get("image_url") or payload.get("url", ""),
                "prompt": payload.get("prompt", ""),
            }
        else:
            raise ValueError(f"Unsupported capability for Meshy: {cap}")

        data = self._try_post(candidates, body)

        # Extract job id from multiple possible shapes
        job_id: Optional[str] = None
        possible = [
            data.get("id"),
            data.get("result"),
            data.get("task_id"),
            data.get("job_id"),
            data.get("taskId"),
            data.get("jobId"),
        ]
        # If 'result' was a dict, peek inside it.
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
        """
        Returns a normalized dict:
            {"state": "running|succeeded|failed|...", "progress": int(0..100), "raw": <original>}
        """
        paths = [p.format(job_id=job_id) for p in self._eps_task]
        raw = self._try_get(paths)

        # Normalize state
        state = (raw.get("state")
                 or raw.get("status")
                 or raw.get("task_status")
                 or raw.get("phase")
                 or "unknown")
        state_l = str(state).lower()

        # Normalize progress (try top-level, then nested)
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

        # Map to a compact state label
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
        """
        Return a unified result:
          {"files": [{"url": "...", "format": "fbx|obj|glb|gltf|zip|..."}], "meta": <raw>}
        """
        # Fresh fetch (status includes the raw payload on task endpoint)
        paths = [p.format(job_id=job_id) for p in self._eps_task]
        raw = self._try_get(paths)

        urls: List[str] = []

        # Common flat fields
        for k in ("model_url", "mesh_url", "output_url", "url"):
            u = raw.get(k)
            if isinstance(u, str):
                urls.append(u)

        # model_urls dict (Meshy often provides this)
        model_urls = raw.get("model_urls")
        if isinstance(model_urls, dict):
            for k, v in model_urls.items():
                if isinstance(v, str):
                    urls.append(v)

        # files array (list of dicts with 'url')
        files_list = raw.get("files")
        if isinstance(files_list, list):
            for item in files_list:
                if isinstance(item, dict):
                    u = item.get("url") or item.get("href") or item.get("download_url")
                    if isinstance(u, str):
                        urls.append(u)

        # Nested places
        for parent in ("result", "data", "output", "asset", "assets"):
            sub = raw.get(parent)
            if isinstance(sub, dict):
                # direct url fields inside sub
                for k in ("model_url", "mesh_url", "output_url", "url"):
                    u = sub.get(k)
                    if isinstance(u, str):
                        urls.append(u)
                # arrays under sub
                for _, val in sub.items():
                    if isinstance(val, list):
                        for it in val:
                            if isinstance(it, dict) and isinstance(it.get("url"), str):
                                urls.append(it["url"])

        # de-duplicate
        seen = set()
        dedup = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                dedup.append(u)

        files: List[Json] = []
        for u in dedup:
            fmt = None
            lower = u.lower()
            for ext in ("fbx", "obj", "glb", "gltf", "zip", "usdz"):
                if lower.endswith("." + ext) or f".{ext}?" in lower:
                    fmt = ext
                    break
            files.append({"url": u, "format": fmt})

        return {"files": files, "meta": raw}
