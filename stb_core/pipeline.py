# stb_core/pipeline.py
import time
from typing import Dict, Any, Callable
# stb_core/progress.py
import os, json, time
from typing import Dict, Any

def write_progress(repo_root: str, job_id: str, progress: int, state: str) -> None:
    os.makedirs(os.path.join(repo_root, "logs"), exist_ok=True)
    path = os.path.join(repo_root, "logs", "progress.json")
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "job_id": job_id,
        "progress": progress,
        "state": state,
    }
    # atomic-ish write
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)

ProgressFn = Callable[[str, int, str], None]  # (job_id, progress, state)

class Pipeline:
    """
    Orchestrates a provider job: submit -> poll -> fetch_result.
    Calls on_progress(job_id, progress:int 0..100, state:str).
    """
    def __init__(self, cfg: Dict[str, Any], provider):
        self.cfg = cfg
        self.provider = provider

    def run(self, capability: str, task_payload: Dict[str, Any],
            on_progress: ProgressFn | None = None) -> Dict[str, Any]:
        job_id = self.provider.submit({"capability": capability, **task_payload})

        poll = int(self.cfg.get("pipeline", {}).get("poll_interval_s", 3))
        timeout_s = int(self.cfg.get("pipeline", {}).get("timeout_s", 900))
        deadline = time.time() + timeout_s

        last_progress = -1
        while True:
            if time.time() > deadline:
                raise TimeoutError("Pipeline timed out waiting for provider job")

            s = self.provider.status(job_id)
            state = (s.get("state") or s.get("status") or "unknown").lower()
            progress = int(s.get("progress", 0))

            if on_progress and progress != last_progress:
                on_progress(job_id, progress, state)
                last_progress = progress

            if state in {"succeeded", "completed", "done"}:
                break
            if state in {"failed", "error"}:
                raise RuntimeError(s.get("error", "Provider reported failure"))

            time.sleep(poll)

        return self.provider.fetch_result(job_id)
