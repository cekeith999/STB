# stb_core/providers/mock.py
import time
from typing import Dict, Any

class MockProvider:
    name = "mock"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        cfg = cfg or {}
        self.delay_s = int(cfg.get("delay_s", 3))
        self.result_path = cfg.get("result_path")  # absolute path to an existing .glb/.fbx
        self._start = None

    def supports(self, capability: str) -> bool:
        return True

    def submit(self, task: Dict[str, Any]) -> str:
        self._start = time.time()
        return "mock_job_1"

    def status(self, job_id: str) -> Dict[str, Any]:
        elapsed = 0 if self._start is None else time.time() - self._start
        pct = 100 if self.delay_s <= 0 else int(min(100, max(0, (elapsed / self.delay_s) * 100)))
        return {"state": ("succeeded" if pct >= 100 else "running"), "progress": pct}

    def fetch_result(self, job_id: str) -> Dict[str, Any]:
        return {"files": [{"path": self.result_path}], "meta": {"provider": "mock"}}
