# Speech To Blender/stb_core/providers/base.py
from typing import Protocol, Dict, Any

class ModelProvider(Protocol):
    name: str
    def supports(self, capability: str) -> bool: ...
    def submit(self, task: Dict[str, Any]) -> str: ...         # returns job_id
    def status(self, job_id: str) -> Dict[str, Any]: ...       # {state, progress, ...}
    def fetch_result(self, job_id: str) -> Dict[str, Any]: ... # {files:[...], meta:{...}}
