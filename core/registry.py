# core/registry.py
from typing import Dict, Callable
class CapabilityRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, Callable] = {}
    def register(self, capability: str, factory: Callable) -> None:
        self._factories[capability.lower()] = factory
    def make(self, capability: str):
        return self._factories[capability.lower()]()
REGISTRY = CapabilityRegistry()
