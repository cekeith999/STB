# Speech To Blender/stb_core/providers/registry.py
from typing import Dict, Any, Type
from .meshy import MeshyProvider
from .mock import MockProvider

_REGISTRY: Dict[str, Type] = {
    "meshy": MeshyProvider,
    "mock": MockProvider,
}

def get_provider(name: str, cfg_block: Dict[str, Any]):
    cls = _REGISTRY.get(name)
    if not cls:
        raise KeyError(f"Unknown provider: {name}")
    return cls(cfg_block)

def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())
