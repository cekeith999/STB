# stb_core/commands/schema.py
from dataclasses import dataclass
from typing import Literal, Dict, Any

CommandType = Literal[
    "ADD_MESH",
    "IMPORT",
    "TRANSFORM",
    "RENDER",
    "SET_MATERIAL",
    "SET_CAMERA"
]

@dataclass
class Command:
    type: CommandType
    args: Dict[str, Any]
