"""
Shared data structures for the multi-agent system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class TaskType(Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    QUERY = "query"
    NAVIGATE = "navigate"

@dataclass
class InferredOperation:
    """A single inferred operation to achieve the user's goal."""
    action: str                      # e.g., "bevel", "scale", "subdivide"
    target: str                      # Object name or "selected" or "active"
    parameters: Dict[str, Any]       # Operation-specific params
    reason: str                      # Why this operation helps
    priority: int = 1                # Execution order (1 = first)

@dataclass 
class TargetProperties:
    """Properties the user wants the object to have."""
    shape: Optional[str] = None              # "cylindrical", "cubic", etc.
    proportions: Optional[str] = None        # "squat", "tall", "1:1:2 ratio"
    surface_quality: Optional[str] = None    # "smooth", "rough", "textured"
    edge_treatment: Optional[str] = None     # "sharp", "beveled", "rounded"
    size: Optional[Dict[str, float]] = None  # {"width": 2.0, "height": 0.5}
    custom: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskSpec:
    """
    The contract between Language Translator and Code Generator.
    Language Translator produces this; Code Generator consumes it.
    """
    # Original input
    raw_transcript: str
    
    # Parsed intent
    task_type: TaskType
    user_intent: str                         # Human-readable description
    
    # Reference resolution
    target_objects: List[str]                # Resolved object names
    referent_resolution_method: str          # How we resolved "it"/"this"
    
    # Target state (what user wants)
    target_concept: Optional[str] = None     # "Echo Dot", "coffee mug", etc.
    target_properties: Optional[TargetProperties] = None
    
    # Inferred operations
    inferred_operations: List[InferredOperation] = field(default_factory=list)
    
    # Context that was used
    scene_context_summary: str = ""
    current_object_summary: str = ""
    
    # Confidence
    confidence: float = 1.0                  # 0-1, how confident in interpretation
    ambiguities: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result from Code Generator execution."""
    success: bool
    operations_executed: List[str]
    objects_modified: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    

@dataclass
class EvaluationResult:
    """Result from Semantic Evaluator."""
    semantic_match_score: float              # 0-1
    questions_asked: Dict[str, str]          # Question -> Answer
    issues_found: List[str]
    suggested_refinements: List[InferredOperation]
    should_retry: bool

# Lazy import for LanguageTranslator (to avoid circular imports)
def get_language_translator(openai_client):
    """Get LanguageTranslator instance (lazy import)."""
    from agents.language_translator import LanguageTranslator
    return LanguageTranslator(openai_client)

# Lazy import for Orchestrator (to avoid circular imports)
def get_orchestrator(api_key: str):
    """Get Orchestrator instance (lazy import)."""
    from agents.orchestrator import get_orchestrator as _get_orchestrator
    return _get_orchestrator(api_key)

__all__ = [
    'TaskType',
    'InferredOperation',
    'TargetProperties',
    'TaskSpec',
    'ExecutionResult',
    'EvaluationResult',
    'get_language_translator',
    'get_orchestrator',
]
