# Speech-to-Blender: Multi-Agent Architecture Implementation

## Overview

This document describes the implementation of a multi-agent architecture for Speech-to-Blender (STB). The goal is to separate concerns that are currently handled monolithically by GPT-4o into specialized components with clear responsibilities.

**Current Problem:**
- GPT-4o does everything: language understanding, geometric reasoning, code generation
- Ambiguous references ("it", "this") aren't resolved before GPT sees them
- No closed-loop verification that results match user intent
- Multi-object relationships aren't analyzed
- LLMs hallucinate geometric details they shouldn't be reasoning about

**Solution:**
- Separate specialized agents with clear contracts
- Deterministic analyzers for geometry (not LLMs)
- Focus stack for reference resolution
- Semantic evaluator for closed-loop verification

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                       │
│                      (agents/orchestrator.py)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   LANGUAGE    │          │     CODE      │          │   SEMANTIC    │
│  TRANSLATOR   │          │   GENERATOR   │          │   EVALUATOR   │
│               │          │               │          │               │
│ - Parse intent│          │ - Gen Blender │          │ - Verify via  │
│ - Resolve refs│          │   Python code │          │   VLM Q&A     │
│ - Infer ops   │          │ - Use RAG     │          │ - Feedback    │
└───────────────┘          └───────────────┘          └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   GEOMETRIC   │          │    FOCUS      │          │   GEOMETRIC   │
│   ANALYZER    │          │    STACK      │          │   VERIFIER    │
│  (via RPC)    │          │               │          │  (via RPC)    │
│               │          │ - Track refs  │          │               │
│ - Mesh data   │          │ - Cmd history │          │ - Manifold    │
│ - Scene graph │          │ - Selection   │          │ - Thickness   │
│ - Relations   │          │               │          │ - Printable   │
└───────────────┘          └───────────────┘          └───────────────┘
       [DETERMINISTIC]        [DETERMINISTIC]           [DETERMINISTIC]
```

---

## File Structure

Create these new files/folders in the SpeechToBlender directory:

```
SpeechToBlender/
├── __init__.py                    # MODIFY: Add new RPC methods
├── voice_to_blender.py            # MODIFY: Slim down, call orchestrator
│
├── agents/                        # NEW FOLDER
│   ├── __init__.py                # Exports: Orchestrator, TaskSpec
│   ├── orchestrator.py            # Main flow coordinator
│   ├── language_translator.py     # Intent parsing, reference resolution
│   ├── code_generator.py          # Blender Python code generation
│   └── semantic_evaluator.py      # Visual verification with VLM
│
├── analyzers/                     # NEW FOLDER
│   ├── __init__.py                # Exports: FocusStack, GeometricAnalyzer
│   ├── focus_stack.py             # Context/reference tracking
│   └── relationships.py           # Multi-object spatial analysis
│
└── prompts/                       # NEW FOLDER
    ├── __init__.py
    └── templates.py               # All LLM prompt templates
```

---

## Data Structures

### TaskSpec (shared between agents)

Create `agents/__init__.py`:

```python
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
```

---

## Implementation Details

### 1. Focus Stack (analyzers/focus_stack.py)

This tracks what the user is referring to when they say "it", "this", "that".

```python
"""
Focus Stack - Tracks context for reference resolution.

The focus stack maintains awareness of:
- Currently selected objects
- Recently created/modified objects  
- Objects mentioned in recent commands
- Command history with object associations

This enables resolving ambiguous references like "make it smoother" 
to specific objects without asking the user.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class CommandRecord:
    """Record of a single voice command and its context."""
    transcript: str
    target_objects: List[str]
    operations_performed: List[str]
    timestamp: float
    success: bool

class FocusStack:
    """
    Maintains focus context for reference resolution.
    
    Priority order for resolving "it"/"this"/"that":
    1. Explicitly selected object(s) in Blender
    2. Most recently modified object
    3. Most recently created object
    4. Object mentioned in previous command
    5. Active object (even if not selected)
    """
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        
        # Focus tracking
        self._last_created: Optional[str] = None
        self._last_modified: Optional[str] = None
        self._last_mentioned: Optional[str] = None
        self._command_history: List[CommandRecord] = []
        
        # Object name -> last interaction timestamp
        self._interaction_times: Dict[str, float] = {}
    
    def record_creation(self, object_name: str):
        """Record that an object was created."""
        self._last_created = object_name
        self._last_modified = object_name
        self._interaction_times[object_name] = time.time()
    
    def record_modification(self, object_name: str):
        """Record that an object was modified."""
        self._last_modified = object_name
        self._interaction_times[object_name] = time.time()
    
    def record_mention(self, object_name: str):
        """Record that an object was explicitly mentioned by user."""
        self._last_mentioned = object_name
        self._interaction_times[object_name] = time.time()
    
    def record_command(self, transcript: str, target_objects: List[str], 
                       operations: List[str], success: bool):
        """Record a completed command."""
        record = CommandRecord(
            transcript=transcript,
            target_objects=target_objects,
            operations_performed=operations,
            timestamp=time.time(),
            success=success
        )
        self._command_history.append(record)
        
        # Trim history
        if len(self._command_history) > self.max_history:
            self._command_history = self._command_history[-self.max_history:]
        
        # Update last mentioned
        if target_objects:
            self._last_mentioned = target_objects[0]
            for obj in target_objects:
                self._interaction_times[obj] = time.time()
    
    def resolve_reference(self, 
                          selected_objects: List[str],
                          active_object: Optional[str]) -> Dict[str, Any]:
        """
        Resolve ambiguous references to specific objects.
        
        Returns dict with:
        - resolved_objects: List of object names
        - resolution_method: How we resolved it
        - confidence: How confident we are (0-1)
        """
        
        # Priority 1: Explicit selection (user clicked on something)
        if selected_objects:
            return {
                "resolved_objects": selected_objects,
                "resolution_method": "explicit_selection",
                "confidence": 1.0,
                "explanation": f"User has {len(selected_objects)} object(s) selected"
            }
        
        # Priority 2: Last modified (most relevant for "make it X")
        if self._last_modified:
            return {
                "resolved_objects": [self._last_modified],
                "resolution_method": "last_modified",
                "confidence": 0.9,
                "explanation": f"Most recently modified: {self._last_modified}"
            }
        
        # Priority 3: Last created
        if self._last_created:
            return {
                "resolved_objects": [self._last_created],
                "resolution_method": "last_created", 
                "confidence": 0.85,
                "explanation": f"Most recently created: {self._last_created}"
            }
        
        # Priority 4: Last mentioned in command
        if self._last_mentioned:
            return {
                "resolved_objects": [self._last_mentioned],
                "resolution_method": "last_mentioned",
                "confidence": 0.8,
                "explanation": f"Last mentioned in command: {self._last_mentioned}"
            }
        
        # Priority 5: Active object
        if active_object:
            return {
                "resolved_objects": [active_object],
                "resolution_method": "active_object",
                "confidence": 0.7,
                "explanation": f"Falling back to active object: {active_object}"
            }
        
        # No resolution possible
        return {
            "resolved_objects": [],
            "resolution_method": "none",
            "confidence": 0.0,
            "explanation": "Could not resolve reference - no context available"
        }
    
    def get_recent_context(self, n: int = 5) -> List[CommandRecord]:
        """Get the N most recent commands."""
        return self._command_history[-n:]
    
    def get_focus_summary(self, selected: List[str], active: Optional[str]) -> Dict:
        """Get complete focus state for debugging/prompts."""
        return {
            "selected_objects": selected,
            "active_object": active,
            "last_created": self._last_created,
            "last_modified": self._last_modified,
            "last_mentioned": self._last_mentioned,
            "recent_commands": [
                {"transcript": c.transcript, "targets": c.target_objects}
                for c in self._command_history[-5:]
            ]
        }
    
    def clear(self):
        """Clear all focus state (e.g., on new session)."""
        self._last_created = None
        self._last_modified = None
        self._last_mentioned = None
        self._command_history = []
        self._interaction_times = {}


# Global instance (initialized by orchestrator)
_focus_stack: Optional[FocusStack] = None

def get_focus_stack() -> FocusStack:
    global _focus_stack
    if _focus_stack is None:
        _focus_stack = FocusStack()
    return _focus_stack
```

Create `analyzers/__init__.py`:

```python
"""Analyzers module - deterministic analysis components."""

from analyzers.focus_stack import FocusStack, get_focus_stack, CommandRecord

__all__ = ['FocusStack', 'get_focus_stack', 'CommandRecord']
```

---

### 2. Prompt Templates (prompts/templates.py)

```python
"""
LLM Prompt Templates

All prompts are centralized here for easy tuning and version control.
"""

LANGUAGE_TRANSLATOR_PROMPT = """You are a Language Translator for a 3D modeling voice assistant.

Your ONLY job is to:
1. Understand what the user wants to do
2. Resolve what objects they're referring to
3. Infer what Blender operations would achieve their goal
4. Output a structured JSON specification

You have encyclopedic knowledge of 3D objects. When users reference real objects like 
"Echo Dot", "iPhone", "coffee mug", "Nike shoe" - you know their typical shape, 
proportions, surface qualities, and construction.

When users give vague commands like "make it smoother" or "make it look more like X":
- Infer the TARGET PROPERTIES from your knowledge of X
- Compare to what you know about the current object
- Determine what operations would bridge the gap
- Be SPECIFIC about what changes are needed

CURRENT SCENE CONTEXT:
{context_summary}
{resolution_note}

OUTPUT FORMAT (JSON only, no markdown):
{{
    "task_type": "create|modify|delete|query|navigate",
    "user_intent": "Brief description of what user wants",
    "target_objects": ["Object1", "Object2"],
    "target_concept": "Real-world object name if referenced, else null",
    "target_properties": {{
        "shape": "cylindrical|cubic|spherical|organic|etc",
        "proportions": "description like 'squat, diameter >> height'",
        "surface_quality": "smooth|rough|textured|matte|glossy",
        "edge_treatment": "sharp|beveled|rounded|chamfered",
        "size": {{"width": 2.0, "height": 0.5}},
        "custom": {{}}
    }},
    "inferred_operations": [
        {{
            "action": "bevel|scale|subdivide|smooth|extrude|etc",
            "target": "object_name or 'selected' or 'active'",
            "parameters": {{"width": 0.1, "segments": 3}},
            "reason": "Why this operation helps achieve the goal",
            "priority": 1
        }}
    ],
    "confidence": 0.9,
    "ambiguities": ["List any unclear aspects"]
}}

RULES:
1. Always output valid JSON, no markdown code fences
2. If user says "it"/"this"/"that", use the resolved reference from context
3. If you can't determine what object, set confidence low and note in ambiguities
4. Infer operations even for vague requests - use your knowledge of the target concept
5. For "make it look like X", provide detailed target_properties based on X
6. Operations should be Blender-appropriate (bevel, subdivide, scale, etc.)
"""


CODE_GENERATOR_PROMPT = """You are a Blender Python Code Generator.

You receive a structured TaskSpec and generate executable Blender Python code.

TASK SPECIFICATION:
{task_spec_json}

CURRENT GEOMETRIC STATE:
{geometry_json}

AVAILABLE OPERATIONS REFERENCE:
{blender_api_examples}

OUTPUT FORMAT:
Return ONLY a JSON object with this structure:
{{
    "code": "# Blender Python code here\\nimport bpy\\n...",
    "operations_summary": ["List of operations this code performs"],
    "expected_changes": ["What should change after execution"],
    "requires_mode": "OBJECT|EDIT|SCULPT",
    "warnings": ["Any potential issues"]
}}

RULES:
1. Generate complete, executable Blender Python code
2. Always include proper context management (mode switching if needed)
3. Use the exact object names from the TaskSpec
4. Include error handling for missing objects
5. Use bpy.ops for standard operations, bpy.data for direct manipulation
6. Comment the code explaining each step
7. Prefer modifiers over destructive mesh edits when appropriate
"""


SEMANTIC_EVALUATOR_PROMPT = """You are a Semantic Evaluator for 3D modeling.

Your job is to look at a rendered image of a 3D model and evaluate whether it matches 
the user's original intent.

ORIGINAL USER REQUEST:
{user_intent}

TARGET CONCEPT (if any):
{target_concept}

TARGET PROPERTIES:
{target_properties}

OPERATIONS THAT WERE PERFORMED:
{operations_performed}

Now examine the image and answer these specific questions:

{evaluation_questions}

OUTPUT FORMAT (JSON only):
{{
    "overall_match_score": 0.8,
    "question_answers": {{
        "question_text": {{
            "answer": "yes|no|partial",
            "explanation": "Brief explanation"
        }}
    }},
    "issues_found": [
        "List specific problems with the result"
    ],
    "suggested_refinements": [
        {{
            "action": "operation_name",
            "target": "object_name",
            "parameters": {{}},
            "reason": "Why this would improve the result"
        }}
    ],
    "should_retry": true|false
}}
"""


def build_evaluation_questions(target_concept: str, target_properties: dict) -> str:
    """Build specific evaluation questions based on the target."""
    
    questions = []
    
    if target_concept:
        questions.append(
            f"1. Does this look like a {target_concept}? "
            f"Consider the overall shape and recognizability."
        )
    
    if target_properties:
        if target_properties.get("shape"):
            questions.append(
                f"2. Is the shape {target_properties['shape']}? "
                f"Evaluate the basic form."
            )
        
        if target_properties.get("edge_treatment"):
            questions.append(
                f"3. Are the edges {target_properties['edge_treatment']}? "
                f"Look at edge sharpness/roundness."
            )
        
        if target_properties.get("surface_quality"):
            questions.append(
                f"4. Is the surface {target_properties['surface_quality']}? "
                f"Evaluate surface smoothness/roughness."
            )
        
        if target_properties.get("proportions"):
            questions.append(
                f"5. Are the proportions correct ({target_properties['proportions']})? "
                f"Compare height/width/depth ratios."
            )
    
    if not questions:
        questions = [
            "1. Does the result match what the user asked for?",
            "2. Are there any obvious issues with the geometry?",
            "3. Would the user be satisfied with this result?"
        ]
    
    return "\n".join(questions)
```

Create `prompts/__init__.py`:

```python
"""Prompts module - centralized LLM prompt templates."""

from prompts.templates import (
    LANGUAGE_TRANSLATOR_PROMPT,
    CODE_GENERATOR_PROMPT, 
    SEMANTIC_EVALUATOR_PROMPT,
    build_evaluation_questions
)

__all__ = [
    'LANGUAGE_TRANSLATOR_PROMPT',
    'CODE_GENERATOR_PROMPT',
    'SEMANTIC_EVALUATOR_PROMPT',
    'build_evaluation_questions'
]
```

---

### 3. Language Translator (agents/language_translator.py)

```python
"""
Language Translator Agent

Responsibility: Convert natural language to structured TaskSpec
- Parse user intent
- Resolve ambiguous references using FocusStack
- Infer operations needed (using LLM's encyclopedic knowledge)
- Output structured TaskSpec for Code Generator

This agent does NOT:
- Generate Blender code
- Execute anything
- Reason about specific geometry (that's the Geometric Analyzer's job)
"""

import json
import re
from typing import Optional, Dict, Any, List

from openai import OpenAI

from agents import TaskSpec, TaskType, InferredOperation, TargetProperties
from analyzers.focus_stack import get_focus_stack
from prompts.templates import LANGUAGE_TRANSLATOR_PROMPT


class LanguageTranslator:
    """
    Translates natural language commands into structured TaskSpecs.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.focus_stack = get_focus_stack()
    
    def translate(self,
                  transcript: str,
                  scene_context: Dict[str, Any],
                  selected_objects: List[str],
                  active_object: Optional[str],
                  modeling_context: Optional[Dict] = None) -> TaskSpec:
        """
        Translate a voice transcript into a structured TaskSpec.
        
        Args:
            transcript: Raw voice transcript
            scene_context: Output from analyze_scene() RPC
            selected_objects: List of currently selected object names
            active_object: Name of active object (or None)
            modeling_context: Output from get_modeling_context() RPC
            
        Returns:
            TaskSpec with parsed intent and inferred operations
        """
        
        # Step 1: Check for ambiguous references and resolve them
        has_ambiguous_ref = self._has_ambiguous_reference(transcript)
        resolution = None
        
        if has_ambiguous_ref:
            resolution = self.focus_stack.resolve_reference(
                selected_objects, active_object
            )
        
        # Step 2: Build context summary for LLM
        context_summary = self._build_context_summary(
            scene_context, modeling_context, resolution
        )
        
        # Step 3: Call LLM for intent parsing and operation inference
        llm_response = self._call_llm(transcript, context_summary, resolution)
        
        # Step 4: Parse LLM response into TaskSpec
        task_spec = self._parse_response(transcript, llm_response, resolution)
        
        return task_spec
    
    def _has_ambiguous_reference(self, transcript: str) -> bool:
        """Check if transcript contains ambiguous references."""
        patterns = [
            r'\b(it|this|that|them|these|those)\b',
            r'\b(the object|the mesh|the model)\b',
            r'\b(make it|move it|scale it|rotate it)\b',
        ]
        text_lower = transcript.lower()
        return any(re.search(p, text_lower) for p in patterns)
    
    def _build_context_summary(self,
                                scene_context: Dict,
                                modeling_context: Optional[Dict],
                                resolution: Optional[Dict]) -> str:
        """Build a concise context summary for the LLM."""
        
        parts = []
        
        # Scene overview
        if scene_context:
            summary = scene_context.get("scene_summary", {})
            parts.append(f"Scene: {summary.get('total_objects', 0)} objects")
            
            # List objects briefly
            objects = scene_context.get("objects", [])[:10]  # Max 10
            if objects:
                obj_list = ", ".join(f"{o['name']}({o['type']})" for o in objects)
                parts.append(f"Objects: {obj_list}")
        
        # Current selection/active
        if modeling_context:
            selected = modeling_context.get("selected_objects", [])
            if selected:
                names = [o["name"] for o in selected]
                parts.append(f"Selected: {', '.join(names)}")
            
            active = modeling_context.get("active_object")
            if active:
                parts.append(f"Active: {active['name']} ({active.get('type', 'UNKNOWN')})")
                
            # Modifiers on active
            mods = modeling_context.get("modifiers", [])
            if mods:
                mod_names = [m["name"] for m in mods]
                parts.append(f"Active object modifiers: {', '.join(mod_names)}")
        
        # Reference resolution
        if resolution and resolution.get("resolved_objects"):
            parts.append(
                f"Reference '{resolution['resolution_method']}' resolved to: "
                f"{', '.join(resolution['resolved_objects'])}"
            )
        
        return "\n".join(parts)
    
    def _call_llm(self,
                  transcript: str,
                  context_summary: str,
                  resolution: Optional[Dict]) -> Dict:
        """Call LLM to parse intent and infer operations."""
        
        # Build the prompt
        resolution_note = ""
        if resolution and resolution.get("resolved_objects"):
            resolution_note = (
                f"\n\nIMPORTANT: When the user says 'it'/'this'/'that', they mean: "
                f"{', '.join(resolution['resolved_objects'])} "
                f"(resolved via {resolution['resolution_method']}, "
                f"confidence: {resolution['confidence']})"
            )
        
        prompt = LANGUAGE_TRANSLATOR_PROMPT.format(
            context_summary=context_summary,
            resolution_note=resolution_note
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _parse_response(self,
                        transcript: str,
                        llm_response: Dict,
                        resolution: Optional[Dict]) -> TaskSpec:
        """Parse LLM response into TaskSpec dataclass."""
        
        # Parse task type
        task_type_str = llm_response.get("task_type", "modify").lower()
        task_type_map = {
            "create": TaskType.CREATE,
            "modify": TaskType.MODIFY,
            "delete": TaskType.DELETE,
            "query": TaskType.QUERY,
            "navigate": TaskType.NAVIGATE,
        }
        task_type = task_type_map.get(task_type_str, TaskType.MODIFY)
        
        # Parse target objects
        target_objects = llm_response.get("target_objects", [])
        if not target_objects and resolution:
            target_objects = resolution.get("resolved_objects", [])
        
        # Parse target properties
        target_props_raw = llm_response.get("target_properties", {})
        target_properties = None
        if target_props_raw:
            target_properties = TargetProperties(
                shape=target_props_raw.get("shape"),
                proportions=target_props_raw.get("proportions"),
                surface_quality=target_props_raw.get("surface_quality"),
                edge_treatment=target_props_raw.get("edge_treatment"),
                size=target_props_raw.get("size"),
                custom=target_props_raw.get("custom", {})
            )
        
        # Parse inferred operations
        ops_raw = llm_response.get("inferred_operations", [])
        inferred_ops = []
        for i, op in enumerate(ops_raw):
            inferred_ops.append(InferredOperation(
                action=op.get("action", "unknown"),
                target=op.get("target", "active"),
                parameters=op.get("parameters", {}),
                reason=op.get("reason", ""),
                priority=op.get("priority", i + 1)
            ))
        
        # Build TaskSpec
        return TaskSpec(
            raw_transcript=transcript,
            task_type=task_type,
            user_intent=llm_response.get("user_intent", transcript),
            target_objects=target_objects,
            referent_resolution_method=resolution["resolution_method"] if resolution else "direct",
            target_concept=llm_response.get("target_concept"),
            target_properties=target_properties,
            inferred_operations=inferred_ops,
            scene_context_summary=llm_response.get("scene_context_used", ""),
            current_object_summary=llm_response.get("object_context_used", ""),
            confidence=llm_response.get("confidence", 0.8),
            ambiguities=llm_response.get("ambiguities", [])
        )
```

---

### 4. Code Generator (agents/code_generator.py)

```python
"""
Code Generator Agent

Responsibility: Generate executable Blender Python code from TaskSpec
- Receives structured TaskSpec from Language Translator
- Receives geometric state from Analyzer
- Generates Blender Python code
- Does NOT execute the code (that's the RPC bridge's job)
"""

import json
from typing import Dict, Any, Optional, List

from openai import OpenAI

from agents import TaskSpec, ExecutionResult
from prompts.templates import CODE_GENERATOR_PROMPT


class CodeGenerator:
    """
    Generates Blender Python code from structured TaskSpecs.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self._blender_api_cache: Dict[str, str] = {}
    
    def generate(self,
                 task_spec: TaskSpec,
                 geometry_state: Dict[str, Any],
                 mesh_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate Blender Python code for the given TaskSpec.
        
        Args:
            task_spec: Structured task from Language Translator
            geometry_state: Current scene/object state from Geometric Analyzer
            mesh_analysis: Detailed mesh analysis if available
            
        Returns:
            Dict with 'code', 'operations_summary', 'expected_changes', etc.
        """
        
        # Build geometry context
        geometry_json = self._build_geometry_context(geometry_state, mesh_analysis)
        
        # Get relevant Blender API examples
        api_examples = self._get_relevant_api_examples(task_spec)
        
        # Build TaskSpec JSON for prompt
        task_spec_json = self._serialize_task_spec(task_spec)
        
        # Call LLM
        prompt = CODE_GENERATOR_PROMPT.format(
            task_spec_json=task_spec_json,
            geometry_json=geometry_json,
            blender_api_examples=api_examples
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the Blender Python code for this task."}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _build_geometry_context(self,
                                 geometry_state: Dict,
                                 mesh_analysis: Optional[Dict]) -> str:
        """Build concise geometry context for the prompt."""
        
        parts = []
        
        # Scene state
        if geometry_state:
            parts.append(f"Objects in scene: {geometry_state.get('object_count', 'unknown')}")
            
            # Selected objects with details
            selected = geometry_state.get("selected_objects", [])
            if selected:
                for obj in selected[:5]:  # Max 5
                    parts.append(
                        f"- {obj['name']}: {obj['type']} at {obj.get('location', 'unknown')}"
                    )
            
            # Active object
            active = geometry_state.get("active_object")
            if active:
                parts.append(f"Active object: {active['name']} (mode: {active.get('mode', 'OBJECT')})")
        
        # Mesh details
        if mesh_analysis and not mesh_analysis.get("error"):
            parts.append(f"\nMesh Analysis:")
            parts.append(f"- Vertices: {mesh_analysis.get('vertex_count', 'N/A')}")
            parts.append(f"- Faces: {mesh_analysis.get('face_count', 'N/A')}")
            parts.append(f"- Bounding box: {mesh_analysis.get('bounding_box', 'N/A')}")
            
            topology = mesh_analysis.get("topology", {})
            if topology:
                parts.append(f"- Is manifold: {topology.get('is_manifold', 'unknown')}")
                parts.append(f"- Has ngons: {topology.get('has_ngons', 'unknown')}")
        
        return "\n".join(parts)
    
    def _get_relevant_api_examples(self, task_spec: TaskSpec) -> str:
        """Get relevant Blender API examples for the operations."""
        
        examples = []
        
        for op in task_spec.inferred_operations:
            action = op.action.lower()
            
            if action == "bevel":
                examples.append("""
# Bevel edges
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.bevel(offset=0.1, segments=3, affect='EDGES')
bpy.ops.object.mode_set(mode='OBJECT')
""")
            elif action == "subdivide":
                examples.append("""
# Subdivide mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.subdivide(number_cuts=2)
bpy.ops.object.mode_set(mode='OBJECT')
""")
            elif action == "smooth":
                examples.append("""
# Smooth shading + subdivision surface
bpy.ops.object.shade_smooth()
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subdivision"].levels = 2
""")
            elif action == "scale":
                examples.append("""
# Scale object
bpy.ops.transform.resize(value=(1.0, 1.0, 0.5))  # Scale Z to 50%
# Or set scale directly:
bpy.context.object.scale = (1.0, 1.0, 0.5)
""")
            elif action == "extrude":
                examples.append("""
# Extrude faces
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, 1)})
bpy.ops.object.mode_set(mode='OBJECT')
""")
        
        if not examples:
            examples.append("# Use standard Blender operators (bpy.ops.*)")
        
        return "\n".join(examples)
    
    def _serialize_task_spec(self, task_spec: TaskSpec) -> str:
        """Serialize TaskSpec to JSON for the prompt."""
        
        return json.dumps({
            "task_type": task_spec.task_type.value,
            "user_intent": task_spec.user_intent,
            "target_objects": task_spec.target_objects,
            "target_concept": task_spec.target_concept,
            "target_properties": {
                "shape": task_spec.target_properties.shape if task_spec.target_properties else None,
                "proportions": task_spec.target_properties.proportions if task_spec.target_properties else None,
                "surface_quality": task_spec.target_properties.surface_quality if task_spec.target_properties else None,
                "edge_treatment": task_spec.target_properties.edge_treatment if task_spec.target_properties else None,
            } if task_spec.target_properties else None,
            "inferred_operations": [
                {
                    "action": op.action,
                    "target": op.target,
                    "parameters": op.parameters,
                    "reason": op.reason,
                    "priority": op.priority
                }
                for op in task_spec.inferred_operations
            ]
        }, indent=2)
```

---

### 5. Semantic Evaluator (agents/semantic_evaluator.py)

```python
"""
Semantic Evaluator Agent

Responsibility: Verify that execution results match user intent
- Uses VLM (GPT-4o with vision) to analyze rendered output
- Asks specific questions based on target properties
- Provides structured feedback for refinement
"""

import json
import base64
from typing import Dict, Any, List, Optional

from openai import OpenAI

from agents import TaskSpec, EvaluationResult, InferredOperation
from prompts.templates import SEMANTIC_EVALUATOR_PROMPT, build_evaluation_questions


class SemanticEvaluator:
    """
    Evaluates execution results against user intent using vision.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def evaluate(self,
                 task_spec: TaskSpec,
                 screenshot_base64: str,
                 operations_performed: List[str]) -> EvaluationResult:
        """
        Evaluate whether the result matches the user's intent.
        
        Args:
            task_spec: Original task specification
            screenshot_base64: Base64-encoded screenshot of result
            operations_performed: List of operations that were executed
            
        Returns:
            EvaluationResult with scores, issues, and suggestions
        """
        
        # Build evaluation questions
        target_props = None
        if task_spec.target_properties:
            target_props = {
                "shape": task_spec.target_properties.shape,
                "proportions": task_spec.target_properties.proportions,
                "surface_quality": task_spec.target_properties.surface_quality,
                "edge_treatment": task_spec.target_properties.edge_treatment,
            }
        
        questions = build_evaluation_questions(
            task_spec.target_concept or "",
            target_props or {}
        )
        
        # Build prompt
        prompt = SEMANTIC_EVALUATOR_PROMPT.format(
            user_intent=task_spec.user_intent,
            target_concept=task_spec.target_concept or "None specified",
            target_properties=json.dumps(target_props, indent=2) if target_props else "None specified",
            operations_performed=", ".join(operations_performed),
            evaluation_questions=questions
        )
        
        # Call VLM with image
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Evaluate this 3D model against the criteria above."
                        }
                    ]
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Parse into EvaluationResult
        suggested_refinements = []
        for ref in result.get("suggested_refinements", []):
            suggested_refinements.append(InferredOperation(
                action=ref.get("action", "unknown"),
                target=ref.get("target", "active"),
                parameters=ref.get("parameters", {}),
                reason=ref.get("reason", "")
            ))
        
        return EvaluationResult(
            semantic_match_score=result.get("overall_match_score", 0.5),
            questions_asked=result.get("question_answers", {}),
            issues_found=result.get("issues_found", []),
            suggested_refinements=suggested_refinements,
            should_retry=result.get("should_retry", False)
        )
```

---

### 6. Orchestrator (agents/orchestrator.py)

```python
"""
Orchestrator - Coordinates the multi-agent flow.

This is the main entry point that voice_to_blender.py calls.
It coordinates:
1. Focus Stack for reference resolution
2. Language Translator for intent parsing
3. Code Generator for Blender code
4. Semantic Evaluator for verification (optional)
"""

import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict

from openai import OpenAI

from agents import TaskSpec, ExecutionResult, EvaluationResult
from agents.language_translator import LanguageTranslator
from agents.code_generator import CodeGenerator
from agents.semantic_evaluator import SemanticEvaluator
from analyzers.focus_stack import get_focus_stack, FocusStack


class Orchestrator:
    """
    Coordinates the multi-agent pipeline for voice command processing.
    """
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize agents
        self.translator = LanguageTranslator(self.client)
        self.code_generator = CodeGenerator(self.client)
        self.evaluator = SemanticEvaluator(self.client)
        
        # Focus stack (shared state)
        self.focus_stack = get_focus_stack()
        
        # Configuration
        self.max_refinement_attempts = 3
        self.enable_semantic_evaluation = True
    
    def process_command(self,
                        transcript: str,
                        rpc_client,
                        include_evaluation: bool = True) -> Dict[str, Any]:
        """
        Process a voice command through the full agent pipeline.
        
        Args:
            transcript: Raw voice transcript
            rpc_client: XML-RPC client for Blender communication
            include_evaluation: Whether to run semantic evaluation
            
        Returns:
            Dict with results from each stage
        """
        
        result = {
            "transcript": transcript,
            "success": False,
            "task_spec": None,
            "code_generated": None,
            "execution_result": None,
            "evaluation_result": None,
            "errors": []
        }
        
        try:
            # ===== STAGE 1: Gather Context =====
            print("[Orchestrator] Stage 1: Gathering context...")
            
            modeling_context = rpc_client.get_modeling_context()
            scene_context = rpc_client.analyze_scene()
            mesh_analysis = None
            
            # Get mesh analysis if there's an active mesh object
            active = modeling_context.get("active_object")
            if active and active.get("type") == "MESH":
                try:
                    mesh_analysis = rpc_client.analyze_current_mesh()
                except Exception as e:
                    print(f"[Orchestrator] Mesh analysis failed: {e}")
            
            # Extract selection info
            selected_objects = [
                obj["name"] for obj in modeling_context.get("selected_objects", [])
            ]
            active_object = active["name"] if active else None
            
            # ===== STAGE 2: Language Translation =====
            print("[Orchestrator] Stage 2: Translating language...")
            
            task_spec = self.translator.translate(
                transcript=transcript,
                scene_context=scene_context,
                selected_objects=selected_objects,
                active_object=active_object,
                modeling_context=modeling_context
            )
            result["task_spec"] = asdict(task_spec)
            
            print(f"[Orchestrator] TaskSpec: {task_spec.user_intent}")
            print(f"[Orchestrator] Target objects: {task_spec.target_objects}")
            print(f"[Orchestrator] Operations: {[op.action for op in task_spec.inferred_operations]}")
            
            # Check confidence
            if task_spec.confidence < 0.5:
                result["errors"].append(
                    f"Low confidence ({task_spec.confidence}): {task_spec.ambiguities}"
                )
            
            # ===== STAGE 3: Code Generation =====
            print("[Orchestrator] Stage 3: Generating code...")
            
            code_result = self.code_generator.generate(
                task_spec=task_spec,
                geometry_state=modeling_context,
                mesh_analysis=mesh_analysis
            )
            result["code_generated"] = code_result
            
            # ===== STAGE 4: Execution =====
            print("[Orchestrator] Stage 4: Executing code...")
            
            code = code_result.get("code", "")
            if code:
                try:
                    exec_response = rpc_client.execute({"code": code})
                    result["execution_result"] = exec_response
                    
                    # Update focus stack
                    if exec_response.get("success", True):
                        for obj_name in task_spec.target_objects:
                            self.focus_stack.record_modification(obj_name)
                        self.focus_stack.record_command(
                            transcript=transcript,
                            target_objects=task_spec.target_objects,
                            operations=[op.action for op in task_spec.inferred_operations],
                            success=True
                        )
                except Exception as e:
                    result["errors"].append(f"Execution failed: {e}")
                    result["execution_result"] = {"success": False, "error": str(e)}
            
            # ===== STAGE 5: Semantic Evaluation (Optional) =====
            if include_evaluation and self.enable_semantic_evaluation:
                print("[Orchestrator] Stage 5: Semantic evaluation...")
                
                try:
                    screenshot = rpc_client.capture_viewport_screenshot()
                    
                    if screenshot and screenshot.get("image_base64"):
                        eval_result = self.evaluator.evaluate(
                            task_spec=task_spec,
                            screenshot_base64=screenshot["image_base64"],
                            operations_performed=[op.action for op in task_spec.inferred_operations]
                        )
                        result["evaluation_result"] = asdict(eval_result)
                        
                        if eval_result.should_retry and eval_result.suggested_refinements:
                            print("[Orchestrator] Evaluation suggests refinements...")
                except Exception as e:
                    print(f"[Orchestrator] Evaluation failed: {e}")
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(str(e))
            import traceback
            traceback.print_exc()
        
        return result


# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None

def get_orchestrator(api_key: str) -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(api_key)
    return _orchestrator
```

---

### 7. RPC Additions (add to __init__.py)

Add these new RPC methods inside the `_server_loop` function where other methods are registered:

```python
# Add near the top of the file with other globals
import time

_FOCUS_STACK_DATA = {
    "last_created": None,
    "last_modified": None,
    "last_mentioned": None,
    "command_history": [],
}

# Add these functions inside _server_loop, before server.serve_forever()

def get_focus_context():
    """RPC method: Get focus stack for reference resolution."""
    global _FOCUS_STACK_DATA
    context = bpy.context
    
    selected = [obj.name for obj in context.view_layer.objects.selected]
    active = context.view_layer.objects.active.name if context.view_layer.objects.active else None
    
    return {
        "selected_objects": selected,
        "active_object": active,
        "last_created": _FOCUS_STACK_DATA.get("last_created"),
        "last_modified": _FOCUS_STACK_DATA.get("last_modified"),
        "last_mentioned": _FOCUS_STACK_DATA.get("last_mentioned"),
        "recent_commands": _FOCUS_STACK_DATA.get("command_history", [])[-5:],
    }

def update_focus_stack(event_type: str, object_name: str, command_text: str = ""):
    """RPC method: Update focus stack after operations."""
    global _FOCUS_STACK_DATA
    
    if event_type == "created":
        _FOCUS_STACK_DATA["last_created"] = object_name
        _FOCUS_STACK_DATA["last_modified"] = object_name
    elif event_type == "modified":
        _FOCUS_STACK_DATA["last_modified"] = object_name
    elif event_type == "mentioned":
        _FOCUS_STACK_DATA["last_mentioned"] = object_name
    
    if command_text:
        _FOCUS_STACK_DATA["command_history"].append({
            "command": command_text,
            "target": object_name,
            "timestamp": time.time()
        })
        _FOCUS_STACK_DATA["command_history"] = _FOCUS_STACK_DATA["command_history"][-20:]
    
    return "OK"

def analyze_object_relationships():
    """RPC method: Analyze spatial relationships between objects."""
    context = bpy.context
    scene = context.scene
    
    objects = list(scene.objects)
    relationships = []
    
    for i, obj1 in enumerate(objects):
        for obj2 in objects[i+1:]:
            loc1 = obj1.location
            loc2 = obj2.location
            distance = (loc1 - loc2).length
            
            rel = {
                "objects": [obj1.name, obj2.name],
                "distance": round(distance, 3),
                "relations": []
            }
            
            # Vertical alignment
            if abs(loc1.x - loc2.x) < 0.1 and abs(loc1.y - loc2.y) < 0.1:
                if loc1.z > loc2.z:
                    rel["relations"].append(f"{obj1.name} above {obj2.name}")
                else:
                    rel["relations"].append(f"{obj1.name} below {obj2.name}")
            
            # Concentric
            if abs(loc1.x - loc2.x) < 0.05 and abs(loc1.y - loc2.y) < 0.05:
                rel["relations"].append("concentric_z")
            
            # Adjacent
            max_dim = max(
                obj1.dimensions.length if hasattr(obj1, 'dimensions') else 1,
                obj2.dimensions.length if hasattr(obj2, 'dimensions') else 1
            )
            if distance < max_dim * 1.5:
                rel["relations"].append("adjacent")
            
            if rel["relations"] or distance < 2.0:
                relationships.append(rel)
    
    # Build hierarchy
    hierarchy = {}
    for obj in objects:
        if obj.parent is None:
            hierarchy[obj.name] = _build_hierarchy(obj)
    
    return {
        "hierarchy": hierarchy,
        "relationships": relationships
    }

def _build_hierarchy(obj):
    """Helper to build hierarchy tree."""
    return {
        "name": obj.name,
        "type": obj.type,
        "location": tuple(obj.location),
        "children": [_build_hierarchy(child) for child in obj.children]
    }

# Register the new methods (add near other server.register_function calls)
server.register_function(get_focus_context, "get_focus_context")
server.register_function(update_focus_stack, "update_focus_stack")
server.register_function(analyze_object_relationships, "analyze_object_relationships")
```

---

### 8. Integration with voice_to_blender.py

Add this integration code to use the orchestrator:

```python
# Add at top of voice_to_blender.py with other imports
import sys
import os

# Add the SpeechToBlender directory to path for imports
STB_DIR = os.path.dirname(os.path.abspath(__file__))
if STB_DIR not in sys.path:
    sys.path.insert(0, STB_DIR)

# Import orchestrator (after path setup)
try:
    from agents.orchestrator import get_orchestrator
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Multi-agent system not available: {e}")
    MULTI_AGENT_AVAILABLE = False

# Add configuration flag near other config
USE_MULTI_AGENT = True  # Set to False to use legacy GPT fallback


def process_with_orchestrator(transcript: str, rpc) -> bool:
    """Process command through the multi-agent orchestrator."""
    
    if not MULTI_AGENT_AVAILABLE:
        print("[Warning] Multi-agent system not available, falling back to legacy")
        return False
    
    try:
        api_key = rpc.get_openai_api_key()
        if not api_key:
            print("[Error] No OpenAI API key available")
            return False
            
        orchestrator = get_orchestrator(api_key)
        
        result = orchestrator.process_command(
            transcript=transcript,
            rpc_client=rpc,
            include_evaluation=USE_REACT_REASONING
        )
        
        if result["success"]:
            print(f"✅ Command processed successfully")
            if result.get("task_spec"):
                print(f"   Intent: {result['task_spec'].get('user_intent', 'N/A')}")
                print(f"   Targets: {result['task_spec'].get('target_objects', [])}")
            if result.get("evaluation_result"):
                score = result["evaluation_result"].get("semantic_match_score", "N/A")
                print(f"   Semantic match: {score}")
            return True
        else:
            print(f"❌ Command failed: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"❌ Orchestrator error: {e}")
        import traceback
        traceback.print_exc()
        return False


# In the main command processing loop, modify the GPT fallback section:
# Replace or augment the existing gpt_to_json fallback with:

# After local rules fail:
if USE_MULTI_AGENT and MULTI_AGENT_AVAILABLE:
    success = process_with_orchestrator(text, rpc)
    if success:
        continue  # Move to next command
    # If orchestrator failed, fall through to legacy GPT

# Existing GPT fallback (keep as backup)
# ... existing gpt_to_json code ...
```

---

## Configuration

Add to `config/config.json`:

```json
{
    "agents": {
        "enable_multi_agent": true,
        "enable_semantic_evaluation": true,
        "max_refinement_attempts": 3,
        "confidence_threshold": 0.5
    },
    "focus_stack": {
        "max_history": 20
    }
}
```

---

## Testing Checklist

### Phase 1: Focus Stack
- [ ] Test reference resolution with selected objects
- [ ] Test resolution with no selection (falls back to last_modified)
- [ ] Test command history tracking
- [ ] Verify RPC methods work from voice script

### Phase 2: Language Translator
- [ ] Test "add a cube" → CREATE task type
- [ ] Test "make it smoother" → resolves "it" and infers operations
- [ ] Test "make it look like an Echo Dot" → infers target properties

### Phase 3: Code Generator
- [ ] Test code generation for basic operations
- [ ] Verify generated code executes without errors
- [ ] Test error handling for invalid objects

### Phase 4: Semantic Evaluator
- [ ] Test screenshot capture works
- [ ] Test evaluation returns meaningful scores
- [ ] Test refinement suggestions

### Phase 5: End-to-End
- [ ] Full pipeline: voice → task spec → code → execution → evaluation
- [ ] Test with complex multi-step commands
- [ ] Test refinement loop

---

## Migration Strategy

1. **Keep existing code working**: The legacy `gpt_to_json()` remains as fallback
2. **Feature flag**: `USE_MULTI_AGENT` toggles between new and old system
3. **Gradual rollout**: Start with simple commands, expand to complex ones
4. **Monitor and tune**: Adjust prompts based on real-world usage

---

## Summary

This implementation adds:

| Component | Purpose | Location |
|-----------|---------|----------|
| Focus Stack | Resolve "it"/"this"/"that" | `analyzers/focus_stack.py` |
| Language Translator | Parse intent, infer operations | `agents/language_translator.py` |
| Code Generator | Generate Blender Python | `agents/code_generator.py` |
| Semantic Evaluator | Verify results visually | `agents/semantic_evaluator.py` |
| Orchestrator | Coordinate the pipeline | `agents/orchestrator.py` |
| Prompt Templates | Centralized prompts | `prompts/templates.py` |
| New RPC Methods | Focus tracking, relationships | `__init__.py` additions |
