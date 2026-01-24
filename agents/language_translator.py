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

from agents import TaskSpec, TaskType, InferredOperation, TargetProperties
from prompts.templates import LANGUAGE_TRANSLATOR_PROMPT

# Lazy import for focus stack
try:
    from analyzers.focus_stack import get_focus_stack
    FOCUS_STACK_AVAILABLE = True
except ImportError:
    FOCUS_STACK_AVAILABLE = False


class LanguageTranslator:
    """
    Translates natural language commands into structured TaskSpecs.
    """
    
    def __init__(self, openai_client):
        """
        Initialize Language Translator.
        
        Args:
            openai_client: OpenAI client instance (from openai import OpenAI)
        """
        self.client = openai_client
        self.focus_stack = None
        if FOCUS_STACK_AVAILABLE:
            try:
                self.focus_stack = get_focus_stack()
            except Exception as e:
                print(f"[LanguageTranslator] ⚠️ Focus Stack not available: {e}")
    
    def translate(self,
                  transcript: str,
                  scene_context: Optional[Dict[str, Any]] = None,
                  selected_objects: Optional[List[str]] = None,
                  active_object: Optional[str] = None,
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
        
        if has_ambiguous_ref and self.focus_stack:
            try:
                resolution = self.focus_stack.resolve_reference(
                    selected_objects or [],
                    active_object
                )
            except Exception as e:
                print(f"[LanguageTranslator] ⚠️ Focus stack resolution failed: {e}")
                resolution = None
        
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
                                scene_context: Optional[Dict],
                                modeling_context: Optional[Dict],
                                resolution: Optional[Dict]) -> str:
        """Build a concise context summary for the LLM."""
        
        parts = []
        
        # Scene overview
        if scene_context and not scene_context.get("error"):
            summary = scene_context.get("scene_summary", {})
            total_objects = summary.get("total_objects", 0)
            if total_objects > 0:
                parts.append(f"Scene: {total_objects} objects")
                
                # List objects briefly
                objects = scene_context.get("objects", [])[:10]  # Max 10
                if objects:
                    obj_list = ", ".join(f"{o.get('name', 'Unknown')}({o.get('type', 'UNKNOWN')})" for o in objects)
                    parts.append(f"Objects: {obj_list}")
        
        # Current selection/active
        if modeling_context and not modeling_context.get("error"):
            selected = modeling_context.get("selected_objects", [])
            if selected:
                names = [o.get("name", "Unknown") for o in selected]
                parts.append(f"Selected: {', '.join(names)}")
            
            active = modeling_context.get("active_object")
            if active:
                active_name = active.get("name", "Unknown")
                active_type = active.get("type", "UNKNOWN")
                parts.append(f"Active: {active_name} ({active_type})")
                
                # Modifiers on active
                mods = modeling_context.get("modifiers", [])
                if mods:
                    mod_names = [m.get("name", "Unknown") for m in mods]
                    parts.append(f"Active object modifiers: {', '.join(mod_names)}")
        
        # Reference resolution
        if resolution and resolution.get("resolved_objects"):
            parts.append(
                f"Reference '{resolution.get('resolution_method', 'unknown')}' resolved to: "
                f"{', '.join(resolution['resolved_objects'])}"
            )
        
        if not parts:
            return "Scene: Empty scene (no objects)"
        
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
                f"(resolved via {resolution.get('resolution_method', 'unknown')}, "
                f"confidence: {resolution.get('confidence', 0.0):.2f})"
            )
        
        prompt = LANGUAGE_TRANSLATOR_PROMPT.format(
            context_summary=context_summary,
            resolution_note=resolution_note
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                print("[LanguageTranslator] ⚠️ Empty response from LLM")
                return self._default_response(transcript)
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[LanguageTranslator] ⚠️ JSON parse error: {e}")
            print(f"[LanguageTranslator] Response was: {content[:200] if 'content' in locals() else 'N/A'}")
            return self._default_response(transcript)
        except Exception as e:
            print(f"[LanguageTranslator] ⚠️ LLM call failed: {e}")
            return self._default_response(transcript)
    
    def _default_response(self, transcript: str) -> Dict:
        """Generate a default response when LLM fails."""
        # Try to infer task type from transcript
        transcript_lower = transcript.lower()
        if any(word in transcript_lower for word in ["add", "create", "make", "new"]):
            task_type = "create"
        elif any(word in transcript_lower for word in ["delete", "remove"]):
            task_type = "delete"
        else:
            task_type = "modify"
        
        return {
            "task_type": task_type,
            "user_intent": transcript,
            "target_objects": [],
            "target_concept": None,
            "target_properties": {},
            "inferred_operations": [],
            "confidence": 0.5,
            "ambiguities": ["LLM call failed, using fallback interpretation"]
        }
    
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
            referent_resolution_method=resolution.get("resolution_method", "direct") if resolution else "direct",
            target_concept=llm_response.get("target_concept"),
            target_properties=target_properties,
            inferred_operations=inferred_ops,
            scene_context_summary=llm_response.get("scene_context_used", ""),
            current_object_summary=llm_response.get("object_context_used", ""),
            confidence=llm_response.get("confidence", 0.8),
            ambiguities=llm_response.get("ambiguities", [])
        )
