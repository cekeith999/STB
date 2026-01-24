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
from dataclasses import dataclass

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
