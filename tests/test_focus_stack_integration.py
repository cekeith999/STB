"""
Test script to demonstrate Focus Stack integration and performance comparison.

This script shows how Focus Stack improves reference resolution compared to
the current ad-hoc approach in prompts.
"""

import sys
import os

# Add repo root to path (parent of tests/)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from analyzers.focus_stack import get_focus_stack, FocusStack

def test_focus_stack_resolution():
    """Test Focus Stack reference resolution with various scenarios."""
    
    print("=" * 60)
    print("Focus Stack Reference Resolution Test")
    print("=" * 60)
    print()
    
    focus_stack = get_focus_stack()
    
    # Scenario 1: Create an object, then reference it
    print("Scenario 1: Create object, then reference 'it'")
    print("-" * 60)
    focus_stack.record_creation("Cube")
    resolution = focus_stack.resolve_reference(
        selected_objects=[],
        active_object="Cube"
    )
    print(f"Command: 'make it smoother'")
    print(f"Resolution: {resolution['resolved_objects']}")
    print(f"Method: {resolution['resolution_method']}")
    print(f"Confidence: {resolution['confidence']}")
    print(f"Explanation: {resolution['explanation']}")
    print()
    
    # Scenario 2: Modify an object, then reference it
    print("Scenario 2: Modify object, then reference 'it'")
    print("-" * 60)
    focus_stack.record_modification("Cube")
    focus_stack.record_creation("Sphere")  # Create another object
    resolution = focus_stack.resolve_reference(
        selected_objects=[],
        active_object="Sphere"
    )
    print(f"Command: 'make it bigger'")
    print(f"Resolution: {resolution['resolved_objects']}")
    print(f"Method: {resolution['resolution_method']}")
    print(f"Note: Should prefer last_modified (Cube) over last_created (Sphere)")
    print()
    
    # Scenario 3: Explicit selection takes priority
    print("Scenario 3: Explicit selection takes priority")
    print("-" * 60)
    focus_stack.record_creation("Cylinder")
    resolution = focus_stack.resolve_reference(
        selected_objects=["Sphere", "Cylinder"],
        active_object="Cube"
    )
    print(f"Command: 'delete them'")
    print(f"Resolution: {resolution['resolved_objects']}")
    print(f"Method: {resolution['resolution_method']}")
    print(f"Note: Should use selected objects, not last_created")
    print()
    
    # Scenario 4: Command history tracking
    print("Scenario 4: Command history tracking")
    print("-" * 60)
    focus_stack.record_command(
        transcript="add a cube",
        target_objects=["Cube"],
        operations=["mesh.primitive_cube_add"],
        success=True
    )
    focus_stack.record_command(
        transcript="add a sphere",
        target_objects=["Sphere"],
        operations=["mesh.primitive_sphere_add"],
        success=True
    )
    recent = focus_stack.get_recent_context(2)
    print(f"Recent commands: {len(recent)}")
    for cmd in recent:
        print(f"  - '{cmd.transcript}' -> {cmd.target_objects}")
    print()
    
    # Scenario 5: Focus summary
    print("Scenario 5: Complete focus state")
    print("-" * 60)
    summary = focus_stack.get_focus_summary(
        selected=["Sphere"],
        active="Sphere"
    )
    print(f"Selected: {summary['selected_objects']}")
    print(f"Active: {summary['active_object']}")
    print(f"Last Created: {summary['last_created']}")
    print(f"Last Modified: {summary['last_modified']}")
    print(f"Last Mentioned: {summary['last_mentioned']}")
    print()
    
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)


def compare_with_current_system():
    """Compare Focus Stack approach vs current ad-hoc prompt approach."""
    
    print("\n" + "=" * 60)
    print("Comparison: Focus Stack vs Current System")
    print("=" * 60)
    print()
    
    print("CURRENT SYSTEM (ad-hoc in prompts):")
    print("-" * 60)
    print("[X] No structured tracking of object history")
    print("[X] Reference resolution relies on GPT inference")
    print("[X] No command history - context is lost between commands")
    print("[X] Ambiguous when multiple objects exist")
    print("[X] Can't distinguish between 'it' referring to created vs modified")
    print()
    
    print("FOCUS STACK (deterministic):")
    print("-" * 60)
    print("[OK] Structured tracking: last_created, last_modified, last_mentioned")
    print("[OK] Deterministic priority: selected > modified > created > mentioned > active")
    print("[OK] Command history with object associations")
    print("[OK] Clear confidence scores (0.0-1.0)")
    print("[OK] Can explain why a reference was resolved a certain way")
    print()
    
    print("EXAMPLE IMPROVEMENT:")
    print("-" * 60)
    print("Command sequence:")
    print("  1. 'add a cube' -> Creates Cube")
    print("  2. 'add a sphere' -> Creates Sphere")
    print("  3. 'make it red' -> What does 'it' refer to?")
    print()
    print("Current system: GPT guesses (might pick wrong object)")
    print("Focus Stack: Deterministically picks Sphere (last_created)")
    print()


if __name__ == "__main__":
    test_focus_stack_resolution()
    compare_with_current_system()
