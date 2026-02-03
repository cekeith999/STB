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

from agents import TaskSpec
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
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate result structure
        if "code" not in result:
            raise ValueError("Code Generator must return 'code' field")
        
        return result
    
    def _build_geometry_context(self,
                                 geometry_state: Dict,
                                 mesh_analysis: Optional[Dict]) -> str:
        """Build detailed geometry context for the prompt, including positions, sizes, and relationships."""
        
        parts = []
        
        # Scene state
        if geometry_state:
            parts.append(f"=== SCENE OVERVIEW ===")
            parts.append(f"Total objects: {geometry_state.get('object_count', 0)}")
            
            # All objects with detailed information
            all_objects = []
            
            # Selected objects with full details
            selected = geometry_state.get("selected_objects", [])
            if selected:
                parts.append(f"\n=== SELECTED OBJECTS ({len(selected)}) ===")
                for obj in selected:
                    name = obj.get('name', 'Unknown')
                    obj_type = obj.get('type', 'Unknown')
                    location = obj.get('location', (0, 0, 0))
                    scale = obj.get('scale', (1, 1, 1))
                    rotation = obj.get('rotation_euler', (0, 0, 0))
                    
                    parts.append(f"\n{name}:")
                    parts.append(f"  Type: {obj_type}")
                    parts.append(f"  Location: ({location[0]:.3f}, {location[1]:.3f}, {location[2]:.3f})")
                    parts.append(f"  Scale: ({scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f})")
                    parts.append(f"  Rotation: ({rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f})")
                    
                    all_objects.append({
                        "name": name,
                        "location": location,
                        "scale": scale,
                        "type": obj_type
                    })
            
            # Active object with full details
            active = geometry_state.get("active_object")
            if active:
                name = active.get('name', 'Unknown')
                obj_type = active.get('type', 'Unknown')
                location = active.get('location', (0, 0, 0))
                scale = active.get('scale', (1, 1, 1))
                rotation = active.get('rotation_euler', (0, 0, 0))
                mode = active.get('mode', 'OBJECT')
                
                parts.append(f"\n=== ACTIVE OBJECT ===")
                parts.append(f"{name}:")
                parts.append(f"  Type: {obj_type}")
                parts.append(f"  Mode: {mode}")
                parts.append(f"  Location: ({location[0]:.3f}, {location[1]:.3f}, {location[2]:.3f})")
                parts.append(f"  Scale: ({scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f})")
                parts.append(f"  Rotation: ({rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f})")
                
                # Check if already in all_objects
                if not any(o["name"] == name for o in all_objects):
                    all_objects.append({
                        "name": name,
                        "location": location,
                        "scale": scale,
                        "type": obj_type
                    })
            
            # All objects summary for reference
            if all_objects:
                parts.append(f"\n=== ALL OBJECTS IN SCENE ===")
                for obj in all_objects:
                    parts.append(f"{obj['name']}: location={obj['location']}, scale={obj['scale']}, type={obj['type']}")
        
        # Mesh details with dimensions
        if mesh_analysis and not mesh_analysis.get("error"):
            parts.append(f"\n=== MESH ANALYSIS (Active Object) ===")
            parts.append(f"Vertices: {mesh_analysis.get('vertex_count', 'N/A')}")
            parts.append(f"Edges: {mesh_analysis.get('edge_count', 'N/A')}")
            parts.append(f"Faces: {mesh_analysis.get('face_count', 'N/A')}")
            
            bounds = mesh_analysis.get('bounding_box', {})
            if bounds:
                size = bounds.get('size', (0, 0, 0))
                center = bounds.get('center', (0, 0, 0))
                parts.append(f"Bounding box size: ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})")
                parts.append(f"Bounding box center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                if bounds.get('shape_class'):
                    parts.append(f"Shape class: {bounds['shape_class']}")
            
            topology = mesh_analysis.get("topology", {})
            if topology:
                parts.append(f"Is manifold: {topology.get('is_manifold', 'unknown')}")
                parts.append(f"Has ngons: {topology.get('has_ngons', 'unknown')}")
        
        return "\n".join(parts) if parts else "No geometry context available"
    
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
            elif action == "create" or action == "add":
                examples.append("""
# Create primitive
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
# Or sphere:
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
""")
            elif action == "material" or action == "shader":
                examples.append("""
# Create and assign material
import bpy
mat = bpy.data.materials.new(name="MyMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # Red
obj = bpy.context.active_object
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
""")
        
        if not examples:
            examples.append("# Use standard Blender operators (bpy.ops.*) and data API (bpy.data.*)")
        
        return "\n".join(examples)
    
    def _get_relevant_api_examples_for_operation(self, operation: str) -> str:
        """Get relevant Blender API examples for a specific operation."""
        
        operation_lower = operation.lower()
        
        if "bevel" in operation_lower:
            return """
# Bevel edges
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.bevel(offset=0.1, segments=3, affect='EDGES')
bpy.ops.object.mode_set(mode='OBJECT')
"""
        elif "subdivide" in operation_lower:
            return """
# Subdivide mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.subdivide(number_cuts=2)
bpy.ops.object.mode_set(mode='OBJECT')
"""
        elif "smooth" in operation_lower:
            return """
# Smooth shading + subdivision surface
bpy.ops.object.shade_smooth()
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subdivision"].levels = 2
"""
        elif "scale" in operation_lower or "resize" in operation_lower:
            return """
# Scale object
bpy.ops.transform.resize(value=(1.0, 1.0, 0.5))  # Scale Z to 50%
# Or set scale directly:
bpy.context.object.scale = (1.0, 1.0, 0.5)
"""
        elif "extrude" in operation_lower:
            return """
# Extrude faces
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, 1)})
bpy.ops.object.mode_set(mode='OBJECT')
"""
        elif "create" in operation_lower or "add" in operation_lower or "primitive" in operation_lower:
            return """
# Create primitive
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
# Or sphere:
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
# Or cylinder:
bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 0))
"""
        elif "material" in operation_lower or "shader" in operation_lower:
            return """
# Create and assign material
import bpy
mat = bpy.data.materials.new(name="MyMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # Red
obj = bpy.context.active_object
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
"""
        else:
            return "# Use standard Blender operators (bpy.ops.*) and data API (bpy.data.*)"
    
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
