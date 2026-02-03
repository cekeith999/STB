# Phase 4 & Phase 5 Implementation Plan

## Overview
This document outlines the implementation plan for the missing features identified from the history document:
- **Phase 4**: Complex Operation Executor (support ANY Blender operation)
- **Phase 5**: Edit Mode Commands (loop cuts, subdivide, select by region)
- **Advanced Shape Analysis**: Pattern detection (cylinders, spheres, organic shapes)
- **Material/Node Operation Fixes**: Direct operator support for material.new, node.add_node
- **Vertex Pattern Storage**: Enhanced reference knowledge base with geometric patterns

---

## Phase 4: Complex Operation Executor

### Current State
- ✅ Safety system uses whitelist/blacklist (`mesh.*`, `object.*`, `transform.*`)
- ✅ Basic operations work (primitives, transforms, modifiers)
- ❌ Restricted to whitelisted prefixes only
- ❌ Material/node operations fail (`material.new`, `node.add_node`)
- ❌ Workaround: Uses `execute` with Python code for materials (not ideal)

### Goal
Support **ANY** Blender operation with GPT choosing the best approach, including:
- Modifiers (all types)
- Transforms (all types)
- Topology operations
- Materials and nodes
- Curves, surfaces, armatures
- Custom operators from add-ons

### Implementation Steps

#### Step 1: Expand Safety System
**File**: `__init__.py` (`_is_safe_op`)

**Changes**:
- Keep blacklist for dangerous operations (file.quit, wm.quit_blender, etc.)
- Remove whitelist restriction - allow any operator not in blacklist
- Add "unknown operator" handling with warning (not blocking)
- Add operator validation (check if operator exists before allowing)

**Code Structure**:
```python
def _is_safe_op(op_fullname: str):
    # Blacklist (always block)
    dangerous_ops = [
        "file.quit",
        "wm.quit_blender",
        "script.reload",
        "preferences.addon_disable",
        "preferences.addon_remove",
    ]
    
    if op_fullname in dangerous_ops:
        return False, "Dangerous operation blocked"
    
    # Validate operator exists
    try:
        cat, name = op_fullname.split(".", 1)
        cat_obj = getattr(bpy.ops, cat, None)
        if cat_obj is None:
            return False, f"Category '{cat}' not found"
        fn = getattr(cat_obj, name, None)
        if fn is None:
            return False, f"Operator '{op_fullname}' not found"
    except Exception as e:
        return False, f"Invalid operator format: {e}"
    
    # Allow all other operators (with warning in strict mode)
    return True, "Allowed"
```

#### Step 2: Material/Node Operation Support
**File**: `__init__.py` (`_safe_call_operator`)

**Problem**: `material.new` and `node.add_node` are not standard operators - they're Python API calls.

**Solution**: Create wrapper operators or use execute with proper context.

**Option A**: Create custom RPC methods for common material/node operations
```python
def create_material(name: str, use_nodes: bool = True):
    """RPC method: Create a new material."""
    mat = bpy.data.materials.new(name=name)
    if use_nodes:
        mat.use_nodes = True
    return {"ok": True, "material_name": mat.name}

def add_shader_node(material_name: str, node_type: str, location: tuple = (0, 0)):
    """RPC method: Add a shader node to a material."""
    mat = bpy.data.materials.get(material_name)
    if not mat or not mat.use_nodes:
        return {"ok": False, "error": "Material not found or nodes not enabled"}
    node = mat.node_tree.nodes.new(type=node_type)
    node.location = location
    return {"ok": True, "node_name": node.name}
```

**Option B**: Enhance `execute` Python code execution with better context
- Ensure proper Blender context is set
- Add material/node helper functions in execute scope
- Provide examples in prompts

**Recommendation**: Use Option A for common operations, Option B as fallback.

#### Step 3: GPT Operation Selection Guidance
**File**: `voice_to_blender.py` (ReAct system prompt)

**Changes**:
- Add section: "AVAILABLE OPERATIONS: You can use ANY Blender operator"
- List common operator categories with examples
- Emphasize GPT should choose best approach (modifier vs direct edit, etc.)
- Add guidance for material/node operations (use RPC methods or execute)

**Prompt Addition**:
```
--- OPERATION SELECTION GUIDANCE ---
You have access to ANY Blender operator (except dangerous ones like file.quit).

Common operator categories:
- mesh.* - Mesh operations (primitive_*, subdivide, bevel, loopcut, etc.)
- object.* - Object operations (select, delete, duplicate, modifier_add, etc.)
- transform.* - Transform operations (translate, rotate, resize, etc.)
- material.* - Material operations (if available)
- node.* - Node editor operations (if available)
- curve.* - Curve operations
- surface.* - Surface operations
- armature.* - Armature operations

For materials/nodes:
- Use RPC methods: create_material(), add_shader_node()
- OR use execute with Python code: bpy.data.materials.new(), mat.node_tree.nodes.new()

Choose the best approach:
- Modifiers vs direct mesh editing: Use modifiers for non-destructive editing
- Loop cuts vs subdivision: Use loop cuts for specific edge loops, subdivision for general detail
- Transform operations: Use transform.* operators for precise control
```

#### Step 4: Testing & Validation
- Test unknown operators (should work with warning)
- Test material/node operations (both RPC and execute methods)
- Test complex operation chains (modifier + transform + material)
- Verify safety still blocks dangerous operations

---

## Phase 5: Edit Mode Commands

### Current State
- ✅ Edit mode operations documented in prompts
- ✅ Basic edit mode support (mesh.extrude_region, mesh.bevel, etc.)
- ❌ No automatic mode switching logic
- ❌ No edit mode context management
- ❌ No reliability guarantees for edit mode operations

### Goal
Reliable edit mode operations with:
- Automatic mode switching (OBJECT → EDIT when needed)
- Context preservation (selection state, active object)
- Operation validation (ensure prerequisites are met)
- Error recovery (handle mode switching failures)

### Implementation Steps

#### Step 1: Mode Management System
**File**: `voice_to_blender.py` (new function)

**Function**: `_ensure_mode(op: str, kwargs: dict, context: dict) -> dict`

**Logic**:
```python
def _ensure_mode(op: str, kwargs: dict, context: dict) -> list:
    """
    Ensure correct mode is set before executing operation.
    Returns list of commands to execute (mode switch + operation).
    """
    commands = []
    
    # Operations that require EDIT mode
    edit_mode_ops = [
        "mesh.extrude_region",
        "mesh.inset_faces",
        "mesh.loopcut",
        "mesh.subdivide",
        "mesh.bevel",
        "mesh.knife_project",
        "mesh.select_all",  # In edit mode, this selects geometry
        "mesh.select_mode",
        "mesh.delete",  # In edit mode, deletes geometry
    ]
    
    # Operations that require OBJECT mode
    object_mode_ops = [
        "object.modifier_add",
        "object.modifier_remove",
        "object.modifier_apply",
        "object.select_all",  # In object mode, selects objects
        "object.delete",  # In object mode, deletes objects
    ]
    
    current_mode = context.get("current_mode", "OBJECT")
    active_object = context.get("active_object")
    
    if op in edit_mode_ops and current_mode != "EDIT":
        if not active_object:
            return [{"op": "error", "kwargs": {"message": "No active object to enter edit mode"}}]
        commands.append({
            "op": "object.mode_set",
            "kwargs": {"mode": "EDIT"}
        })
    
    if op in object_mode_ops and current_mode != "OBJECT":
        commands.append({
            "op": "object.mode_set",
            "kwargs": {"mode": "OBJECT"}
        })
    
    # Add the actual operation
    commands.append({"op": op, "kwargs": kwargs})
    
    return commands
```

#### Step 2: Context Preservation
**File**: `voice_to_blender.py` (`_react_execute`)

**Changes**:
- Before mode switch: Save selection state
- After mode switch: Restore selection state
- Track active object across mode switches
- Handle selection loss gracefully

**Implementation**:
```python
def _save_selection_context(rpc) -> dict:
    """Save current selection and active object."""
    try:
        context = rpc.get_modeling_context()
        return {
            "active_object": context.get("active_object", {}).get("name"),
            "selected_objects": [obj["name"] for obj in context.get("selected_objects", [])],
        }
    except:
        return {}

def _restore_selection_context(rpc, saved: dict):
    """Restore selection and active object."""
    if not saved:
        return
    
    # Restore active object
    if saved.get("active_object"):
        try:
            rpc.enqueue_op_safe("object.select_by_name", {"name": saved["active_object"]})
        except:
            pass
```

#### Step 3: Edit Mode Operation Validation
**File**: `voice_to_blender.py` (`_react_execute`)

**Validation Rules**:
- `mesh.loopcut`: Requires at least one edge selected (or works on all edges)
- `mesh.subdivide`: Requires geometry selected (or works on all)
- `mesh.bevel`: Requires edges/vertices selected
- `mesh.extrude_region`: Requires faces/edges/vertices selected

**Implementation**:
```python
def _validate_edit_mode_op(op: str, context: dict) -> tuple[bool, str]:
    """Validate that prerequisites for edit mode operation are met."""
    
    if op == "mesh.loopcut":
        # Loop cut works even without selection (cuts all edges)
        return True, ""
    
    if op == "mesh.subdivide":
        # Subdivide works on all geometry if nothing selected
        return True, ""
    
    if op in ["mesh.bevel", "mesh.extrude_region", "mesh.inset_faces"]:
        # These require selection
        mesh_analysis = context.get("mesh_analysis", {})
        selection = mesh_analysis.get("selection", {})
        has_selection = (
            selection.get("vertices") or
            selection.get("edges") or
            selection.get("faces")
        )
        if not has_selection:
            return False, f"{op} requires geometry to be selected. Select vertices, edges, or faces first."
    
    return True, ""
```

#### Step 4: Enhanced Edit Mode Prompts
**File**: `voice_to_blender.py` (ReAct system prompt)

**Add Section**:
```
--- EDIT MODE OPERATIONS (CRITICAL) ---
Edit mode operations require geometry to be selected. Always:
1. Enter EDIT mode first: object.mode_set(mode='EDIT')
2. Select geometry: mesh.select_mode(type='VERT'|'EDGE'|'FACE'), then mesh.select_all(action='SELECT')
3. Execute operation: mesh.loopcut, mesh.subdivide, etc.

Common edit mode workflow:
- "Add loop cuts": object.mode_set(mode='EDIT') → mesh.loopcut(number_cuts=2)
- "Subdivide": object.mode_set(mode='EDIT') → mesh.select_all(action='SELECT') → mesh.subdivide()
- "Bevel edges": object.mode_set(mode='EDIT') → mesh.select_mode(type='EDGE') → mesh.select_all(action='SELECT') → mesh.bevel(offset=0.1)

IMPORTANT: The system will automatically switch modes, but you should still include mode_set in your plan for clarity.
```

#### Step 5: Testing
- Test mode switching (OBJECT → EDIT → OBJECT)
- Test selection preservation
- Test edit mode operations with/without selection
- Test error recovery (invalid mode, no selection, etc.)

---

## Advanced Shape Analysis

### Current State
- ✅ Basic shape classification (flat/tall/balanced)
- ❌ No detection of specific shapes (cylinders, spheres, organic)
- ❌ No pattern recognition for common objects

### Goal
Detect specific geometric patterns:
- Cylinders (circular cross-section, consistent radius)
- Spheres (all vertices equidistant from center)
- Cubes/boxes (rectangular, 6 faces, right angles)
- Organic shapes (irregular, curved surfaces)
- Complex shapes (combinations of above)

### Implementation Steps

#### Step 1: Shape Detection Algorithms
**File**: `__init__.py` (`analyze_current_mesh`)

**New Function**: `_detect_shape_pattern(mesh, obj) -> dict`

**Algorithms**:

**Cylinder Detection**:
```python
def _detect_cylinder(mesh, obj):
    """Detect if mesh is cylindrical."""
    # Check if cross-sections are circular
    # Check if radius is consistent along axis
    # Check if height >> radius or height ≈ radius
    pass
```

**Sphere Detection**:
```python
def _detect_sphere(mesh, obj):
    """Detect if mesh is spherical."""
    # Check if all vertices are roughly equidistant from center
    # Check if surface is uniformly curved
    pass
```

**Box Detection**:
```python
def _detect_box(mesh, obj):
    """Detect if mesh is box-like."""
    # Check if 6 faces, all rectangular
    # Check if right angles (90 degrees)
    pass
```

**Organic Detection**:
```python
def _detect_organic(mesh, obj):
    """Detect if mesh is organic (irregular, curved)."""
    # Check for irregular topology
    # Check for non-uniform curvature
    # Check for non-geometric patterns
    pass
```

#### Step 2: Integration into Mesh Analysis
**File**: `__init__.py` (`analyze_current_mesh`)

**Add to return dict**:
```python
return {
    # ... existing fields ...
    "shape_patterns": {
        "primary_shape": "cylinder" | "sphere" | "box" | "organic" | "complex",
        "confidence": 0.0-1.0,
        "details": {
            "cylinder": {"radius": 1.0, "height": 2.0, "axis": "Z"},
            "sphere": {"radius": 1.0, "center": (0, 0, 0)},
            "box": {"dimensions": (2, 2, 2), "is_cube": True},
            "organic": {"irregularity_score": 0.5},
        },
        "secondary_shapes": ["box", "cylinder"],  # If complex
    }
}
```

#### Step 3: Use in Prompts
**File**: `voice_to_blender.py` (ReAct system prompt)

**Add Section**:
```
--- SHAPE ANALYSIS ---
The mesh has been analyzed for geometric patterns:
- Primary Shape: {shape_patterns.primary_shape} (confidence: {confidence})
- Details: {shape_patterns.details}

Use this information to:
- Choose appropriate operations (e.g., loop cuts for cylinders, subdivision for spheres)
- Maintain shape characteristics when modifying
- Recognize when user wants to change shape type
```

---

## Material/Node Operation Fixes

### Current State
- ❌ `material.new` fails (not a standard operator)
- ❌ `node.add_node` fails (not a standard operator)
- ✅ Workaround: Uses `execute` with Python code

### Goal
Direct support for material/node operations via:
- Custom RPC methods for common operations
- Enhanced `execute` with proper context
- Better error messages and guidance

### Implementation Steps

#### Step 1: Material RPC Methods
**File**: `__init__.py` (`_server_loop`)

**Add RPC Methods**:
```python
def create_material(name: str, use_nodes: bool = True, base_color: tuple = None):
    """Create a new material with optional base color."""
    mat = bpy.data.materials.new(name=name)
    if use_nodes:
        mat.use_nodes = True
        if base_color:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    return {"ok": True, "material_name": mat.name}

def assign_material(object_name: str, material_name: str):
    """Assign a material to an object."""
    obj = bpy.data.objects.get(object_name)
    mat = bpy.data.materials.get(material_name)
    if not obj:
        return {"ok": False, "error": f"Object '{object_name}' not found"}
    if not mat:
        return {"ok": False, "error": f"Material '{material_name}' not found"}
    
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    
    return {"ok": True, "message": f"Material '{material_name}' assigned to '{object_name}'"}

def add_shader_node(material_name: str, node_type: str, location: tuple = (0, 0)):
    """Add a shader node to a material."""
    mat = bpy.data.materials.get(material_name)
    if not mat:
        return {"ok": False, "error": f"Material '{material_name}' not found"}
    if not mat.use_nodes:
        return {"ok": False, "error": "Material does not use nodes"}
    
    node = mat.node_tree.nodes.new(type=node_type)
    node.location = location
    return {"ok": True, "node_name": node.name}

def set_node_input(material_name: str, node_name: str, input_name: str, value):
    """Set an input value on a shader node."""
    mat = bpy.data.materials.get(material_name)
    if not mat or not mat.use_nodes:
        return {"ok": False, "error": "Material not found or nodes not enabled"}
    
    node = mat.node_tree.nodes.get(node_name)
    if not node:
        return {"ok": False, "error": f"Node '{node_name}' not found"}
    
    if input_name not in node.inputs:
        return {"ok": False, "error": f"Input '{input_name}' not found on node"}
    
    node.inputs[input_name].default_value = value
    return {"ok": True, "message": f"Set {input_name} on {node_name}"}
```

#### Step 2: Register RPC Methods
**File**: `__init__.py` (`_server_loop`)

```python
server.register_function(create_material, "create_material")
server.register_function(assign_material, "assign_material")
server.register_function(add_shader_node, "add_shader_node")
server.register_function(set_node_input, "set_node_input")
```

#### Step 3: Update Prompts
**File**: `voice_to_blender.py` (ReAct system prompt)

**Replace material section with**:
```
--- MATERIAL OPERATIONS (USE RPC METHODS) ---
DO NOT use material.new or node.add_node (these are not operators).

CORRECT WAY:
1. Create material: Use RPC method create_material(name="RedMaterial", base_color=(1, 0, 0))
2. Assign material: Use RPC method assign_material(object_name="Cube", material_name="RedMaterial")
3. Add shader node: Use RPC method add_shader_node(material_name="RedMaterial", node_type="ShaderNodeEmission")
4. Set node input: Use RPC method set_node_input(material_name="RedMaterial", node_name="Principled BSDF", input_name="Base Color", value=(1, 0, 0, 1))

OR use execute with Python code:
{"op": "execute", "kwargs": {"code": "import bpy\nmat = bpy.data.materials.new(name='RedMaterial')\nmat.use_nodes = True\nbsdf = mat.node_tree.nodes.get('Principled BSDF')\nbsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)\nobj = bpy.context.active_object\nobj.data.materials.append(mat)"}}
```

---

## Vertex Pattern Storage

### Current State
- ✅ Reference knowledge base exists (`_get_reference_knowledge`)
- ✅ Has modeling approaches and step-by-step templates
- ❌ No actual vertex pattern storage
- ❌ No geometric pattern matching

### Goal
Store and match geometric patterns:
- Vertex count ranges for common objects
- Typical topology patterns (edge loops, face counts)
- Proportions and dimensions
- Use for validation and guidance

### Implementation Steps

#### Step 1: Enhanced Reference Knowledge Structure
**File**: `voice_to_blender.py` (`_get_hardcoded_reference`)

**Add to knowledge structure**:
```python
"echo dot": {
    # ... existing fields ...
    "geometric_patterns": {
        "typical_vertex_count": (200, 1000),  # Range
        "typical_face_count": (100, 500),
        "topology_patterns": [
            "circular_top_face",
            "cylindrical_body",
            "beveled_edges",
        ],
        "proportions": {
            "height_diameter_ratio": 1.0,  # height ≈ diameter
            "typical_size": (3.0, 3.0, 3.0),  # Blender units
        },
        "vertex_pattern_examples": [
            # Sample vertex positions for validation
            # (not full mesh, just key points)
        ],
    }
}
```

#### Step 2: Pattern Matching Function
**File**: `voice_to_blender.py` (new function)

```python
def _match_geometric_pattern(mesh_analysis: dict, target_object: str) -> dict:
    """Match current mesh geometry to known patterns for target object."""
    reference = _get_reference_knowledge(target_object, use_gpt=False)
    if not reference or "geometric_patterns" not in reference:
        return {"match": False, "confidence": 0.0}
    
    patterns = reference["geometric_patterns"]
    vertex_count = mesh_analysis.get("vertex_count", 0)
    face_count = mesh_analysis.get("face_count", 0)
    
    # Check vertex count range
    v_range = patterns.get("typical_vertex_count", (0, float('inf')))
    vertex_match = v_range[0] <= vertex_count <= v_range[1]
    
    # Check face count range
    f_range = patterns.get("typical_face_count", (0, float('inf')))
    face_match = f_range[0] <= face_count <= f_range[1]
    
    # Check proportions
    bounds = mesh_analysis.get("bounds", {})
    size = bounds.get("size", (0, 0, 0))
    proportions = patterns.get("proportions", {})
    height_diameter_ratio = proportions.get("height_diameter_ratio")
    if height_diameter_ratio and len(size) >= 3:
        actual_ratio = size[2] / max(size[0], size[1]) if max(size[0], size[1]) > 0 else 0
        ratio_match = abs(actual_ratio - height_diameter_ratio) < 0.3
    
    confidence = (
        (0.4 if vertex_match else 0) +
        (0.3 if face_match else 0) +
        (0.3 if ratio_match else 0)
    )
    
    return {
        "match": confidence > 0.5,
        "confidence": confidence,
        "details": {
            "vertex_match": vertex_match,
            "face_match": face_match,
            "ratio_match": ratio_match,
        }
    }
```

#### Step 3: Use in Quality Assessment
**File**: `voice_to_blender.py` (`gpt_to_json_react` - finish action)

**Add to quality check**:
```python
# Match geometric patterns
pattern_match = _match_geometric_pattern(mesh_analysis, target_object)
if pattern_match["match"]:
    quality_check_passed = True
    print(f"[ReAct] ✅ Geometric pattern match: {pattern_match['confidence']:.2f}")
else:
    print(f"[ReAct] ⚠️ Geometric pattern mismatch: {pattern_match['confidence']:.2f}")
    # Add guidance to improve match
```

---

## Implementation Priority

### Phase 1 (High Priority)
1. **Material/Node Operation Fixes** - Blocks current functionality
2. **Edit Mode Context Management** - Needed for reliable edit operations

### Phase 2 (Medium Priority)
3. **Phase 4: Complex Operation Executor** - Expands capabilities
4. **Advanced Shape Analysis** - Improves AI understanding

### Phase 3 (Lower Priority)
5. **Vertex Pattern Storage** - Nice to have for validation

---

## Testing Plan

### Phase 4 Testing
- [ ] Test unknown operators (should work)
- [ ] Test material/node RPC methods
- [ ] Test complex operation chains
- [ ] Verify safety still works

### Phase 5 Testing
- [ ] Test mode switching (OBJECT ↔ EDIT)
- [ ] Test selection preservation
- [ ] Test edit mode operations with/without selection
- [ ] Test error recovery

### Shape Analysis Testing
- [ ] Test cylinder detection
- [ ] Test sphere detection
- [ ] Test box detection
- [ ] Test organic detection

### Material/Node Testing
- [ ] Test create_material RPC
- [ ] Test assign_material RPC
- [ ] Test add_shader_node RPC
- [ ] Test set_node_input RPC

---

## Notes

- This implementation should be done **after** the multi-agent architecture (Phase 2) is complete
- These features enhance the existing system but don't require architectural changes
- Material/node fixes are critical and should be done first
- Edit mode reliability is important for user experience
