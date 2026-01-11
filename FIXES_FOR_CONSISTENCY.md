# Fixes for Consistent Results

Based on run analysis, here are the specific fixes needed:

## 🔧 Priority 1: Fix Duplicate Detection for Multi-Object Operations

**Problem**: `object.modifier_add` on different objects is treated as duplicate

**Current Code** (line 1323-1326):
```python
if op not in allow_repeat_ops:
    cmd_signature = (op, json.dumps(kwargs, sort_keys=True))
    if cmd_signature in _REACT_STATE_MEMORY["executed_ops"]:
        return f"⚠️ Duplicate operation skipped: {op} (already executed with same parameters)"
```

**Fix**: Include active object name in signature OR allow modifier operations to repeat

**Solution A** (Include object context):
```python
# Get active object name if available
active_obj_name = None
if op in ("object.modifier_add", "object.modifier_remove", "object.modifier_apply"):
    # These operations are object-specific, include object name
    try:
        # Try to get from kwargs or context
        active_obj_name = kwargs.get("object_name") or "unknown"
    except:
        pass

if op not in allow_repeat_ops:
    if active_obj_name:
        cmd_signature = (op, active_obj_name, json.dumps(kwargs, sort_keys=True))
    else:
        cmd_signature = (op, json.dumps(kwargs, sort_keys=True))
```

**Solution B** (Simpler - allow modifier ops to repeat):
```python
allow_repeat_ops = {
    "object.select_all",
    "object.select_by_type",
    "object.join",
    "object.delete",
    "mesh.select_all",
    "mesh.select_mode",
    "object.mode_set",
    "object.modifier_add",  # ← ADD THIS
    "object.modifier_remove",
    "object.modifier_apply",
}
```

**Recommendation**: Use Solution B (simpler, less error-prone)

---

## 🔧 Priority 2: Fix Transform Operations Timing

**Problem**: Mesh analysis reads object size before `transform.resize` applies

**Current Code** (line 1336):
```python
result = send_to_blender(single_cmd)
time.sleep(0.15)  # Give Blender a moment to process
```

**Fix**: Add longer delay after transform operations, OR verify transform applied

**Solution**:
```python
result = send_to_blender(single_cmd)

# Extra delay for transform operations that affect mesh analysis
if op in ("transform.resize", "transform.translate", "transform.rotate", "object.scale"):
    time.sleep(0.3)  # Longer delay for transforms
else:
    time.sleep(0.15)
```

**Alternative**: Verify transform applied by checking object scale/size after operation

---

## 🔧 Priority 3: Improve Quality Assessment Integration

**Problem**: AI doesn't know exact format for quality assessment

**Current System Prompt**: Mentions quality check but no explicit format

**Fix**: Add explicit example in system prompt

**Add to System Prompt**:
```
MANDATORY QUALITY CHECK BEFORE FINISHING:
Before using 'finish', you MUST check quality using:
  Action: observe
  Action Input: assess_object_quality iPhone 16

Example output format:
  📊 OBJECT QUALITY ASSESSMENT:
  Overall Quality Score: 0.75/1.0
  Target Match Score: 0.67/1.0
  ✅ FOUND FEATURES: Body found, Screen found
  ❌ MISSING FEATURES: Missing camera
  💡 Suggestions: Add camera module

Only use 'finish' if quality_score >= 0.5 AND target_match_score >= 0.5
```

---

## 🔧 Priority 4: Add Relative Positioning Guidance

**Problem**: AI uses absolute coordinates instead of relative to body

**Fix**: Add positioning examples to system prompt

**Add to System Prompt**:
```
OBJECT PLACEMENT GUIDELINES:
- Always position objects RELATIVE to the main body
- Example: If body is at [0, 0, 0] with size [0.07, 0.15, 0.01]:
  * Left edge: X = -0.035 (body center - half width)
  * Right edge: X = +0.035
  * Top edge: Y = +0.075
  * Bottom edge: Y = -0.075
  * Front face: Z = +0.005
  * Back face: Z = -0.005

- Buttons on left side: X = body_left_edge - 0.001 = -0.036
- Buttons should be spaced: ~0.015 units apart vertically
- Camera lenses should be spaced: ~0.02 units apart vertically
```

---

## 🔧 Priority 5: Fix Active Object Tracking

**Problem**: Mesh analysis sometimes reads wrong object

**Current**: Analysis reads `bpy.context.active_object` but active object might have changed

**Fix**: Verify active object matches before analysis, OR pass object name explicitly

**Solution**: In `analyze_current_mesh()`, verify active object is what we expect:
```python
def analyze_current_mesh():
    active_obj = context.view_layer.objects.active
    if not active_obj:
        return {"error": "No active object"}
    
    # Verify object name matches what we expect
    # (This might need to be passed as parameter)
```

**Alternative**: Always select object before analyzing it

---

## 🔧 Priority 6: Add Position/Size Validation RPC Methods

**New RPC Methods to Add**:

1. **`validate_object_position(object_name, expected_position, tolerance=0.001)`**
   - Check if object is at expected position
   - Returns: `{"valid": bool, "actual_position": [...], "offset": [...]}`

2. **`validate_object_size(object_name, expected_size, tolerance=0.001)`**
   - Check if object matches expected size
   - Returns: `{"valid": bool, "actual_size": [...], "difference": [...]}`

3. **`get_object_relative_position(object_name, reference_object_name)`**
   - Get position relative to another object
   - Returns: `{"relative_position": [...], "distance": float}`

These can be called by AI to verify correctness after creation.

---

## 📋 Implementation Order

1. ✅ **Fix Duplicate Detection** (Solution B - add modifier ops to allow_repeat_ops)
2. ✅ **Add Transform Delay** (Increase delay after transform operations)
3. ✅ **Improve Quality Assessment Format** (Add explicit example to system prompt)
4. ✅ **Add Positioning Guidance** (Add relative positioning examples)
5. ⚠️ **Fix Active Object Tracking** (Verify before analysis)
6. ⚠️ **Add Validation Methods** (New RPC methods for position/size checking)

---

## 🧪 Testing Plan

After fixes, test with:
1. "Create an iPhone 16"
   - Verify buttons are correct size and position
   - Verify camera lenses don't overlap
   - Verify quality assessment works
   - Verify finishes successfully

2. "Create a sports car"
   - Verify 4 wheels are positioned correctly
   - Verify body proportions

3. "Create a coffee mug"
   - Verify handle is positioned correctly relative to body

---

**Priority**: Fix #1 and #2 first (duplicate detection and transform timing) as these are blocking issues.
