# VideoCAD Integration Guide

This guide explains how to integrate VideoCAD dataset and VideoCADFormer model into your Speech-to-Blender project.

## Overview

**VideoCAD** provides:
- 41K+ videos of Onshape CAD operations
- UI actions (mouse clicks, keyboard, typed input)
- Target CAD images
- VideoCADFormer: Pre-trained model for predicting CAD UI actions

**Goal**: Translate Onshape CAD operations → Blender operations

## Architecture

```
VideoCAD Dataset
    ↓
[Stream from Dataverse] → Process → Cache in Azure
    ↓
VideoCAD Actions (Onshape UI)
    ↓
[Translation Layer] ← VideoCADToBlenderTranslator
    ↓
Blender Commands (bpy.ops)
    ↓
Your Speech-to-Blender Pipeline
```

## Setup

### 1. Install Dependencies

Add to your `requirements.txt`:
```txt
# VideoCAD dependencies (if not already installed globally)
azure-storage-blob
adlfs
python-dotenv
opencv-python
numpy
```

### 2. Configure VideoCAD

1. Set up VideoCAD streaming (see VideoCAD project docs)
2. Add file IDs to `dataverse_streamer.py`
3. Configure Azure credentials in `.env`

### 3. Integration Points

#### Option A: Use VideoCAD Data for Training

Train/fine-tune your speech-to-blender model on VideoCAD data:

```python
from videocad_integration import VideoCADDataLoader

loader = VideoCADDataLoader(use_azure=True)

# Get training pairs: (frames, actions) → Blender commands
training_data = loader.get_batch(["00000070", "00000071", "00000072"])

# Use for:
# - Fine-tuning GPT on CAD → Blender mappings
# - Training a model: speech → VideoCAD → Blender
# - Creating few-shot examples
```

#### Option B: Use VideoCADFormer Model

If you have access to the pre-trained VideoCADFormer model:

```python
# Load VideoCADFormer model
from model.autoregressive_transformer import VideoCADFormer

model = VideoCADFormer.load_from_checkpoint("path/to/checkpoint.pt")

# Predict actions from video frames
frames = ...  # Your video frames
predicted_actions = model.predict(frames)

# Translate to Blender
from videocad_integration import VideoCADToBlenderTranslator
translator = VideoCADToBlenderTranslator()
blender_cmds = translator.translate_action_sequence(predicted_actions)
```

#### Option C: Enhance GPT Prompts

Use VideoCAD examples as few-shot prompts:

```python
from videocad_integration import VideoCADDataLoader, VideoCADToBlenderTranslator

loader = VideoCADDataLoader()
translator = VideoCADToBlenderTranslator()

# Get examples
examples = loader.get_batch(["00000070", "00000071"])

# Create few-shot prompt
few_shot_prompt = "Here are examples of CAD operations translated to Blender:\n"
for ex in examples:
    few_shot_prompt += f"Operation: {ex['operation_type']}\n"
    few_shot_prompt += f"Blender: {ex['blender_commands']}\n\n"

# Use in your GPT call
```

## Translation Mapping

### Onshape → Blender Operations

| Onshape Operation | Blender Equivalent |
|------------------|-------------------|
| Extrude | `bpy.ops.mesh.extrude_region()` |
| Sketch | `bpy.ops.mesh.primitive_*_add()` or manual mesh |
| Revolve | `bpy.ops.mesh.spin()` |
| Fillet | `bpy.ops.mesh.bevel()` |
| Chamfer | `bpy.ops.mesh.bevel()` |
| Loft | `bpy.ops.mesh.bridge_edge_loops()` |
| Sweep | Manual mesh editing |

### Action Classes

VideoCAD actions have classes:
- `0`: Mouse click → Context-dependent (selection, tool activation)
- `1`: Mouse drag → Transform, extrude, etc.
- `2`: Keyboard shortcut → Blender shortcuts
- `3`: Typed input → Numeric parameters
- `4`: Other → Context-dependent

## Usage Examples

### Example 1: Translate VideoCAD Sequence

```python
from videocad_integration import VideoCADDataLoader, VideoCADToBlenderTranslator

# Load VideoCAD data
loader = VideoCADDataLoader(use_azure=True)
data = loader.get_training_pair("00000070")

# Get translated Blender commands
blender_commands = data["blender_commands"]

# Send to Blender (using your existing RPC system)
for cmd in blender_commands:
    send_to_blender(cmd)  # Your existing function
```

### Example 2: Use in GPT Pipeline

Modify your `voice_to_blender.py` to use VideoCAD examples:

```python
from videocad_integration import VideoCADDataLoader

# In your gpt_to_json function
loader = VideoCADDataLoader()
examples = loader.get_batch(["00000070", "00000071"])

system_prompt = f"""
You are a CAD-to-Blender translator. Here are examples:

{format_videocad_examples(examples)}

Translate the user's CAD operation request to Blender commands.
"""

# Use in GPT call
```

### Example 3: Train on VideoCAD Data

```python
from videocad_integration import VideoCADDataLoader

loader = VideoCADDataLoader()

# Get training dataset
file_ids = ["00000070", "00000071", ...]  # Your subset
training_data = loader.get_batch(file_ids)

# Create training pairs: (speech_description, blender_commands)
# You'll need to add speech descriptions (maybe from VideoCAD metadata)

# Train your model
# ...
```

## Integration with Existing Code

### Modify `voice_to_blender.py`

Add VideoCAD integration:

```python
# At top of file
from videocad_integration import VideoCADDataLoader, create_blender_command_from_videocad

# In your command processing
def process_with_videocad(transcript: str):
    # Try VideoCAD-based translation
    loader = VideoCADDataLoader()
    
    # Find similar operations in VideoCAD
    # Translate to Blender
    # Fall back to GPT if needed
    ...
```

### Modify `command_exec.py`

Add VideoCAD-translated operations:

```python
# In execute_command function
elif t == "EXTRUDE":  # From VideoCAD translation
    bpy.ops.mesh.extrude_region()
    # Apply distance from args
    if "distance" in a:
        bpy.ops.transform.translate(value=(0, 0, a["distance"]))
```

## Next Steps

1. **Set up VideoCAD streaming** (see VideoCAD project)
2. **Test translation** with a few examples
3. **Expand translation mappings** based on your needs
4. **Integrate into pipeline** (GPT prompts or direct translation)
5. **Train/fine-tune** if needed

## Resources

- VideoCAD Paper: https://arxiv.org/abs/2505.24838
- VideoCAD Project: https://ghadinehme.github.io/videocad.github.io/
- VideoCAD Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WX8PCK

## Notes

- VideoCAD uses Onshape (web-based CAD), Blender is different
- Some operations don't map 1:1 - you'll need custom logic
- Frame analysis can help determine context (what tool is active)
- Consider using VideoCADFormer model if available

