"""
Example: Using VideoCAD in Speech-to-Blender Pipeline

This shows how to integrate VideoCAD data/model into your existing
speech-to-blender workflow.
"""

import sys
import os

# Add VideoCAD to path
sys.path.append(r"C:\Users\Jordan\Downloads\myVideoCAD\myVideoCAD-main")

from videocad_integration import VideoCADDataLoader, VideoCADToBlenderTranslator
import json


def example_1_use_videocad_for_training():
    """
    Example 1: Use VideoCAD data to create training examples
    for your speech-to-blender model.
    """
    print("=" * 60)
    print("Example 1: Training Data from VideoCAD")
    print("=" * 60)
    
    loader = VideoCADDataLoader(use_azure=True)
    
    # Get training pairs
    file_ids = ["00000070", "00000071", "00000072"]  # Your configured IDs
    training_data = loader.get_batch(file_ids)
    
    # Create training examples
    examples = []
    for data in training_data:
        examples.append({
            "input": {
                "frames": data["frames"],
                "videocad_actions": data["videocad_actions"]
            },
            "output": {
                "blender_commands": data["blender_commands"]
            }
        })
    
    print(f"Created {len(examples)} training examples")
    return examples


def example_2_enhance_gpt_prompts():
    """
    Example 2: Use VideoCAD examples as few-shot prompts for GPT.
    """
    print("\n" + "=" * 60)
    print("Example 2: GPT Few-Shot Examples")
    print("=" * 60)
    
    loader = VideoCADDataLoader(use_azure=True)
    translator = VideoCADToBlenderTranslator()
    
    # Get a few examples
    examples = loader.get_batch(["00000070", "00000071"])
    
    # Format as few-shot prompt
    few_shot_prompt = """Here are examples of CAD operations translated to Blender commands:

"""
    
    for i, ex in enumerate(examples[:3], 1):  # Use first 3
        few_shot_prompt += f"Example {i}:\n"
        few_shot_prompt += f"  VideoCAD Actions: {len(ex['videocad_actions'])} actions\n"
        few_shot_prompt += f"  Blender Commands: {len(ex['blender_commands'])} commands\n"
        
        # Show first few commands
        for cmd in ex['blender_commands'][:2]:
            few_shot_prompt += f"    - {cmd}\n"
        few_shot_prompt += "\n"
    
    print("Few-shot prompt created:")
    print(few_shot_prompt[:500] + "...")
    
    # Use this in your GPT call:
    # response = openai.ChatCompletion.create(
    #     messages=[
    #         {"role": "system", "content": few_shot_prompt},
    #         {"role": "user", "content": user_transcript}
    #     ]
    # )
    
    return few_shot_prompt


def example_3_direct_translation():
    """
    Example 3: Directly translate VideoCAD actions to Blender commands.
    """
    print("\n" + "=" * 60)
    print("Example 3: Direct Translation")
    print("=" * 60)
    
    loader = VideoCADDataLoader(use_azure=True)
    
    # Get VideoCAD sequence
    data = loader.get_training_pair("00000070")
    
    # Get translated Blender commands
    blender_commands = data["blender_commands"]
    
    print(f"Translated {len(blender_commands)} Blender commands")
    
    # Format for your RPC system
    for cmd in blender_commands[:5]:  # Show first 5
        print(f"  {cmd}")
    
    # Send to Blender (using your existing system)
    # for cmd in blender_commands:
    #     send_to_blender(cmd)  # Your existing function
    
    return blender_commands


def example_4_integrate_with_voice_pipeline():
    """
    Example 4: Integrate VideoCAD into your voice_to_blender.py pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 4: Voice Pipeline Integration")
    print("=" * 60)
    
    # This shows how you'd modify voice_to_blender.py
    
    integration_code = """
# In voice_to_blender.py, add:

from videocad_integration import VideoCADDataLoader, VideoCADToBlenderTranslator

# Initialize (do this once at startup)
videocad_loader = VideoCADDataLoader(use_azure=True)
videocad_translator = VideoCADToBlenderTranslator()

def process_with_videocad(transcript: str):
    \"\"\"
    Try to use VideoCAD data to help translate speech to Blender commands.
    \"\"\"
    # Option 1: Use VideoCAD examples in GPT prompt
    examples = videocad_loader.get_batch(["00000070", "00000071"])
    few_shot_prompt = create_few_shot_prompt(examples)
    
    # Enhanced GPT call with VideoCAD context
    response = gpt_to_json(transcript, system_prompt=few_shot_prompt)
    
    # Option 2: Direct translation if you can map speech → VideoCAD actions
    # (This would require additional model/training)
    
    return response

# Modify your existing gpt_to_json function:
def gpt_to_json(transcript: str, system_prompt: str = None):
    if system_prompt is None:
        # Add VideoCAD examples to default prompt
        examples = videocad_loader.get_batch(["00000070"])
        system_prompt = create_few_shot_prompt(examples)
    
    # ... rest of your GPT call
"""
    
    print(integration_code)
    return integration_code


def example_5_create_training_dataset():
    """
    Example 5: Create a dataset for training a speech → VideoCAD → Blender model.
    """
    print("\n" + "=" * 60)
    print("Example 5: Training Dataset Creation")
    print("=" * 60)
    
    loader = VideoCADDataLoader(use_azure=True)
    
    # Get VideoCAD data
    file_ids = ["00000070", "00000071", "00000072"]  # Expand as needed
    training_data = loader.get_batch(file_ids)
    
    # Create dataset structure
    dataset = []
    for data in training_data:
        # Each entry: (input, output)
        # Input: Could be speech description (you'd need to add this)
        # Output: Blender commands
        dataset.append({
            "file_id": data["file_id"],
            "input": {
                "frames": data["frames"].tolist() if hasattr(data["frames"], "tolist") else data["frames"],
                "actions": data["videocad_actions"].tolist() if hasattr(data["videocad_actions"], "tolist") else data["videocad_actions"]
            },
            "output": data["blender_commands"],
            "metadata": {
                "num_steps": data["num_steps"]
            }
        })
    
    # Save dataset
    with open("videocad_training_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created training dataset with {len(dataset)} examples")
    print("Saved to: videocad_training_dataset.json")
    
    return dataset


if __name__ == "__main__":
    print("VideoCAD Integration Examples")
    print("=" * 60)
    
    # Run examples (comment out ones that need actual VideoCAD setup)
    # example_1_use_videocad_for_training()
    # example_2_enhance_gpt_prompts()
    # example_3_direct_translation()
    example_4_integrate_with_voice_pipeline()
    # example_5_create_training_dataset()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Set up VideoCAD streaming (see VideoCAD project)")
    print("2. Configure file IDs in dataverse_streamer.py")
    print("3. Test with example_3_direct_translation()")
    print("4. Integrate into voice_to_blender.py")
    print("=" * 60)

