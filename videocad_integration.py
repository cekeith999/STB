"""
VideoCAD Integration for Speech-to-Blender

This module integrates VideoCAD dataset and VideoCADFormer model to translate
Onshape CAD operations into Blender operations.

VideoCAD provides:
- 41K+ videos of Onshape CAD operations
- UI actions (mouse, keyboard, typed input)
- Target CAD images

This integration:
1. Maps VideoCAD actions to Blender bpy.ops operations
2. Uses VideoCAD data for training/fine-tuning
3. Translates Onshape operations to Blender equivalents
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json

# Add VideoCAD to path
VIDEOCAD_PATH = r"C:\Users\Jordan\Downloads\myVideoCAD\myVideoCAD-main"
if VIDEOCAD_PATH not in sys.path:
    sys.path.append(VIDEOCAD_PATH)

try:
    from dataverse_streamer import create_dataverse_streamer
    VIDEOCAD_AVAILABLE = True
except ImportError:
    VIDEOCAD_AVAILABLE = False
    print("Warning: VideoCAD modules not available. Install VideoCAD dependencies.")


class VideoCADToBlenderTranslator:
    """
    Translates VideoCAD (Onshape) UI actions to Blender operations.
    
    VideoCAD actions format: [action_class, x, y, button, param1, param2, param3, param4]
    - action_class: 0=mouse_click, 1=mouse_drag, 2=keyboard, 3=typed_input, 4=other
    """
    
    def __init__(self):
        self.action_mappings = self._build_action_mappings()
    
    def _build_action_mappings(self) -> Dict[int, Dict]:
        """
        Map VideoCAD action classes to Blender operations.
        
        Returns:
            Dictionary mapping action_class to Blender operation info
        """
        return {
            0: {  # Mouse click
                "type": "MOUSE_CLICK",
                "blender_op": None,  # Context-dependent
            },
            1: {  # Mouse drag
                "type": "MOUSE_DRAG",
                "blender_op": None,  # Context-dependent
            },
            2: {  # Keyboard shortcut
                "type": "KEYBOARD",
                "blender_op": None,  # Context-dependent
            },
            3: {  # Typed input
                "type": "TYPED_INPUT",
                "blender_op": None,  # Context-dependent
            },
            4: {  # Other/unknown
                "type": "OTHER",
                "blender_op": None,
            },
        }
    
    def translate_action_sequence(self, actions: List[List[float]], 
                                 frames: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Translate a sequence of VideoCAD actions to Blender commands.
        
        Args:
            actions: List of action vectors from VideoCAD [action_class, x, y, ...]
            frames: Optional video frames for context
        
        Returns:
            List of Blender command dictionaries
        """
        blender_commands = []
        
        for i, action in enumerate(actions):
            action_class = int(action[0])
            
            # Get context from previous commands and current frame
            context = self._get_context(blender_commands, frames[i] if frames else None)
            
            # Translate based on action class and context
            cmd = self._translate_single_action(action, action_class, context)
            
            if cmd:
                blender_commands.append(cmd)
        
        return blender_commands
    
    def _get_context(self, previous_commands: List[Dict], frame: Optional[Any]) -> Dict:
        """Extract context from previous commands and current frame"""
        context = {
            "last_operation": previous_commands[-1]["type"] if previous_commands else None,
            "has_selection": False,  # Would need to track Blender state
            "mode": "OBJECT",  # Would need to track Blender mode
        }
        return context
    
    def _translate_single_action(self, action: List[float], 
                                action_class: int, 
                                context: Dict) -> Optional[Dict[str, Any]]:
        """
        Translate a single VideoCAD action to a Blender command.
        
        This is a simplified version - you'll need to expand based on:
        - Action parameters (x, y coordinates, button, etc.)
        - UI context (what tool is active, what's selected)
        - Frame analysis (what's visible in the UI)
        """
        if action_class == 0:  # Mouse click
            x, y = action[1], action[2]
            button = int(action[3])
            
            # Map to Blender operation based on context
            # This is simplified - real implementation would analyze frame/context
            if context["last_operation"] == "ADD_MESH":
                # Likely adjusting transform
                return {
                    "type": "TRANSFORM",
                    "args": {
                        "translate": [x * 0.01, y * 0.01, 0]  # Scale coordinates
                    }
                }
            else:
                # Default: selection or tool activation
                return None  # Would need frame analysis
        
        elif action_class == 2:  # Keyboard shortcut
            key_code = int(action[4]) if len(action) > 4 else None
            
            # Map common Onshape shortcuts to Blender
            shortcut_map = {
                # Add common shortcuts here
                # e.g., "E" = extrude, "G" = grab/move, etc.
            }
            
            if key_code in shortcut_map:
                return shortcut_map[key_code]
        
        elif action_class == 3:  # Typed input
            # Usually numeric input for dimensions
            value = action[4] if len(action) > 4 else None
            if value and value > 0:
                return {
                    "type": "SET_PARAMETER",
                    "args": {"value": value}
                }
        
        return None
    
    def translate_high_level_operation(self, operation_type: str, 
                                      parameters: Dict) -> Dict[str, Any]:
        """
        Translate high-level CAD operations (extrude, sketch, etc.) to Blender.
        
        Args:
            operation_type: High-level operation name (e.g., "extrude", "sketch")
            parameters: Operation parameters
        
        Returns:
            Blender command dictionary
        """
        operation_map = {
            "extrude": {
                "type": "EXTRUDE",
                "args": {
                    "distance": parameters.get("distance", 1.0),
                    "direction": parameters.get("direction", "Z")
                }
            },
            "sketch": {
                "type": "SKETCH",
                "args": {
                    "plane": parameters.get("plane", "XY")
                }
            },
            "revolve": {
                "type": "REVOLVE",
                "args": {
                    "angle": parameters.get("angle", 360),
                    "axis": parameters.get("axis", "Z")
                }
            },
            "fillet": {
                "type": "FILLET",
                "args": {
                    "radius": parameters.get("radius", 0.1)
                }
            },
            "chamfer": {
                "type": "CHAMFER",
                "args": {
                    "distance": parameters.get("distance", 0.1)
                }
            },
        }
        
        if operation_type.lower() in operation_map:
            return operation_map[operation_type.lower()]
        
        return None


class VideoCADDataLoader:
    """
    Load VideoCAD data for training/fine-tuning speech-to-blender models.
    """
    
    def __init__(self, use_azure: bool = True):
        self.use_azure = use_azure
        if VIDEOCAD_AVAILABLE and use_azure:
            self.streamer = create_dataverse_streamer(use_azure=True)
        else:
            self.streamer = None
    
    def get_training_pair(self, file_id: str) -> Dict[str, Any]:
        """
        Get a training pair: (video frames, VideoCAD actions) → Blender commands.
        
        Args:
            file_id: VideoCAD file ID (e.g., "00000070")
        
        Returns:
            Dictionary with:
            - frames: Video frames
            - videocad_actions: Original VideoCAD actions
            - blender_commands: Translated Blender commands
            - target_image: Target CAD image
        """
        if not self.streamer:
            raise RuntimeError("VideoCAD streamer not available")
        
        # Get VideoCAD data
        frames, actions, base_file_id = self.streamer.get_sequence(file_id)
        
        # Translate to Blender commands
        translator = VideoCADToBlenderTranslator()
        blender_commands = translator.translate_action_sequence(actions, frames)
        
        return {
            "file_id": base_file_id,
            "frames": frames,
            "videocad_actions": actions,
            "blender_commands": blender_commands,
            "num_steps": len(frames),
        }
    
    def get_batch(self, file_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple training pairs"""
        results = []
        for file_id in file_ids:
            try:
                pair = self.get_training_pair(file_id)
                results.append(pair)
            except Exception as e:
                print(f"Error loading {file_id}: {e}")
                continue
        return results


def create_blender_command_from_videocad(videocad_action: List[float], 
                                        context: Dict = None) -> Dict[str, Any]:
    """
    Helper function to create Blender command from VideoCAD action.
    
    This can be used in your voice_to_blender.py pipeline.
    """
    translator = VideoCADToBlenderTranslator()
    action_class = int(videocad_action[0])
    return translator._translate_single_action(
        videocad_action, 
        action_class, 
        context or {}
    )


# Integration with existing speech-to-blender system
def integrate_videocad_into_pipeline():
    """
    Example integration: Use VideoCAD data to improve GPT prompts
    or train a model that maps speech → VideoCAD actions → Blender commands.
    """
    loader = VideoCADDataLoader(use_azure=True)
    
    # Get training data
    training_data = loader.get_training_pair("00000070")
    
    # Use this to:
    # 1. Fine-tune GPT on VideoCAD → Blender mappings
    # 2. Train a model: speech → VideoCAD actions → Blender
    # 3. Create few-shot examples for GPT prompts
    
    return training_data


if __name__ == "__main__":
    print("VideoCAD Integration for Speech-to-Blender")
    print("=" * 60)
    
    if not VIDEOCAD_AVAILABLE:
        print("⚠ VideoCAD modules not available")
        print("Make sure VideoCAD is set up and Azure credentials are configured")
    else:
        print("✓ VideoCAD modules loaded")
        
        # Test integration
        try:
            loader = VideoCADDataLoader(use_azure=True)
            print("\nTesting VideoCAD data loading...")
            data = loader.get_training_pair("00000070")  # Use a configured file ID
            print(f"✓ Loaded: {data['num_steps']} steps")
            print(f"  Frames: {data['frames'].shape}")
            print(f"  Actions: {len(data['videocad_actions'])}")
            print(f"  Blender commands: {len(data['blender_commands'])}")
        except Exception as e:
            print(f"✗ Error: {e}")
            print("\nMake sure:")
            print("1. VideoCAD file IDs are configured in dataverse_streamer.py")
            print("2. Azure credentials are set in .env")
            print("3. You've tested the streaming setup")

