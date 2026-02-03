"""
Orchestrator - Coordinates the multi-agent flow with ITERATIVE execution.

This orchestrator:
1. Plans the task (breaks down into detailed steps)
2. Executes ONE step at a time
3. Observes the result after each step
4. Continues to next step
5. Repeats until all steps are done
"""

import json
import base64
import tempfile
import os
import time
from typing import Dict, Any, Optional, List

from dataclasses import asdict

from openai import OpenAI

from agents import TaskSpec, ExecutionResult, EvaluationResult
from agents.language_translator import LanguageTranslator
from agents.code_generator import CodeGenerator
from agents.semantic_evaluator import SemanticEvaluator
from analyzers.focus_stack import get_focus_stack, FocusStack
from prompts.templates import PLANNING_PROMPT


def _capture_screen_local():
    """Capture screen using local Python libraries (PIL/mss) - runs in voice script, not Blender."""
    try:
        # Method 1: Try PIL/Pillow
        try:
            from PIL import ImageGrab
            print("[Orchestrator] [Screenshot] Trying PIL.ImageGrab...")
            screenshot = ImageGrab.grab()
            temp_path = tempfile.mktemp(suffix='.png')
            screenshot.save(temp_path, 'PNG')
            
            with open(temp_path, 'rb') as f:
                image_data = f.read()
                screenshot_data = base64.b64encode(image_data).decode('utf-8')
            
            os.remove(temp_path)
            print(f"[Orchestrator] [Screenshot] ✅ PIL.ImageGrab succeeded: {len(image_data)} bytes raw, {len(screenshot_data)} chars base64")
            return screenshot_data
        except ImportError as e:
            print(f"[Orchestrator] [Screenshot] ⚠️ PIL not available: {e}")
        except Exception as e:
            print(f"[Orchestrator] [Screenshot] ⚠️ PIL.ImageGrab failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Method 2: Try mss
        try:
            import mss
            print("[Orchestrator] [Screenshot] Trying mss...")
            with mss.mss() as sct:
                screenshot = sct.grab(sct.monitors[0])
                temp_path = tempfile.mktemp(suffix='.png')
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=temp_path)
                
                with open(temp_path, 'rb') as f:
                    image_data = f.read()
                    screenshot_data = base64.b64encode(image_data).decode('utf-8')
                
                os.remove(temp_path)
                print(f"[Orchestrator] [Screenshot] ✅ mss succeeded: {len(image_data)} bytes raw, {len(screenshot_data)} chars base64")
                return screenshot_data
        except ImportError as e:
            print(f"[Orchestrator] [Screenshot] ⚠️ mss not available: {e}")
        except Exception as e:
            print(f"[Orchestrator] [Screenshot] ⚠️ mss failed: {e}")
            import traceback
            traceback.print_exc()
        
        # If both fail, return None
        print("[Orchestrator] [Screenshot] ❌ Both PIL and mss unavailable or failed for screen capture")
        return None
    except Exception as e:
        print(f"[Orchestrator] [Screenshot] ❌ Screen capture error: {e}")
        import traceback
        traceback.print_exc()
        return None


class Orchestrator:
    """
    Coordinates the multi-agent pipeline with ITERATIVE step-by-step execution.
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
        self.max_iterations = 50  # Max steps in iterative loop
        self.enable_semantic_evaluation = True
    
    def _create_plan(self, task_spec: TaskSpec, scene_context: Dict) -> List[Dict]:
        """Create a detailed step-by-step plan for the task."""
        
        print("[Orchestrator] 📋 Creating detailed plan...")
        
        # Build context summary
        scene_summary = f"Objects: {scene_context.get('total_objects', 0)}, "
        scene_summary += f"Mesh objects: {scene_context.get('mesh_objects', 0)}"
        
        # Build prompt
        prompt = PLANNING_PROMPT.format(
            user_intent=task_spec.user_intent,
            target_concept=task_spec.target_concept or "None specified",
            target_properties=json.dumps(
                {
                    "shape": task_spec.target_properties.shape if task_spec.target_properties else None,
                    "proportions": task_spec.target_properties.proportions if task_spec.target_properties else None,
                    "surface_quality": task_spec.target_properties.surface_quality if task_spec.target_properties else None,
                    "edge_treatment": task_spec.target_properties.edge_treatment if task_spec.target_properties else None,
                } if task_spec.target_properties else {},
                indent=2
            ),
            scene_context=scene_summary
        )
        
        # Call LLM
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Create a detailed step-by-step plan for this task."}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        plan = result.get("plan", [])
        
        print(f"[Orchestrator] ✅ Plan created: {len(plan)} steps")
        for i, step in enumerate(plan[:5], 1):  # Show first 5 steps
            print(f"   Step {step.get('step_number', i)}: {step.get('description', 'N/A')}")
        if len(plan) > 5:
            print(f"   ... and {len(plan) - 5} more steps")
        
        return plan
    
    def _generate_code_for_step(self, step: Dict, geometry_state: Dict, mesh_analysis: Optional[Dict]) -> str:
        """Generate code for a single step."""
        
        from prompts.templates import CODE_GENERATOR_PROMPT
        
        step_num = step.get("step_number", 0)
        description = step.get("description", "")
        details = step.get("details", "")
        target_obj = step.get("target_object", "active")
        operation = step.get("operation", "unknown")
        
        # Build geometry context
        geometry_json = self.code_generator._build_geometry_context(geometry_state, mesh_analysis)
        
        # Get relevant API examples
        api_examples = self.code_generator._get_relevant_api_examples_for_operation(operation)
        
        # Build prompt
        prompt = CODE_GENERATOR_PROMPT.format(
            step_description=description,
            step_details=details,
            target_object=target_obj,
            geometry_json=geometry_json,
            blender_api_examples=api_examples
        )
        
        # Call LLM
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Generate code for step {step_num}: {description}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        code = result.get("code", "")
        
        return code
    
    def _observe_scene(self, rpc_client) -> Dict[str, Any]:
        """Observe the current scene state after a step."""
        
        observation = {
            "modeling_context": None,
            "mesh_analysis": None,
            "scene_analysis": None,
            "screenshot": None
        }
        
        try:
            observation["modeling_context"] = rpc_client.get_modeling_context()
            observation["scene_analysis"] = rpc_client.analyze_scene()
            
            active = observation["modeling_context"].get("active_object")
            if active and active.get("type") == "MESH":
                try:
                    observation["mesh_analysis"] = rpc_client.analyze_current_mesh()
                except Exception:
                    pass
            
            # Capture screenshot
            observation["screenshot"] = _capture_screen_local()
            
        except Exception as e:
            print(f"[Orchestrator] ⚠️ Observation error: {e}")
        
        return observation
    
    def _create_refinement_steps(self, 
                                 eval_result: EvaluationResult,
                                 task_spec: TaskSpec,
                                 rpc_client,
                                 next_step_number: int) -> List[Dict]:
        """Create refinement steps based on evaluation feedback."""
        
        if not eval_result.suggested_refinements:
            return []
        
        print(f"[Orchestrator] Creating refinement steps from {len(eval_result.suggested_refinements)} suggestions...")
        
        # Get current scene state
        modeling_context = rpc_client.get_modeling_context()
        scene_context = rpc_client.analyze_scene()
        
        # Build prompt for creating refinement steps
        suggestions_text = "\n".join([
            f"- {ref.action} on {ref.target}: {ref.reason}"
            for ref in eval_result.suggested_refinements
        ])
        
        issues_text = "\n".join(eval_result.issues_found) if eval_result.issues_found else "None"
        
        refinement_prompt = f"""You are creating refinement steps to fix issues found in a 3D model.

ORIGINAL TASK: {task_spec.user_intent}
TARGET CONCEPT: {task_spec.target_concept or "None"}

ISSUES FOUND:
{issues_text}

SUGGESTED REFINEMENTS:
{suggestions_text}

CURRENT SCENE STATE:
- Total objects: {scene_context.get('total_objects', 0)}
- Mesh objects: {scene_context.get('mesh_objects', 0)}

Create specific, actionable refinement steps. Each step should:
1. Address a specific issue
2. Be a single, focused operation
3. Use existing objects in the scene (don't recreate from scratch)
4. Calculate positions/sizes relative to existing objects

OUTPUT FORMAT (JSON only):
{{
    "plan": [
        {{
            "step_number": {next_step_number},
            "description": "Specific action to fix an issue",
            "operation": "operation_type",
            "target_object": "object_name",
            "details": "Detailed description with positioning/sizing info",
            "dependencies": []
        }}
    ]
}}

RULES:
1. Create steps that fix the specific issues mentioned
2. Make components visible and properly sized
3. Position components correctly relative to existing objects
4. Each step should be independent and executable
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": refinement_prompt},
                    {"role": "user", "content": "Create refinement steps to fix the issues."}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            refinement_steps = result.get("plan", [])
            
            return refinement_steps
            
        except Exception as e:
            print(f"[Orchestrator] ⚠️ Failed to create refinement steps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_command(self,
                        transcript: str,
                        rpc_client,
                        include_evaluation: bool = False) -> Dict[str, Any]:
        """
        Process a voice command ITERATIVELY - one step at a time.
        
        Flow:
        1. Translate language to TaskSpec
        2. Create detailed plan
        3. For each step in plan:
           a. Generate code for this step only
           b. Execute it
           c. Observe result
           d. Continue to next step
        4. Final evaluation (optional)
        """
        
        result = {
            "transcript": transcript,
            "success": False,
            "task_spec": None,
            "plan": None,
            "steps_executed": [],
            "final_evaluation": None,
            "errors": []
        }
        
        try:
            # ===== STAGE 1: Gather Initial Context =====
            print("[Orchestrator] Stage 1: Gathering context...")
            
            modeling_context = rpc_client.get_modeling_context()
            scene_context = rpc_client.analyze_scene()
            mesh_analysis = None
            
            active = modeling_context.get("active_object")
            if active and active.get("type") == "MESH":
                try:
                    mesh_analysis = rpc_client.analyze_current_mesh()
                except Exception:
                    pass
            
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
            
            # ===== STAGE 3: Create Detailed Plan =====
            print("[Orchestrator] Stage 3: Creating detailed plan...")
            
            plan = self._create_plan(task_spec, scene_context)
            result["plan"] = plan
            
            if not plan:
                result["errors"].append("Failed to create plan")
                return result
            
            # ===== STAGE 4: ITERATIVE EXECUTION =====
            print(f"[Orchestrator] Stage 4: Executing plan iteratively ({len(plan)} steps)...")
            
            completed_steps = set()
            iteration = 0
            
            while len(completed_steps) < len(plan) and iteration < self.max_iterations:
                iteration += 1
                
                # Find next step to execute (check dependencies)
                next_step = None
                for step in plan:
                    step_num = step.get("step_number", 0)
                    if step_num in completed_steps:
                        continue
                    
                    # Check if dependencies are met
                    dependencies = step.get("dependencies", [])
                    if all(dep in completed_steps for dep in dependencies):
                        next_step = step
                        break
                
                if not next_step:
                    print("[Orchestrator] ⚠️ No executable step found (dependencies not met?)")
                    break
                
                step_num = next_step.get("step_number", 0)
                step_desc = next_step.get("description", "")
                
                print(f"\n[Orchestrator] 🔄 Step {step_num}/{len(plan)}: {step_desc}")
                
                # Generate code for this step only
                try:
                    # Refresh context before generating code
                    modeling_context = rpc_client.get_modeling_context()
                    scene_context = rpc_client.analyze_scene()
                    active = modeling_context.get("active_object")
                    if active and active.get("type") == "MESH":
                        try:
                            mesh_analysis = rpc_client.analyze_current_mesh()
                        except Exception:
                            mesh_analysis = None
                    else:
                        mesh_analysis = None
                    
                    code = self._generate_code_for_step(next_step, modeling_context, mesh_analysis)
                    
                    if not code:
                        print(f"[Orchestrator] ⚠️ No code generated for step {step_num}")
                        result["errors"].append(f"Step {step_num}: No code generated")
                        continue
                    
                    print(f"[Orchestrator]   Code generated: {len(code)} chars")
                    
                except Exception as e:
                    print(f"[Orchestrator] ⚠️ Code generation failed for step {step_num}: {e}")
                    result["errors"].append(f"Step {step_num}: Code generation failed - {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Execute this step
                try:
                    print(f"[Orchestrator]   Executing step {step_num}...")
                    exec_response = rpc_client.execute({
                        "op": "execute",
                        "kwargs": {"code": code}
                    })
                    
                    if not exec_response.get("ok", False):
                        error_msg = exec_response.get("error", "Unknown error")
                        print(f"[Orchestrator] ❌ Step {step_num} execution failed: {error_msg}")
                        result["errors"].append(f"Step {step_num}: Execution failed - {error_msg}")
                        # Continue to next step anyway
                    else:
                        print(f"[Orchestrator] ✅ Step {step_num} executed successfully")
                    
                    # Wait for Blender to update
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"[Orchestrator] ❌ Step {step_num} execution exception: {e}")
                    result["errors"].append(f"Step {step_num}: Execution exception - {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next step
                
                # Observe result
                print(f"[Orchestrator]   Observing result of step {step_num}...")
                observation = self._observe_scene(rpc_client)
                
                # Record step completion
                step_result = {
                    "step_number": step_num,
                    "description": step_desc,
                    "code": code,
                    "execution_result": exec_response if 'exec_response' in locals() else None,
                    "observation": observation
                }
                result["steps_executed"].append(step_result)
                completed_steps.add(step_num)
                
                print(f"[Orchestrator] ✅ Step {step_num} completed. Progress: {len(completed_steps)}/{len(plan)}")
            
            # ===== STAGE 5: Final Evaluation and Refinement Loop =====
            refinement_attempt = 0
            max_refinement_attempts = 3
            
            while True:
                if include_evaluation and self.enable_semantic_evaluation and result["steps_executed"]:
                    print(f"[Orchestrator] Stage 5: Semantic evaluation (attempt {refinement_attempt + 1})...")
                    
                    try:
                        screenshot_base64 = _capture_screen_local()
                        
                        if screenshot_base64:
                            operations_list = [f"Step {s['step_number']}: {s['description']}" for s in result["steps_executed"]]
                            
                            eval_result = self.evaluator.evaluate(
                                task_spec=task_spec,
                                screenshot_base64=screenshot_base64,
                                operations_performed=operations_list
                            )
                            
                            if refinement_attempt == 0:
                                result["final_evaluation"] = asdict(eval_result)
                            else:
                                if "refinement_evaluations" not in result:
                                    result["refinement_evaluations"] = []
                                result["refinement_evaluations"].append(asdict(eval_result))
                            
                            score = eval_result.semantic_match_score
                            print(f"[Orchestrator] ✅ Evaluation - Score: {score:.2f}")
                            
                            if eval_result.issues_found:
                                print(f"[Orchestrator] Issues found: {eval_result.issues_found}")
                            
                            # Check if we need refinement
                            needs_refinement = (eval_result.should_retry and score < 0.8 and 
                                              refinement_attempt < max_refinement_attempts)
                            
                            if not needs_refinement:
                                if score >= 0.8:
                                    print(f"[Orchestrator] ✅ Score {score:.2f} is acceptable (>= 0.8)")
                                elif not eval_result.should_retry:
                                    print(f"[Orchestrator] ✅ Evaluator says no retry needed")
                                else:
                                    print(f"[Orchestrator] ⚠️ Max refinement attempts reached ({max_refinement_attempts})")
                                break
                            
                            # Create refinement steps based on evaluation feedback
                            print(f"\n[Orchestrator] 🔄 Creating refinement steps (attempt {refinement_attempt + 1}/{max_refinement_attempts})...")
                            
                            refinement_steps = self._create_refinement_steps(
                                eval_result, 
                                task_spec, 
                                rpc_client,
                                len(plan) + len([s for s in result.get("refinement_steps_executed", [])])
                            )
                            
                            if not refinement_steps:
                                print("[Orchestrator] ⚠️ No refinement steps created")
                                break
                            
                            print(f"[Orchestrator] ✅ Created {len(refinement_steps)} refinement steps")
                            for step in refinement_steps[:3]:
                                print(f"   - Step {step.get('step_number')}: {step.get('description')}")
                            if len(refinement_steps) > 3:
                                print(f"   ... and {len(refinement_steps) - 3} more")
                            
                            # Execute refinement steps iteratively
                            if "refinement_steps_executed" not in result:
                                result["refinement_steps_executed"] = []
                            
                            refinement_completed = set()
                            for refinement_step in refinement_steps:
                                step_num = refinement_step.get("step_number", 0)
                                step_desc = refinement_step.get("description", "")
                                
                                print(f"\n[Orchestrator] 🔄 Refinement Step {step_num}: {step_desc}")
                                
                                # Generate code for this refinement step
                                try:
                                    modeling_context = rpc_client.get_modeling_context()
                                    scene_context = rpc_client.analyze_scene()
                                    active = modeling_context.get("active_object")
                                    if active and active.get("type") == "MESH":
                                        try:
                                            mesh_analysis = rpc_client.analyze_current_mesh()
                                        except Exception:
                                            mesh_analysis = None
                                    else:
                                        mesh_analysis = None
                                    
                                    code = self._generate_code_for_step(refinement_step, modeling_context, mesh_analysis)
                                    
                                    if not code:
                                        print(f"[Orchestrator] ⚠️ No code generated for refinement step {step_num}")
                                        continue
                                    
                                    print(f"[Orchestrator]   Code generated: {len(code)} chars")
                                    
                                except Exception as e:
                                    print(f"[Orchestrator] ⚠️ Code generation failed: {e}")
                                    continue
                                
                                # Execute refinement step
                                try:
                                    print(f"[Orchestrator]   Executing refinement step {step_num}...")
                                    exec_response = rpc_client.execute({
                                        "op": "execute",
                                        "kwargs": {"code": code}
                                    })
                                    
                                    if exec_response.get("ok", False):
                                        print(f"[Orchestrator] ✅ Refinement step {step_num} executed")
                                        time.sleep(0.5)
                                    else:
                                        print(f"[Orchestrator] ❌ Refinement step {step_num} failed: {exec_response.get('error', 'Unknown')}")
                                    
                                    # Observe result
                                    observation = self._observe_scene(rpc_client)
                                    
                                    # Record step
                                    step_result = {
                                        "step_number": step_num,
                                        "description": step_desc,
                                        "code": code,
                                        "execution_result": exec_response,
                                        "observation": observation
                                    }
                                    result["refinement_steps_executed"].append(step_result)
                                    refinement_completed.add(step_num)
                                    
                                except Exception as e:
                                    print(f"[Orchestrator] ❌ Refinement step {step_num} exception: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            refinement_attempt += 1
                            print(f"[Orchestrator] ✅ Refinement attempt {refinement_attempt} completed")
                            
                        else:
                            print("[Orchestrator] ⚠️ Could not capture screenshot for evaluation")
                            break
                    except Exception as e:
                        print(f"[Orchestrator] Evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                else:
                    # No evaluation requested, break
                    break
            
            result["success"] = len(completed_steps) == len(plan) and len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(str(e))
            import traceback
            traceback.print_exc()
        
        return result


# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None

def get_orchestrator(api_key: str) -> Orchestrator:
    """Get or create the global Orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(api_key)
    return _orchestrator
