"""
Semantic Evaluator Agent

Responsibility: Verify that execution results match user intent
- Uses VLM (GPT-4o with vision) to analyze rendered output
- Asks specific questions based on target properties
- Provides structured feedback for refinement
"""

import json
from typing import Dict, Any, List, Optional

from openai import OpenAI

from agents import TaskSpec, EvaluationResult, InferredOperation
from prompts.templates import SEMANTIC_EVALUATOR_PROMPT, build_evaluation_questions


class SemanticEvaluator:
    """
    Evaluates execution results against user intent using vision.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def evaluate(self,
                 task_spec: TaskSpec,
                 screenshot_base64: str,
                 operations_performed: List[str]) -> EvaluationResult:
        """
        Evaluate whether the result matches the user's intent.
        
        Args:
            task_spec: Original task specification
            screenshot_base64: Base64-encoded screenshot of result
            operations_performed: List of operations that were executed
            
        Returns:
            EvaluationResult with scores, issues, and suggestions
        """
        
        # Build evaluation questions
        target_props = None
        if task_spec.target_properties:
            target_props = {
                "shape": task_spec.target_properties.shape,
                "proportions": task_spec.target_properties.proportions,
                "surface_quality": task_spec.target_properties.surface_quality,
                "edge_treatment": task_spec.target_properties.edge_treatment,
            }
        
        questions = build_evaluation_questions(
            task_spec.target_concept or "",
            target_props or {}
        )
        
        # Build prompt
        prompt = SEMANTIC_EVALUATOR_PROMPT.format(
            user_intent=task_spec.user_intent,
            target_concept=task_spec.target_concept or "None specified",
            target_properties=json.dumps(target_props, indent=2) if target_props else "None specified",
            operations_performed=", ".join(operations_performed),
            evaluation_questions=questions
        )
        
        # Call VLM with image
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Evaluate this 3D model against the criteria above."
                        }
                    ]
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Parse into EvaluationResult
        suggested_refinements = []
        for ref in result.get("suggested_refinements", []):
            suggested_refinements.append(InferredOperation(
                action=ref.get("action", "unknown"),
                target=ref.get("target", "active"),
                parameters=ref.get("parameters", {}),
                reason=ref.get("reason", "")
            ))
        
        return EvaluationResult(
            semantic_match_score=result.get("overall_match_score", 0.5),
            questions_asked=result.get("question_answers", {}),
            issues_found=result.get("issues_found", []),
            suggested_refinements=suggested_refinements,
            should_retry=result.get("should_retry", False)
        )
