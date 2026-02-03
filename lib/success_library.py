"""
Success Library: Store and learn from successful Blender runs.
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import base64

class BlenderSuccessLibrary:
    """
    Store successful Blender runs - learn from what actually works.
    """
    
    def __init__(self, storage_path: str = "success_library"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.candidates_path = self.storage_path / "candidates"
        self.candidates_path.mkdir(exist_ok=True)
        self.confirmed_path = self.storage_path / "confirmed"
        self.confirmed_path.mkdir(exist_ok=True)
    
    def store_run(self, target_object: str, screenshot: bytes, commands: list,
                  quality_score: float, target_match_score: float,
                  scene_analysis: dict, mesh_analysis: dict,
                  transcript: str, summary: str, iterations: int,
                  status: str = "candidate") -> str:
        """
        Store a run (candidate or confirmed success).
        
        Args:
            status: "candidate" (auto-detected) or "confirmed" (manually marked)
        
        Returns:
            run_id: Unique identifier for this run
        """
        run_id = f"{target_object.replace(' ', '_')}_{int(time.time())}"
        storage_dir = self.confirmed_path if status == "confirmed" else self.candidates_path
        
        # Save screenshot
        screenshot_path = storage_dir / f"{run_id}_screenshot.png"
        with open(screenshot_path, "wb") as f:
            f.write(screenshot)
        
        # Prepare metadata
        metadata = {
            "run_id": run_id,
            "target_object": target_object,
            "timestamp": time.time(),
            "status": status,
            "quality_score": quality_score,
            "target_match_score": target_match_score,
            "iterations": iterations,
            "commands_count": len(commands),
            "commands": commands,  # Store full command list
            "transcript": transcript,
            "summary": summary,
            "scene_analysis": scene_analysis,
            "mesh_analysis": mesh_analysis,
            "screenshot_path": str(screenshot_path.relative_to(self.storage_path))
        }
        
        # Save metadata
        metadata_path = storage_dir / f"{run_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return run_id
    
    def auto_detect_success(self, target_object: str, quality_score: float,
                           target_match_score: float, finished_successfully: bool) -> bool:
        """
        Auto-detect if a run should be considered a success candidate.
        
        Criteria:
        - Finished successfully (not max iterations)
        - Quality score >= 0.7 OR target_match_score >= 0.7
        - Has target_object (not empty)
        """
        if not target_object:
            return False
        
        if not finished_successfully:
            return False
        
        # High quality threshold for auto-detection
        if quality_score >= 0.7 or target_match_score >= 0.7:
            return True
        
        return False
    
    def get_similar_successful_runs(self, target_object: str, 
                                   min_quality: float = 0.6) -> List[Dict]:
        """Get confirmed successful runs for this object type"""
        results = []
        
        # Check confirmed successes
        for metadata_file in self.confirmed_path.glob("*_metadata.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    if target_object.lower() in data.get("target_object", "").lower():
                        if data.get("quality_score", 0) >= min_quality:
                            results.append(data)
            except Exception as e:
                print(f"Error loading {metadata_file}: {e}")
                continue
        
        # Also check high-quality candidates
        for metadata_file in self.candidates_path.glob("*_metadata.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    if target_object.lower() in data.get("target_object", "").lower():
                        if data.get("quality_score", 0) >= 0.8:  # Very high threshold
                            results.append(data)
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x.get("quality_score", 0), reverse=True)
    
    def promote_candidate_to_confirmed(self, run_id: str) -> bool:
        """Promote a candidate to confirmed success"""
        candidate_file = self.candidates_path / f"{run_id}_metadata.json"
        if not candidate_file.exists():
            return False
        
        # Load candidate
        with open(candidate_file) as f:
            data = json.load(f)
        
        # Move to confirmed
        confirmed_file = self.confirmed_path / f"{run_id}_metadata.json"
        data["status"] = "confirmed"
        data["promoted_at"] = time.time()
        
        with open(confirmed_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        # Move screenshot
        screenshot_name = f"{run_id}_screenshot.png"
        candidate_screenshot = self.candidates_path / screenshot_name
        if candidate_screenshot.exists():
            confirmed_screenshot = self.confirmed_path / screenshot_name
            candidate_screenshot.rename(confirmed_screenshot)
        
        # Remove candidate
        candidate_file.unlink()
        
        return True
    
    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run (candidate or confirmed) - for removing falsely marked successes.
        
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        
        # Try to delete from candidates
        candidate_meta = self.candidates_path / f"{run_id}_metadata.json"
        candidate_screenshot = self.candidates_path / f"{run_id}_screenshot.png"
        
        if candidate_meta.exists():
            candidate_meta.unlink()
            deleted = True
        if candidate_screenshot.exists():
            candidate_screenshot.unlink()
        
        # Try to delete from confirmed
        confirmed_meta = self.confirmed_path / f"{run_id}_metadata.json"
        confirmed_screenshot = self.confirmed_path / f"{run_id}_screenshot.png"
        
        if confirmed_meta.exists():
            confirmed_meta.unlink()
            deleted = True
        if confirmed_screenshot.exists():
            confirmed_screenshot.unlink()
        
        return deleted
    
    def get_all_runs(self, include_candidates: bool = True, 
                    include_confirmed: bool = True) -> List[Dict]:
        """Get all stored runs, sorted by timestamp (newest first)"""
        all_runs = []
        
        if include_candidates:
            for metadata_file in self.candidates_path.glob("*_metadata.json"):
                try:
                    with open(metadata_file) as f:
                        data = json.load(f)
                        all_runs.append(data)
                except:
                    continue
        
        if include_confirmed:
            for metadata_file in self.confirmed_path.glob("*_metadata.json"):
                try:
                    with open(metadata_file) as f:
                        data = json.load(f)
                        all_runs.append(data)
                except:
                    continue
        
        # Sort by timestamp (newest first)
        return sorted(all_runs, key=lambda x: x.get("timestamp", 0), reverse=True)
    
    def get_last_run_info(self) -> Optional[Dict]:
        """Get info about the most recent run (for manual marking)"""
        all_runs = self.get_all_runs()
        return all_runs[0] if all_runs else None
    
    def get_run_by_id(self, run_id: str) -> Optional[Dict]:
        """Get a specific run by ID"""
        # Check candidates
        candidate_file = self.candidates_path / f"{run_id}_metadata.json"
        if candidate_file.exists():
            with open(candidate_file) as f:
                return json.load(f)
        
        # Check confirmed
        confirmed_file = self.confirmed_path / f"{run_id}_metadata.json"
        if confirmed_file.exists():
            with open(confirmed_file) as f:
                return json.load(f)
        
        return None
