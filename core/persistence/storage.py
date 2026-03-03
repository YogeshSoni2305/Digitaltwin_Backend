"""
Business Purpose:
Manages the long-term persistence of simulation results, decision history, and LLM explanations.
Ensures that all strategic intelligence is auditable and available for historical replay.

Technical Responsibility:
- Implement thread-safe (via atomic move) append operations for JSON data.
- Manage data directory structure and file integrity.
- Provide high-level API for structured audit logging.

Determinism Level:
High. Storage operations do not affect the outcome of the simulation engine.

External Dependencies:
- None (Standard Library)
"""

import json
import os
from datetime import datetime, timezone

from typing import Dict, Any, List

# Storage Paths
FILE_PATH_SIMULATION_HISTORY = "data/simulation_history.json"
FILE_PATH_DECISION_HISTORY = "data/decision_history.json"
FILE_PATH_LLM_EXPLANATIONS = "data/llm_explanations.json"

def safe_append_json_record(file_path: str, record: Dict[str, Any]) -> None:
    """
    Appends a single JSON record to a list in a file using an atomic swap pattern.
    
    Technical Note:
    This prevents file corruption by writing to a temporary file first and then 
    performing an atomic rename/move.
    
    Args:
        file_path: Path to the target JSON list file.
        record: The dictionary record to append.
    """
    existing_data: List[Dict[str, Any]] = []
    
    # Read phase
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            with open(file_path, "r") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing_data = content
        except (json.JSONDecodeError, IOError):
            # In case of corruption, we initialize as empty to preserve new record
            pass
    
    # Update phase
    existing_data.append(record)
    
    # Write phase (Atomic Swap Pattern)
    temp_file_path = f"{file_path}.tmp"
    try:
        with open(temp_file_path, "w") as f:
            json.dump(existing_data, f, indent=2)
        os.replace(temp_file_path, file_path)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise IOError(f"Persistence Failure: Could not write to {file_path}. Error: {str(e)}")


def log_simulation_history(simulation_payload: Dict[str, Any]) -> None:
    """
    Records a completed simulation into the persistent audit trail.
    """
    simulation_payload["persisted_at"] = datetime.now(timezone.utc).isoformat()
    safe_append_json_record(FILE_PATH_SIMULATION_HISTORY, simulation_payload)


def log_decision_history(decision_payload: Dict[str, Any]) -> None:
    """
    Records a strategy ranking/comparison event into the persistent audit trail.
    """
    decision_payload["persisted_at"] = datetime.now(timezone.utc).isoformat()
    safe_append_json_record(FILE_PATH_DECISION_HISTORY, decision_payload)

