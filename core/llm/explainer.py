"""
Business Purpose:
Translates complex, numerical simulation data into board-level strategic narratives.
Provides automated executive reasoning to support data-driven decision making.

Technical Responsibility:
- Orchestrate calls to the Groq LLM (OpenAI-compatible) interface.
- Implement grounded prompt engineering to prevent hallucinations.
- Ensure all LLM outputs are logged and persisted for audit governance.
- Manage low-temperature inference for consistent output.

Determinism Level:
Low (Non-deterministic). LLM outputs vary, but grounding and low temperature ensure consistency.

External Dependencies:
- groq
"""

import os
import json
import time
import hashlib
from datetime import datetime, timezone

from typing import Dict, Any, Optional
from groq import Groq

# Internal Imports
from core.logging_config import logger
from core.persistence.storage import safe_append_json_record, FILE_PATH_LLM_EXPLANATIONS

from core.llm.wrapper import LLMWrapper

# Business Metadata
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_SIMULATION_VERSION = "v3.0-enterprise"

# Singleton instance per lifecycle
llm_service = LLMWrapper(model=DEFAULT_LLM_MODEL)

def interpret_simulation_outcome(
    target_scenario_data: Dict[str, Any],
    strategy_name: str,
    simulation_seed: int,
    engine_version: str = DEFAULT_SIMULATION_VERSION
) -> Dict[str, Any]:
    """
    Interprets a single simulation scenario using hardened LLMWrapper.
    """
    system_persona = (
        "You are a Senior Strategic Advisor. "
        "Your objective is to provide objective, data-driven interpretations of workforce simulations. "
        "RULES:\n"
        "1. GROUNDING: Use only the metrics provided in the JSON data.\n"
        "2. NO HALUCINATION: If a metric is missing, do not invent its value.\n"
        "3. TONE: Analytical, professional, and risk-aware.\n"
        "4. STRUCTURE: Use clear executive headers."
    )

    user_context = f"""
Core Simulation Snapshot (JSON):
{json.dumps(target_scenario_data, indent=2)}

Analysis Parameters:
- Human-Readable Strategy ID: {strategy_name}
- Reproducibility Seed: {simulation_seed}
- Model Engine Version: {engine_version}
"""
    return llm_service.generate_explanation(system_persona, user_context)

# Compatibility mapping for legacy calls
def generate_decision_explanation(*args, **kwargs):
    return interpret_simulation_outcome(*args, **kwargs)
