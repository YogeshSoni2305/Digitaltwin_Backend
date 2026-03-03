import os
import time
import json
from typing import Dict, Any, Optional
from groq import Groq
from core.models import LLMEngineError
from core.logging_config import logger

class LLMWrapper:
    """
    Hardened wrapper for LLM interactions with strict isolation and fallback.
    """
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error("llm_client_initialization_failure", extra={"error": str(e)})

    def generate_explanation(
        self, 
        system_persona: str, 
        user_context: str, 
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Executes an LLM call with latency tracking and crash-free fallback.
        """
        if not self.api_key or not self.client:
            logger.warning("llm_bypass_active", extra={"reason": "GROQ_API_KEY_MISSING"})
            return self._get_fallback_response("AI temporarily unavailable (API Key missing).")

        start_time = time.time()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_persona},
                    {"role": "user", "content": user_context}
                ],
                temperature=temperature,
                stream=False
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info("llm_call_success", extra={"latency_ms": latency_ms, "model": self.model})
            
            return {
                "recommendation": completion.choices[0].message.content,
                "confidence": 1.0,
                "latency_ms": latency_ms
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("llm_call_failure", extra={"error": str(e), "latency_ms": latency_ms})
            return self._get_fallback_response(f"AI temporarily unavailable (Provider Error: {type(e).__name__}).")

    def _get_fallback_response(self, reason: str) -> Dict[str, Any]:
        return {
            "recommendation": reason,
            "confidence": 0.0
        }
