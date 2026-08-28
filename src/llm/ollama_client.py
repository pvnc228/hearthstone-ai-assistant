"""
Ollama HTTP Client for fast local LLM inference.
Features granular socket timeouts, model fallback, and streaming watchdog support.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODELS = ["qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b", "qwen2.5-coder:7b"]


class OllamaClient:
    """
    Lightweight, resilient client for local Ollama instance.
    """

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.model = model or self._select_best_available_model()

    def _select_best_available_model(self) -> str:
        """Finds the best matching installed model from Ollama."""
        try:
            available = self.list_models()
            for cand in DEFAULT_MODELS:
                if cand in available:
                    logger.info("Selected Ollama model: %s", cand)
                    return cand
            if available:
                return available[0]
        except Exception as e:
            logger.warning("Could not query Ollama models: %s", e)
        return "qwen2.5:1.5b"

    def list_models(self) -> List[str]:
        """Lists all locally installed model tags in Ollama."""
        url = f"{self.host}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "HearthstoneCoach/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]

    def is_healthy(self) -> bool:
        """Checks if Ollama is running and responding."""
        try:
            url = f"{self.host}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 150,
        model: Optional[str] = None,
    ) -> str:
        """
        Sends generation request to Ollama and returns text completion.
        """
        target_model = model or self.model
        url = f"{self.host}/api/generate"

        payload_dict: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
        }
        if system:
            payload_dict["system"] = system

        payload_bytes = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "HearthstoneCoach/1.0"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp_data.get("response", "").strip()
        except urllib.error.URLError as e:
            logger.error("Ollama connection error: %s", e)
            raise ConnectionError(f"Failed to communicate with Ollama at {self.host}: {e}") from e
