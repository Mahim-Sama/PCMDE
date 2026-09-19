"""
LLM client.

Thin wrapper over the Ollama REST API (/api/generate). Single purpose:
take a prompt, return a text response. Everything about prompts, parsing,
and domain logic lives elsewhere.

This module is provider-agnostic in spirit: any backend that accepts a
POST with {"model", "prompt", "stream", "options"} and returns {"response"}
works. For other backends (OpenAI-compatible chat completions, etc.),
add a new client class here implementing the same .generate() signature.
"""

from dataclasses import dataclass
from typing import Optional

import requests


class LLMError(RuntimeError):
    """Raised on any LLM call failure."""


@dataclass
class LLMClientConfig:
    model_name: str = "deepseek-r1:70b"
    api_base: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout_seconds: int = 120
    max_tokens: int = 600


class OllamaClient:
    """Client for Ollama /api/generate endpoint."""

    def __init__(self, config: LLMClientConfig):
        self.config = config
        # Ollama's done_reason for the most recent call ("stop", "length", ...).
        # "length" means num_predict was exhausted and the reply is truncated,
        # which for reasoning models usually means it ended inside <think>.
        self.last_done_reason: Optional[str] = None

    def generate(self, prompt: str) -> str:
        """Send prompt and return the response text. Raises LLMError on failure."""
        self.last_done_reason = None
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        url = f"{self.config.api_base.rstrip('/')}/api/generate"
        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"LLM API returned status {response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
            text = body["response"]
        except (KeyError, ValueError) as exc:
            raise LLMError(f"Malformed LLM response: {exc}") from exc
        self.last_done_reason = body.get("done_reason")
        return text
