"""Thin client for a local Ollama server, used for grounded report narration.

Talks to Ollama's /api/generate with `format` set to LLMReport's JSON schema, which
makes Ollama grammar-constrain generation to that shape (this works with any
instruction-tuned model Ollama can run — it isn't specific to one model family).
Callers (medvision/pipeline.py) always have a template-based fallback for when the
LLM is unreachable or keeps returning invalid output — the app must work with the
LLM off.
"""

from __future__ import annotations

import json
import logging

import httpx
from pydantic import ValidationError

from medvision.config import get_settings
from medvision.reasoning.schema import LLMReport

logger = logging.getLogger(__name__)


class LlmUnavailableError(RuntimeError):
    """The Ollama server could not be reached (connection refused, timed out, etc.)."""


class LlmInvalidResponseError(RuntimeError):
    """Ollama responded, but its output didn't parse as valid JSON / LLMReport."""


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings().llm
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.model = model or settings.model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.timeout_s = timeout_s if timeout_s is not None else settings.timeout_s
        self.max_retries = max_retries if max_retries is not None else settings.max_retries

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/version", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def generate_report(self, prompt: str) -> LLMReport:
        """Call Ollama once (with retries on invalid JSON) and return a validated LLMReport.

        Raises LlmUnavailableError if the server can't be reached at all, or
        LlmInvalidResponseError if every attempt returned unparseable/invalid output.
        """
        schema = LLMReport.model_json_schema()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):  # first try + max_retries retries
            try:
                response = httpx.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise LlmUnavailableError(f"Could not reach Ollama at {self.base_url}") from exc

            raw_text = response.json().get("response", "")
            try:
                parsed = json.loads(raw_text)
                return LLMReport.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning("Ollama attempt %d produced invalid output: %s", attempt, exc)

        raise LlmInvalidResponseError(
            f"Ollama returned invalid output after {self.max_retries + 1} attempts"
        ) from last_error
