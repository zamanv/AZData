"""Ollama API client — sync/async interface with timeout, health checks, and model listing."""

import json
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.config import (
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
)


@dataclass
class OllamaResponse:
    """Response from Ollama API."""
    success: bool
    text: str = ""
    model: str = ""
    total_duration_ms: int = 0
    eval_count: int = 0
    eval_duration_ms: int = 0
    error: Optional[str] = None


class OllamaClient:
    """Synchronous client for the Ollama local LLM API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_DEFAULT_MODEL,
        temperature: float = OLLAMA_TEMPERATURE,
        max_tokens: int = OLLAMA_MAX_TOKENS,
        timeout: int = OLLAMA_REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Session with retry
        self._session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def api_generate(self) -> str:
        return f"{self.base_url}/api/generate"

    @property
    def api_chat(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def api_tags(self) -> str:
        return f"{self.base_url}/api/tags"

    def health_check(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if Ollama is running and responsive.
        """
        try:
            resp = self._session.get(
                self.api_tags,
                timeout=5,
            )
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout, Exception):
            return False

    def list_models(self) -> List[str]:
        """List all locally available models.

        Returns:
            List of model name strings.
        """
        try:
            resp = self._session.get(self.api_tags, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> OllamaResponse:
        """Generate a completion using the /api/generate endpoint.

        Args:
            prompt: The prompt text.
            system: Optional system prompt.
            model: Override model name.
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Returns:
            OllamaResponse with the generated text.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = self._session.post(
                self.api_generate,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            return OllamaResponse(
                success=True,
                text=data.get("response", ""),
                model=data.get("model", ""),
                total_duration_ms=data.get("total_duration", 0) // 1_000_000,
                eval_count=data.get("eval_count", 0),
                eval_duration_ms=data.get("eval_duration", 0) // 1_000_000,
            )

        except requests.Timeout:
            return OllamaResponse(
                success=False,
                error=f"Request timed out after {self.timeout}s"
            )
        except requests.ConnectionError:
            return OllamaResponse(
                success=False,
                error="Cannot connect to Ollama. Is it running? (ollama serve)"
            )
        except Exception as e:
            return OllamaResponse(
                success=False,
                error=f"Ollama error: {type(e).__name__}: {e}"
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> OllamaResponse:
        """Generate a chat completion using the /api/chat endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Override model name.
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Returns:
            OllamaResponse with the generated text.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        try:
            resp = self._session.post(
                self.api_chat,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            message = data.get("message", {})
            return OllamaResponse(
                success=True,
                text=message.get("content", ""),
                model=data.get("model", ""),
                total_duration_ms=data.get("total_duration", 0) // 1_000_000,
                eval_count=data.get("eval_count", 0),
                eval_duration_ms=data.get("eval_duration", 0) // 1_000_000,
            )

        except requests.Timeout:
            return OllamaResponse(
                success=False,
                error=f"Request timed out after {self.timeout}s"
            )
        except requests.ConnectionError:
            return OllamaResponse(
                success=False,
                error="Cannot connect to Ollama. Is it running? (ollama serve)"
            )
        except Exception as e:
            return OllamaResponse(
                success=False,
                error=f"Ollama error: {type(e).__name__}: {e}"
            )

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream tokens from /api/generate.

        Yields:
            Individual text tokens as they arrive.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            with self._session.post(
                self.api_generate,
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield token
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception:
            yield "[Stream error]"

    def set_model(self, model: str) -> None:
        """Update the default model."""
        self.model = model

    def set_timeout(self, timeout: int) -> None:
        """Update the request timeout."""
        self.timeout = timeout
