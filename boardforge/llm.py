from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0


class LLMCallResult(BaseModel):
    data: dict[str, Any]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    provider: str
    model: str
    location: str = ""
    latency_ms: float = 0.0
    attempts: int = 1
    estimated_cost_usd: float | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral request usable for JSON generation and future tools."""

    system_instruction: str
    payload: dict[str, Any]
    response_model: type[BaseModel] | None = None
    tools: list[Any] = field(default_factory=list)
    temperature: float = 0.0
    max_output_tokens: int = 2048
    thinking_budget: int | None = 128


class LLMProvider(Protocol):
    enabled: bool
    provider_name: str
    model: str
    location: str

    async def generate(self, request: LLMRequest) -> LLMCallResult: ...


class LLMProviderError(RuntimeError):
    def __init__(
        self, message: str, *, attempts: int, usage: TokenUsage | None = None,
        latency_ms: float = 0.0, estimated_cost_usd: float | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.usage = usage or TokenUsage()
        self.latency_ms = latency_ms
        self.estimated_cost_usd = estimated_cost_usd
        self.response_metadata = response_metadata or {}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(os.getenv(name, str(default)))))


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        return int(status) in {408, 429, 500, 502, 503, 504}
    except (TypeError, ValueError):
        return False


class VertexGeminiLLMProvider:
    """Official Google Gen AI SDK adapter using Vertex AI and ADC."""

    provider_name = "vertex-gemini"

    def __init__(self, *, client: Any | None = None) -> None:
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        self.location = os.getenv(
            "BOARDFORGE_RELEVANCE_VERTEX_LOCATION", "global"
        ).strip()
        self.model = os.getenv(
            "BOARDFORGE_RELEVANCE_MODEL", "gemini-3.8-flash"
        ).strip()
        self.enabled = bool(self.project)
        self.timeout_seconds = float(os.getenv("BOARDFORGE_RELEVANCE_TIMEOUT_SECONDS", "30"))
        self.max_retries = _env_int("BOARDFORGE_RELEVANCE_MAX_RETRIES", 2, 0, 4)
        self.concurrency = _env_int("BOARDFORGE_RELEVANCE_CONCURRENCY", 2, 1, 8)
        self.input_usd_per_million = float(
            os.getenv("BOARDFORGE_RELEVANCE_INPUT_USD_PER_MILLION", "0.75")
        )
        self.output_usd_per_million = float(
            os.getenv("BOARDFORGE_RELEVANCE_OUTPUT_USD_PER_MILLION", "3.75")
        )
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex Gemini")
        from google import genai
        from google.genai import types

        # SDK retries are disabled here so BoardForge can count every HTTP
        # attempt and never retry a successful-but-malformed paid response.
        self._client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            http_options=types.HttpOptions(
                api_version="v1",
                timeout=max(1000, int(self.timeout_seconds * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return self._client

    @staticmethod
    def _usage(response: Any) -> TokenUsage:
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=int(getattr(metadata, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(metadata, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(metadata, "thoughts_token_count", 0) or 0),
            cached_input_tokens=int(
                getattr(metadata, "cached_content_token_count", 0) or 0
            ),
            total_tokens=int(getattr(metadata, "total_token_count", 0) or 0),
        )

    def _cost(self, usage: TokenUsage) -> float:
        billable_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        billable_output = usage.output_tokens + usage.thinking_tokens
        return round(
            (billable_input * self.input_usd_per_million / 1_000_000)
            + (billable_output * self.output_usd_per_million / 1_000_000),
            8,
        )

    @staticmethod
    def _response_metadata(response: Any) -> dict[str, Any]:
        finish_reasons = []
        for candidate in getattr(response, "candidates", None) or []:
            reason = getattr(candidate, "finish_reason", None)
            value = getattr(reason, "value", reason)
            if value is not None:
                finish_reasons.append(str(value))
        return {
            "finish_reasons": finish_reasons,
            "model_version": str(getattr(response, "model_version", "") or ""),
            "response_id": str(getattr(response, "response_id", "") or ""),
        }

    async def generate(self, request: LLMRequest) -> LLMCallResult:
        if not self.enabled:
            raise RuntimeError("Vertex Gemini provider is not configured")
        from google.genai import types

        uses_thinking_level = self.model.startswith(("gemini-3.8-", "gemini-3.7-"))
        thinking_config = (
            types.ThinkingConfig(
                thinking_level=os.getenv(
                    "BOARDFORGE_RELEVANCE_THINKING_LEVEL", "LOW"
                ).upper()
            )
            if uses_thinking_level
            else (
                types.ThinkingConfig(thinking_budget=request.thinking_budget)
                if request.thinking_budget is not None else None
            )
        )
        config = types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            temperature=None if uses_thinking_level else request.temperature,
            max_output_tokens=request.max_output_tokens,
            response_mime_type="application/json" if request.response_model else None,
            response_schema=request.response_model,
            tools=request.tools or None,
            thinking_config=thinking_config,
        )
        contents = json.dumps(request.payload, ensure_ascii=False)
        attempts = 0
        started = time.perf_counter()
        async with self._semaphore:
            while True:
                attempts += 1
                try:
                    response = await asyncio.wait_for(
                        self._get_client().aio.models.generate_content(
                            model=self.model, contents=contents, config=config
                        ),
                        timeout=self.timeout_seconds + 1,
                    )
                    break
                except Exception as exc:
                    if attempts > self.max_retries or not _retryable(exc):
                        raise LLMProviderError(
                            f"Vertex Gemini request failed after {attempts} attempt(s): "
                            f"{type(exc).__name__}: {exc}",
                            attempts=attempts,
                        ) from exc
                    delay = min(4.0, (0.4 * (2 ** (attempts - 1))))
                    await asyncio.sleep(delay + random.uniform(0, delay * 0.2))

        latency_ms = (time.perf_counter() - started) * 1000
        usage = self._usage(response)
        response_metadata = self._response_metadata(response)
        if any("MAX_TOKENS" in reason.upper() for reason in response_metadata["finish_reasons"]):
            raise LLMProviderError(
                "Vertex Gemini stopped at MAX_TOKENS before completing structured output",
                attempts=attempts, usage=usage, latency_ms=round(latency_ms, 2),
                estimated_cost_usd=self._cost(usage),
                response_metadata=response_metadata,
            )
        try:
            parsed = getattr(response, "parsed", None)
            function_calls = getattr(response, "function_calls", None) or []
            if request.tools and function_calls:
                parsed_data = {
                    "function_calls": [
                        {
                            "id": str(getattr(call, "id", "") or ""),
                            "name": str(getattr(call, "name", "") or ""),
                            "args": dict(getattr(call, "args", None) or {}),
                        }
                        for call in function_calls
                    ]
                }
            elif isinstance(parsed, BaseModel):
                parsed_data = parsed.model_dump(mode="json")
            elif isinstance(parsed, dict):
                parsed_data = parsed
            else:
                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("Vertex Gemini response contained no structured output")
                parsed_data = json.loads(text)
            if request.response_model is not None:
                parsed_data = request.response_model.model_validate(parsed_data).model_dump(mode="json")
        except Exception as exc:
            raise LLMProviderError(
                f"Vertex Gemini returned malformed structured output: {type(exc).__name__}: {exc}",
                attempts=attempts, usage=usage, latency_ms=round(latency_ms, 2),
                estimated_cost_usd=self._cost(usage),
                response_metadata=response_metadata,
            ) from exc
        return LLMCallResult(
            data=parsed_data,
            usage=usage,
            provider=self.provider_name,
            model=self.model,
            location=self.location,
            latency_ms=round(latency_ms, 2),
            attempts=attempts,
            estimated_cost_usd=self._cost(usage),
            response_metadata=response_metadata,
        )
