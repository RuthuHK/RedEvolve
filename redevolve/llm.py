"""One OpenAI-compatible chat wrapper for Groq (mutator) and Ollama (target)."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    latency: float
    provider: str
    model: str


def strip_think(text: str) -> str:
    """Remove qwen3-style reasoning blocks; an unclosed block is dropped entirely."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


class LLMClient:
    def __init__(self, provider: str, model: str, max_retries: int = 4, timeout: float = 180):
        if provider not in BASE_URLS:
            raise ValueError(f"unknown provider {provider!r}")
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if self.provider == "groq":
                key = os.getenv("GROQ_API_KEY", "").strip()
                if not key:
                    raise LLMError("GROQ_API_KEY not set in .env")
            else:
                key = "ollama"
            self._client = OpenAI(
                base_url=BASE_URLS[self.provider], api_key=key, max_retries=0, timeout=self.timeout
            )
        return self._client

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> Reply:
        client = self._get_client()
        if self.provider == "ollama" and self.model.startswith("qwen3"):
            messages = [dict(m) for m in messages]
            messages[0]["content"] += " /no_think"
        extra = {"seed": seed} if seed is not None else {}
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                text = strip_think(resp.choices[0].message.content or "")
                return Reply(text, time.perf_counter() - t0, self.provider, self.model)
            except RateLimitError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.provider} rate-limited: {e}") from e
                retry_after = e.response.headers.get("retry-after") if e.response else None
                try:
                    wait = float(retry_after) if retry_after else 2**attempt
                except ValueError:
                    wait = 2**attempt
                time.sleep(min(wait, 30))
            except (APIConnectionError, APITimeoutError) as e:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.provider} unreachable: {e}") from e
                time.sleep(2**attempt)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"{self.provider} error {e.status_code}: {e}") from e
        raise LLMError("unreachable")


class FallbackClient:
    """Tries the primary client; on failure uses the fallback. After 3 consecutive
    primary failures the primary is skipped for the rest of the run."""

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self.primary = primary
        self.fallback = fallback
        self._failures = 0
        self.last_error: str | None = None

    def chat(self, messages, temperature=0.7, max_tokens=256, seed=None) -> Reply:
        if self._failures < 3:
            try:
                reply = self.primary.chat(messages, temperature, max_tokens, seed)
                self._failures = 0
                return reply
            except LLMError as e:
                self._failures += 1
                self.last_error = str(e)
        return self.fallback.chat(messages, temperature, max_tokens, seed)


def build_target(cfg: dict) -> LLMClient:
    t = cfg["target"]
    return LLMClient(t["provider"], t["model"])


def build_mutator(cfg: dict) -> FallbackClient:
    m = cfg["mutator"]
    return FallbackClient(
        LLMClient(m["provider"], m["model"], max_retries=3),
        LLMClient(m["fallback_provider"], m["fallback_model"]),
    )


def build_classifier(cfg: dict) -> LLMClient:
    c = cfg["classifier"]
    return LLMClient(c["provider"], c["model"])
