"""Chat clients for Ollama (target, classifier) and Groq (mutator), over plain httpx.

Design notes:
- Ollama is called through its *native* ``/api/chat`` endpoint, because only that
  endpoint honours ``think`` (its OpenAI-compatible endpoint ignores ``/no_think`` and
  spends the token budget on hidden reasoning; see CLAUDE.md).
- Groq is called through its OpenAI-compatible REST endpoint, with a client-side
  rate limiter, so we stay under the free tier's tokens-per-minute cap.
- Every call returns a :class:`ChatResult` carrying the raw output, any separate
  reasoning text, token counts and a truncation flag, so callers can log everything.
- The Groq API key is held privately and never appears in reprs, logs or errors.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from dotenv import dotenv_values

from .cache import ResponseCache, cache_key
from .ratelimit import RateLimiter

Message = dict[str, str]

GROQ_URL = "https://api.groq.com/openai/v1"
OLLAMA_URL = "http://localhost:11434"

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_MAX_WAIT_S = 60.0


class LLMError(RuntimeError):
    """A chat call failed. Base class; see the two subclasses for how to react."""


class LLMUnavailable(LLMError):
    """The service itself is unavailable: retries were exhausted on network errors,
    429s or 5xx, or a daily quota is used up. Retrying the same request soon won't help.

    Attributes:
        daily_limit: True if the provider said a per-day quota is exhausted.
    """

    def __init__(self, message: str, *, daily_limit: bool = False) -> None:
        super().__init__(message)
        self.daily_limit = daily_limit


class LLMRequestError(LLMError):
    """This particular request failed (non-429 4xx, malformed response); the service is up."""


# Groq's 429 body names the exhausted window, e.g. "... on requests per day (RPD) ...".
_DAILY_LIMIT_RE = re.compile(r"per\s+day|\bRPD\b|\bTPD\b", re.I)


class _Secret:
    """Holds a credential; masked in ``repr``/``str`` so it can't leak via logs or ``vars()``."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """Return the raw value (only for building the Authorization header)."""
        return self._value

    def __repr__(self) -> str:
        return "<secret>"

    __str__ = __repr__


@dataclass(frozen=True)
class ChatResult:
    """The outcome of one chat call.

    Attributes:
        text: Visible answer, with any inline ``<think>`` blocks removed. This is what gets judged.
        raw: The answer exactly as returned by the API.
        reasoning: Separately returned reasoning/thinking text, if any (not shown to users).
        provider: ``"ollama"`` or ``"groq"``.
        model: Model name.
        latency_s: Wall-clock time of the call (0 for cache hits).
        prompt_tokens: Prompt tokens, if reported.
        completion_tokens: Generated tokens, if reported.
        finish_reason: ``"stop"``, ``"length"`` (hit max_tokens), etc.
        gen_tok_s: Generation speed, if reported (Ollama only).
        cached: True if served from the response cache.
    """

    text: str
    raw: str
    reasoning: str | None
    provider: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    gen_tok_s: float | None = None
    cached: bool = False

    @property
    def truncated(self) -> bool:
        """True if generation stopped because it hit ``max_tokens``."""
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (includes the derived ``truncated`` flag)."""
        return {**asdict(self), "truncated": self.truncated}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatResult:
        """Inverse of :meth:`to_dict` (ignores derived/unknown keys)."""
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


class ChatClient(Protocol):
    """Anything that can answer a chat request."""

    provider: str
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ChatResult: ...


def strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` blocks; an unclosed block is dropped to the end."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def load_groq_key(env_file: str | Path = ".env") -> str | None:
    """Read ``GROQ_API_KEY`` from the environment, else from ``env_file``.

    The value is returned to the caller only; it is never printed or logged here.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key and Path(env_file).is_file():
        key = (dotenv_values(env_file).get("GROQ_API_KEY") or "").strip()
    return key or None


def _backoff(attempt: int) -> float:
    return float(min(2**attempt, 30))


def _retry_after(r: httpx.Response) -> float | None:
    value = r.headers.get("retry-after")
    if value is None:
        return None
    try:
        return min(max(float(value), 0.0), _MAX_WAIT_S)
    except ValueError:
        return None


def _snippet(r: httpx.Response, n: int = 200) -> str:
    return r.text[:n].replace("\n", " ")


def _post_with_retries(
    http: httpx.Client,
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None,
    max_retries: int,
    provider: str,
    sleep: Callable[[float], None],
) -> httpx.Response:
    """POST with retries on connection errors, timeouts, 429 and 5xx; fail fast on other 4xx.

    Raises:
        LLMUnavailable: retries exhausted, or a daily quota is exhausted (not retried,
            since it won't reset within any sensible backoff).
        LLMRequestError: any other 4xx.
    """
    for attempt in range(max_retries + 1):
        last = attempt == max_retries
        try:
            r = http.post(url, json=payload, headers=headers)
        except httpx.TransportError as e:  # includes timeouts and connection errors
            if last:
                raise LLMUnavailable(f"{provider} unreachable: {type(e).__name__}: {e}") from e
            sleep(_backoff(attempt))
            continue
        if r.status_code == 429 and _DAILY_LIMIT_RE.search(r.text):
            raise LLMUnavailable(
                f"{provider} daily quota exhausted: {_snippet(r)}", daily_limit=True
            )
        if r.status_code == 429 or r.status_code >= 500:
            if last:
                raise LLMUnavailable(
                    f"{provider} HTTP {r.status_code} after {attempt + 1} attempts: {_snippet(r)}"
                )
            wait = _retry_after(r)
            sleep(wait if wait is not None else _backoff(attempt))
            continue
        if r.status_code >= 400:
            raise LLMRequestError(f"{provider} HTTP {r.status_code}: {_snippet(r)}")
        return r
    raise AssertionError("unreachable")  # pragma: no cover


class _LazyHTTP:
    """Creates the httpx client on first use (construction loads TLS certificates, ~0.3 s)."""

    def __init__(self, client: httpx.Client | None, timeout_s: float) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def get(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout_s)
        return self._client


def _json(r: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        return r.json()
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMRequestError(f"{provider} returned non-JSON: {_snippet(r)}") from e


class OllamaClient:
    """Client for a local Ollama model via the native ``/api/chat`` endpoint.

    Args:
        model: Ollama model tag.
        base_url: Ollama server URL (no ``/v1`` suffix).
        think: Sent as Ollama's ``think`` flag; None omits it (for models without thinking).
        keep_alive: How long Ollama keeps the model loaded between calls.
        timeout_s: Per-request timeout. CPU inference is slow, so this is generous.
        max_retries: Retries on transient failures.
        http: Injected httpx client (tests use ``httpx.MockTransport``).
        sleep: Injected sleep function (tests).
    """

    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = OLLAMA_URL,
        think: bool | None = False,
        keep_alive: str = "30m",
        timeout_s: float = 300.0,
        max_retries: int = 2,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.think = think
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self._http = _LazyHTTP(http, timeout_s)
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model!r}, think={self.think!r})"

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ChatResult:
        """Send one non-streamed chat request and return the parsed result."""
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if self.think is not None:
            payload["think"] = self.think
        t0 = time.perf_counter()
        r = _post_with_retries(
            self._http.get(),
            f"{self.base_url}/api/chat",
            payload=payload,
            headers=None,
            max_retries=self.max_retries,
            provider=self.provider,
            sleep=self._sleep,
        )
        latency = time.perf_counter() - t0
        d = _json(r, self.provider)
        msg = d.get("message") or {}
        raw = msg.get("content") or ""
        eval_count = d.get("eval_count")
        eval_ns = d.get("eval_duration")
        return ChatResult(
            text=strip_think(raw),
            raw=raw,
            reasoning=msg.get("thinking") or None,
            provider=self.provider,
            model=self.model,
            latency_s=latency,
            prompt_tokens=d.get("prompt_eval_count"),
            completion_tokens=eval_count,
            finish_reason=d.get("done_reason"),
            gen_tok_s=(eval_count / (eval_ns / 1e9)) if eval_count and eval_ns else None,
        )


class GroqClient:
    """Client for Groq's OpenAI-compatible chat endpoint, with client-side rate limiting.

    Args:
        model: Groq model id (e.g. ``openai/gpt-oss-20b``).
        api_key: Groq API key. Required; never exposed by this object.
        base_url: API base URL.
        reasoning_effort: Passed through for reasoning models (e.g. ``"low"``); None omits it.
        limiter: Shared :class:`RateLimiter`; None disables client-side limiting.
        timeout_s: Per-request timeout.
        max_retries: Retries on transient failures (429/5xx/network).
        http: Injected httpx client (tests).
        sleep: Injected sleep function (tests).
    """

    provider = "groq"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str = GROQ_URL,
        reasoning_effort: str | None = None,
        limiter: RateLimiter | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set (add it to .env)")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.limiter = limiter
        self.max_retries = max_retries
        self._api_key = _Secret(api_key)
        self._http = _LazyHTTP(http, timeout_s)
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"GroqClient(model={self.model!r}, reasoning_effort={self.reasoning_effort!r})"

    @staticmethod
    def estimate_tokens(messages: list[Message], max_tokens: int) -> int:
        """Rough upper estimate of a call's token cost (~4 chars/token plus the output cap)."""
        chars = sum(len(m.get("content", "")) for m in messages)
        return chars // 4 + 8 * len(messages) + max_tokens

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ChatResult:
        """Send one chat request (waiting on the rate limiter first) and return the result."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        slot = (
            self.limiter.acquire(self.estimate_tokens(messages, max_tokens))
            if self.limiter
            else None
        )
        t0 = time.perf_counter()
        r = _post_with_retries(
            self._http.get(),
            f"{self.base_url}/chat/completions",
            payload=payload,
            headers={"Authorization": f"Bearer {self._api_key.reveal()}"},
            max_retries=self.max_retries,
            provider=self.provider,
            sleep=self._sleep,
        )
        latency = time.perf_counter() - t0
        d = _json(r, self.provider)
        try:
            choice = d["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMRequestError(f"groq response missing choices: {_snippet(r)}") from e
        usage = d.get("usage") or {}
        if slot is not None:
            RateLimiter.settle(slot, usage.get("total_tokens"))
        raw = msg.get("content") or ""
        return ChatResult(
            text=strip_think(raw),
            raw=raw,
            reasoning=msg.get("reasoning") or None,
            provider=self.provider,
            model=self.model,
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )


class CachedClient:
    """Wraps a client with a persistent response cache keyed on the full request.

    Only successful calls are cached. Cache hits are marked ``cached=True`` with
    ``latency_s=0``.
    """

    def __init__(self, inner: ChatClient, cache: ResponseCache, *, namespace: str = "") -> None:
        self.inner = inner
        self.cache = cache
        self.namespace = namespace
        self.provider = inner.provider
        self.model = inner.model

    def __repr__(self) -> str:
        return f"CachedClient({self.inner!r})"

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ChatResult:
        """Return the cached result for this exact request, or call through and cache it."""
        key = cache_key(
            {
                "ns": self.namespace,
                "provider": self.inner.provider,
                "model": self.inner.model,
                "reasoning_effort": getattr(self.inner, "reasoning_effort", None),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        hit = self.cache.get(key)
        if hit is not None:
            return ChatResult.from_dict({**hit, "cached": True, "latency_s": 0.0})
        result = self.inner.chat(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed
        )
        self.cache.put(key, result.to_dict())
        return result


class FallbackClient:
    """Tries ``primary``; on :class:`LLMError` answers with ``fallback`` instead.

    After ``max_failures`` consecutive primary failures, the primary is skipped for
    ``cooldown_s`` seconds (not forever), then tried again.
    """

    def __init__(
        self,
        primary: ChatClient,
        fallback: ChatClient,
        *,
        max_failures: int = 3,
        cooldown_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_failures = max_failures
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._consecutive_failures = 0
        self._skip_until = float("-inf")
        self.provider = primary.provider
        self.model = primary.model
        self.last_error: str | None = None
        self.primary_calls = 0
        self.fallback_calls = 0

    def __repr__(self) -> str:
        return f"FallbackClient({self.primary!r} -> {self.fallback!r})"

    @property
    def primary_available(self) -> bool:
        """False while the primary is in its cooldown period."""
        return self._clock() >= self._skip_until

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ChatResult:
        """Answer with the primary if available and working, else with the fallback."""
        if self.primary_available:
            try:
                result = self.primary.chat(
                    messages, temperature=temperature, max_tokens=max_tokens, seed=seed
                )
                self._consecutive_failures = 0
                self.primary_calls += 1
                return result
            except LLMError as e:
                self.last_error = str(e)
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.max_failures:
                    self._skip_until = self._clock() + self.cooldown_s
                    self._consecutive_failures = 0
        self.fallback_calls += 1
        return self.fallback.chat(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed
        )


def _ollama_from(section: dict[str, Any], ollama_cfg: dict[str, Any], model: str) -> OllamaClient:
    return OllamaClient(
        model,
        base_url=os.environ.get("OLLAMA_URL", ollama_cfg.get("base_url", OLLAMA_URL)),
        think=section.get("think", False),
        keep_alive=ollama_cfg.get("keep_alive", "30m"),
        timeout_s=ollama_cfg.get("timeout_s", 300.0),
        max_retries=ollama_cfg.get("max_retries", 2),
    )


def build_target(cfg: dict[str, Any]) -> OllamaClient:
    """Build the target model client from ``cfg["target"]``."""
    return _ollama_from(cfg["target"], cfg.get("ollama", {}), cfg["target"]["model"])


def build_classifier(cfg: dict[str, Any]) -> OllamaClient:
    """Build the guardrail classifier client from ``cfg["classifier"]``."""
    return _ollama_from(cfg["classifier"], cfg.get("ollama", {}), cfg["classifier"]["model"])


def build_groq(cfg: dict[str, Any], api_key: str | None) -> GroqClient:
    """Build the bare Groq mutator client (rate-limited, uncached, no fallback)."""
    m = cfg["mutator"]
    return GroqClient(
        m["model"],
        api_key=api_key,
        reasoning_effort=m.get("reasoning_effort"),
        limiter=RateLimiter(m.get("rpm"), m.get("tpm")),
        timeout_s=m.get("timeout_s", 60.0),
        max_retries=m.get("max_retries", 3),
    )


def build_mutator(
    cfg: dict[str, Any], api_key: str | None, *, allow_fallback: bool = False
) -> ChatClient:
    """Build the mutator client: rate-limited Groq, cached if ``mutator.cache`` is set.

    Args:
        cfg: Full config.
        api_key: Groq API key.
        allow_fallback: If False (the default, and the only mode for experiment runs),
            a Groq outage surfaces as :class:`LLMUnavailable` so the run can pause and
            be resumed, and one run never mixes mutator models. If True (dev/smoke
            only), failures are answered by the local ``mutator.fallback_model``.
    """
    m = cfg["mutator"]
    primary: ChatClient = build_groq(cfg, api_key)
    if m.get("cache"):
        primary = CachedClient(primary, ResponseCache(m["cache"]), namespace="mutator")
    if not allow_fallback:
        return primary
    fallback = _ollama_from({"think": False}, cfg.get("ollama", {}), m["fallback_model"])
    return FallbackClient(
        primary,
        fallback,
        max_failures=m.get("fallback_after_failures", 3),
        cooldown_s=m.get("fallback_cooldown_s", 300.0),
    )
