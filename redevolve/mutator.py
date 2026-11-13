"""LLM-backed mutation (rewrite and crossover) with per-child provenance and per-run stats.

Every call returns a :class:`MutationOutcome` naming the model that actually produced
the reply, so each GA child can record its mutator. Outcomes are classified as:

- ``ok``: a usable prompt was extracted;
- ``refused``: the mutator refused;
- ``invalid``: no usable prompt (missing tags, too short, truncated);
- ``error``: this request failed (e.g. a 400), though the service is up.

For ``refused``/``invalid``/``error`` the GA applies a deterministic operator instead,
which is recorded as such and is not another model. If the service itself is
*unavailable* (outage, rate limits that won't clear, daily quota), :class:`MutatorUnavailable`
is raised instead, so an experiment can checkpoint, stop and be resumed later with the
same mutator, rather than silently switching models mid-run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .llm import ChatClient, LLMError, LLMUnavailable, Message
from .mutator_prompts import (
    crossover_messages,
    extract_prompt,
    looks_like_refusal,
    rewrite_messages,
)


class MutatorUnavailable(RuntimeError):
    """The mutator service is unavailable; the run should checkpoint and stop for resume."""

    def __init__(self, message: str, *, daily_limit: bool = False) -> None:
        super().__init__(message)
        self.daily_limit = daily_limit


class MutationStatus(StrEnum):
    """How an LLM mutation attempt ended."""

    OK = "ok"
    REFUSED = "refused"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True)
class MutationOutcome:
    """Result of one LLM mutation attempt, with everything needed for the lineage log.

    Attributes:
        kind: ``"rewrite"`` or ``"crossover"``.
        status: See :class:`MutationStatus`.
        prompt: The new attack prompt if ``status`` is ok, else None.
        model: ``provider/model`` that produced the reply (None if the call errored).
        cached: True if the reply was replayed from the response cache.
        reply: The mutator's visible reply (kept for auditing refusal classification).
        error: Error message for ``status == "error"``.
        completion_tokens: Tokens generated, if reported.
    """

    kind: str
    status: MutationStatus
    prompt: str | None
    model: str | None
    cached: bool = False
    reply: str = ""
    error: str | None = None
    completion_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {**asdict(self), "status": str(self.status)}


@dataclass
class MutatorStats:
    """Per-run counters, for the report (refusal rate, failure rate, models used)."""

    by_status: Counter[str] = field(default_factory=Counter)
    by_model: Counter[str] = field(default_factory=Counter)
    cache_hits: int = 0

    @property
    def calls(self) -> int:
        """Total LLM mutation attempts."""
        return sum(self.by_status.values())

    def rate(self, status: MutationStatus) -> float:
        """Fraction of attempts that ended with ``status`` (0 if there were none)."""
        return self.by_status[str(status)] / self.calls if self.calls else 0.0

    def record(self, outcome: MutationOutcome) -> None:
        """Count one outcome."""
        self.by_status[str(outcome.status)] += 1
        if outcome.model:
            self.by_model[outcome.model] += 1
        self.cache_hits += outcome.cached

    def to_dict(self) -> dict[str, Any]:
        """Summary for run metadata and the report."""
        return {
            "calls": self.calls,
            "by_status": dict(self.by_status),
            "by_model": dict(self.by_model),
            "cache_hits": self.cache_hits,
            "refusal_rate": self.rate(MutationStatus.REFUSED),
            "failure_rate": 1.0 - self.rate(MutationStatus.OK) if self.calls else 0.0,
        }


class LLMMutator:
    """Rewrites and crosses over attack prompts using a chat model.

    Args:
        client: The mutator chat client. For experiment runs this must be built with
            ``build_mutator(..., allow_fallback=False)``.
        temperature: Sampling temperature.
        max_tokens: Output budget (includes reasoning tokens for reasoning models).
        stats: Shared stats object; a new one is created if omitted.
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        temperature: float,
        max_tokens: int,
        stats: MutatorStats | None = None,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = stats or MutatorStats()

    def rewrite(self, prompt: str, family: str, *, seed: int) -> MutationOutcome:
        """Ask the mutator for an improved variant of one prompt."""
        return self._run("rewrite", rewrite_messages(prompt, family), seed)

    def crossover(
        self, prompt_a: str, family_a: str, prompt_b: str, family_b: str, *, seed: int
    ) -> MutationOutcome:
        """Ask the mutator to combine two prompts."""
        return self._run(
            "crossover", crossover_messages(prompt_a, family_a, prompt_b, family_b), seed
        )

    def _run(self, kind: str, messages: list[Message], seed: int) -> MutationOutcome:
        try:
            r = self.client.chat(
                messages, temperature=self.temperature, max_tokens=self.max_tokens, seed=seed
            )
        except LLMUnavailable as e:
            raise MutatorUnavailable(str(e), daily_limit=e.daily_limit) from e
        except LLMError as e:
            outcome = MutationOutcome(kind, MutationStatus.ERROR, None, None, error=str(e))
            self.stats.record(outcome)
            return outcome

        prompt = extract_prompt(r.text, truncated=r.truncated)
        if prompt is not None:
            status = MutationStatus.OK
        elif looks_like_refusal(r.text):
            status = MutationStatus.REFUSED
        else:
            status = MutationStatus.INVALID
        outcome = MutationOutcome(
            kind,
            status,
            prompt,
            f"{r.provider}/{r.model}",
            cached=r.cached,
            reply=r.text,
            completion_tokens=r.completion_tokens,
        )
        self.stats.record(outcome)
        return outcome
