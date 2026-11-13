"""The target under attack: system prompt with the run's canary, plus judged evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from . import judge
from .llm import ChatClient, ChatResult, Message, build_target

CANARY_PLACEHOLDER = "{canary}"


def system_prompt_hash(template: str) -> str:
    """SHA-256 of the system prompt *template* (before the canary is inserted)."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Evaluation:
    """One attack prompt sent to the target once, with the reply and the judge's verdict."""

    attack_id: str
    family: str
    prompt: str
    trial: int
    seed: int
    result: ChatResult
    verdict: judge.Verdict

    def to_record(self) -> dict[str, Any]:
        """Flat JSON-serialisable record for the run log (raw text included for re-judging)."""
        r, v = self.result, self.verdict
        return {
            "type": "eval",
            "attack_id": self.attack_id,
            "family": self.family,
            "trial": self.trial,
            "seed": self.seed,
            "prompt": self.prompt,
            "reply": r.text,
            "raw": r.raw,
            "reasoning": r.reasoning,
            "leaked": v.leaked,
            "method": v.method,
            "partial": v.partial,
            "partial_method": v.partial_method,
            "fragment_len": v.fragment_len,
            "judge_version": judge.JUDGE_VERSION,
            "truncated": r.truncated,
            "finish_reason": r.finish_reason,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "latency_s": round(r.latency_s, 3),
            "model": f"{r.provider}/{r.model}",
        }


class Target:
    """Wraps the target model client with its canary-bearing system prompt.

    Args:
        client: Chat client for the target model.
        system_prompt_template: Must contain ``{canary}`` (other braces are left alone).
        canary: This run's canary.
        temperature: Sampling temperature for every call.
        max_tokens: Output cap for every call.

    Raises:
        ValueError: If the template has no ``{canary}`` placeholder.
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        system_prompt_template: str,
        canary: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if CANARY_PLACEHOLDER not in system_prompt_template:
            raise ValueError("target system prompt must contain the {canary} placeholder")
        judge.canary_core(canary)  # validates the canary
        self.client = client
        self.template = system_prompt_template
        self.canary = canary
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_config(cls, cfg: dict[str, Any], canary: str) -> Target:
        """Build the target from ``cfg["target"]`` for a run whose canary is ``canary``."""
        t = cfg["target"]
        return cls(
            build_target(cfg),
            system_prompt_template=t["system_prompt"],
            canary=canary,
            temperature=t["temperature"],
            max_tokens=t["max_tokens"],
        )

    @property
    def system_prompt(self) -> str:
        """The system prompt with this run's canary inserted."""
        return self.template.replace(CANARY_PLACEHOLDER, self.canary)

    def messages(self, attack_prompt: str) -> list[Message]:
        """The chat sent to the target: system prompt, then the attack as the user turn."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": attack_prompt},
        ]

    def evaluate(
        self, attack_id: str, family: str, prompt: str, *, trial: int, seed: int
    ) -> Evaluation:
        """Send one attack, judge the visible reply, and return both.

        Raises:
            LLMError: If the target can't be reached; callers checkpoint and stop.
        """
        result = self.client.chat(
            self.messages(prompt),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )
        return Evaluation(
            attack_id, family, prompt, trial, seed, result, judge.detect(self.canary, result.text)
        )
