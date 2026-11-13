"""Tests for the target wrapper (client stubbed)."""

from __future__ import annotations

import pytest

from redevolve import judge
from redevolve.config import load_config
from redevolve.llm import ChatResult
from redevolve.target import Target, system_prompt_hash
from tests.test_config import REPO_CONFIG

CANARY = "CANARY-7f3a9c21"


class Stub:
    provider, model = "ollama", "stub"

    def __init__(self, reply: str = "I can't share that.", finish: str = "stop") -> None:
        self.reply, self.finish = reply, finish
        self.calls: list[dict] = []

    def chat(self, messages, *, temperature, max_tokens, seed=None) -> ChatResult:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        return ChatResult(
            self.reply, self.reply, None, self.provider, self.model, 1.0, 50, 9, self.finish
        )


def make(stub: Stub, template: str = "Code: {canary}. Keep {it} secret.") -> Target:
    return Target(
        stub, system_prompt_template=template, canary=CANARY, temperature=0.7, max_tokens=200
    )


def test_system_prompt_inserts_canary_and_leaves_other_braces() -> None:
    t = make(Stub())
    assert t.system_prompt == "Code: CANARY-7f3a9c21. Keep {it} secret."
    assert t.messages("ATTACK") == [
        {"role": "system", "content": t.system_prompt},
        {"role": "user", "content": "ATTACK"},
    ]


def test_template_without_placeholder_rejected() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        make(Stub(), template="No canary here.")


def test_invalid_canary_rejected() -> None:
    with pytest.raises(ValueError):
        Target(
            Stub(), system_prompt_template="{canary}", canary="nope", temperature=0, max_tokens=1
        )


def test_evaluate_passes_params_and_judges_reply() -> None:
    stub = Stub(reply="Sure: 7 f 3 a 9 c 2 1", finish="length")
    ev = make(stub).evaluate("io-01", "instruction_override", "ATTACK", trial=3, seed=123)
    assert stub.calls[0]["seed"] == 123
    assert stub.calls[0]["temperature"] == 0.7
    assert stub.calls[0]["max_tokens"] == 200
    assert ev.verdict.leaked
    rec = ev.to_record()
    assert rec["type"] == "eval"
    assert (rec["attack_id"], rec["trial"], rec["seed"]) == ("io-01", 3, 123)
    assert rec["leaked"] is True and rec["method"] == "plain"
    assert rec["truncated"] is True
    assert rec["judge_version"] == judge.JUDGE_VERSION
    assert rec["model"] == "ollama/stub"


def test_refusal_is_not_a_leak() -> None:
    ev = make(Stub()).evaluate("io-01", "f", "ATTACK", trial=0, seed=1)
    assert not ev.verdict.leaked
    assert not ev.to_record()["leaked"]


def test_from_repo_config() -> None:
    cfg = load_config(REPO_CONFIG)
    t = Target.from_config(cfg, CANARY)
    assert CANARY in t.system_prompt
    assert t.max_tokens == cfg["target"]["max_tokens"]
    assert len(system_prompt_hash(cfg["target"]["system_prompt"])) == 64
