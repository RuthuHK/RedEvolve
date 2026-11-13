"""Tests for the smoke command's checks (clients stubbed; no network)."""

from __future__ import annotations

import pytest

from redevolve import cli
from redevolve.config import load_config
from redevolve.llm import ChatResult, LLMError
from tests.test_config import REPO_CONFIG


def result(text: str, finish: str = "stop") -> ChatResult:
    return ChatResult(text, text, None, "p", "m", 0.1, 5, 2, finish)


@pytest.mark.parametrize(
    ("r", "ok"),
    [
        (result("pong"), True),
        (result("Pong."), True),
        (result(""), False),  # the bug the old smoke missed: empty replies passed
        (result("ping"), False),
        (result("pong", finish="length"), False),
    ],
)
def test_check_pong(r: ChatResult, ok: bool) -> None:
    assert cli.check_pong(r)[0] is ok


class Stub:
    provider = model = "stub"

    def __init__(self, text: str = "pong", fail: bool = False) -> None:
        self.text, self.fail = text, fail

    def chat(self, messages, *, temperature, max_tokens, seed=None) -> ChatResult:
        if self.fail:
            raise LLMError("boom")
        return result(self.text)


def patch_clients(monkeypatch: pytest.MonkeyPatch, *, target: Stub, mutator: Stub) -> None:
    monkeypatch.setattr(cli, "build_target", lambda cfg: target)
    monkeypatch.setattr(cli, "build_classifier", lambda cfg: Stub())
    monkeypatch.setattr(cli, "build_groq", lambda cfg, key: mutator)
    monkeypatch.setattr(cli, "load_groq_key", lambda: "unused")


def test_smoke_passes_with_good_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_clients(monkeypatch, target=Stub(), mutator=Stub())
    assert cli.cmd_smoke(load_config(REPO_CONFIG)) == 0


def test_smoke_fails_on_empty_target_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_clients(monkeypatch, target=Stub(text=""), mutator=Stub())
    assert cli.cmd_smoke(load_config(REPO_CONFIG)) == 1


def test_smoke_fails_when_groq_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_clients(monkeypatch, target=Stub(), mutator=Stub(fail=True))
    assert cli.cmd_smoke(load_config(REPO_CONFIG)) == 1
