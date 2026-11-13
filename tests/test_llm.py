"""Tests for the LLM clients. All HTTP is mocked with httpx.MockTransport."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from redevolve.cache import ResponseCache
from redevolve.llm import (
    CachedClient,
    ChatResult,
    FallbackClient,
    GroqClient,
    LLMError,
    OllamaClient,
    load_groq_key,
    strip_think,
)
from redevolve.ratelimit import RateLimiter
from tests.helpers import FakeClock, Scripted

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
FAKE_KEY = "gsk_test_" + "x" * 40


def ollama_ok(content: str = "hello", done_reason: str = "stop", **extra) -> httpx.Response:
    body = {
        "message": {"role": "assistant", "content": content, **extra},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 12,
        "eval_count": 20,
        "eval_duration": 2_000_000_000,
    }
    return httpx.Response(200, json=body)


def groq_ok(content: str = "hello", finish: str = "stop", total: int = 50) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": total},
        },
    )


def ollama(script: Scripted, **kw) -> OllamaClient:
    return OllamaClient("m", http=script.client(), sleep=lambda s: None, **kw)


def groq(script: Scripted, **kw) -> GroqClient:
    kw.setdefault("sleep", lambda s: None)
    return GroqClient("g", api_key=FAKE_KEY, http=script.client(), **kw)


# --- strip_think --------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>hmm</think>answer", "answer"),
        ("<think>\n\n</think>\n\nanswer", "answer"),
        ("a<think>x</think>b<think>y</think>c", "abc"),
        ("visible <think>cut off mid-thought", "visible"),
        ("no tags at all", "no tags at all"),
    ],
)
def test_strip_think(raw: str, expected: str) -> None:
    assert strip_think(raw) == expected


# --- OllamaClient --------------------------------------------------------------


def test_ollama_uses_native_api_with_think_flag_and_options() -> None:
    s = Scripted([ollama_ok()])
    ollama(s, think=False, keep_alive="5m").chat(MSGS, temperature=0.3, max_tokens=99, seed=7)
    req = s.requests[0]
    assert req.url.path == "/api/chat"
    body = s.body()
    assert body["think"] is False
    assert body["stream"] is False
    assert body["keep_alive"] == "5m"
    assert body["options"] == {"temperature": 0.3, "num_predict": 99, "seed": 7}
    assert body["messages"] == MSGS  # the system prompt is NOT modified (no "/no_think")


def test_ollama_think_none_omits_flag_and_seed_none_omits_seed() -> None:
    s = Scripted([ollama_ok()])
    ollama(s, think=None).chat(MSGS, temperature=0, max_tokens=5)
    assert "think" not in s.body()
    assert "seed" not in s.body()["options"]


def test_ollama_strips_v1_suffix_from_base_url() -> None:
    s = Scripted([ollama_ok()])
    ollama(s, base_url="http://host:1/v1/").chat(MSGS, temperature=0, max_tokens=5)
    assert str(s.requests[0].url) == "http://host:1/api/chat"


def test_ollama_parses_result_and_keeps_reasoning_separate() -> None:
    s = Scripted([ollama_ok("<think>x</think>The answer", thinking="hidden plan")])
    r = ollama(s).chat(MSGS, temperature=0, max_tokens=5)
    assert r.text == "The answer"
    assert r.raw == "<think>x</think>The answer"
    assert r.reasoning == "hidden plan"
    assert (r.prompt_tokens, r.completion_tokens) == (12, 20)
    assert r.gen_tok_s == pytest.approx(10.0)
    assert not r.truncated


def test_ollama_flags_truncation() -> None:
    s = Scripted([ollama_ok(done_reason="length")])
    assert ollama(s).chat(MSGS, temperature=0, max_tokens=5).truncated


def test_ollama_retries_transient_errors_then_succeeds() -> None:
    s = Scripted([httpx.ConnectError("down"), httpx.Response(503), ollama_ok("ok")])
    r = ollama(s, max_retries=2).chat(MSGS, temperature=0, max_tokens=5)
    assert r.text == "ok"
    assert len(s.requests) == 3


def test_ollama_gives_up_after_max_retries() -> None:
    s = Scripted([httpx.ConnectError("down")] * 3)
    with pytest.raises(LLMError, match="unreachable"):
        ollama(s, max_retries=2).chat(MSGS, temperature=0, max_tokens=5)


def test_ollama_does_not_retry_client_errors() -> None:
    s = Scripted([httpx.Response(404, text="model not found")])
    with pytest.raises(LLMError, match="404"):
        ollama(s, max_retries=5).chat(MSGS, temperature=0, max_tokens=5)
    assert len(s.requests) == 1


def test_ollama_non_json_is_llm_error() -> None:
    s = Scripted([httpx.Response(200, text="<html>")])
    with pytest.raises(LLMError, match="non-JSON"):
        ollama(s).chat(MSGS, temperature=0, max_tokens=5)


# --- GroqClient ----------------------------------------------------------------


def test_groq_request_shape() -> None:
    s = Scripted([groq_ok()])
    groq(s, reasoning_effort="low").chat(MSGS, temperature=0.9, max_tokens=300, seed=3)
    req = s.requests[0]
    assert req.url.path.endswith("/chat/completions")
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert s.body() == {
        "model": "g",
        "messages": MSGS,
        "temperature": 0.9,
        "max_tokens": 300,
        "seed": 3,
        "reasoning_effort": "low",
    }


def test_groq_parses_usage_reasoning_and_truncation() -> None:
    resp = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "", "reasoning": "thinking..."}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        },
    )
    r = groq(Scripted([resp])).chat(MSGS, temperature=0, max_tokens=10)
    assert r.text == ""
    assert r.reasoning == "thinking..."
    assert r.truncated
    assert (r.prompt_tokens, r.completion_tokens) == (5, 10)


def test_groq_429_honours_retry_after() -> None:
    sleeps: list[float] = []
    s = Scripted([httpx.Response(429, headers={"retry-after": "7"}), groq_ok("done")])
    r = groq(s, sleep=sleeps.append).chat(MSGS, temperature=0, max_tokens=5)
    assert r.text == "done"
    assert sleeps == [7.0]


def test_groq_retry_after_is_capped() -> None:
    sleeps: list[float] = []
    s = Scripted([httpx.Response(429, headers={"retry-after": "86400"}), groq_ok()])
    groq(s, sleep=sleeps.append).chat(MSGS, temperature=0, max_tokens=5)
    assert sleeps == [60.0]


def test_groq_persistent_429_raises() -> None:
    s = Scripted([httpx.Response(429)] * 3)
    with pytest.raises(LLMError, match="429"):
        groq(s, max_retries=2).chat(MSGS, temperature=0, max_tokens=5)


def test_groq_missing_choices_is_llm_error() -> None:
    s = Scripted([httpx.Response(200, json={"error": "weird"})])
    with pytest.raises(LLMError, match="choices"):
        groq(s).chat(MSGS, temperature=0, max_tokens=5)


def test_groq_requires_key() -> None:
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient("g", api_key=None)


def test_groq_key_never_in_repr_or_errors() -> None:
    s = Scripted([httpx.Response(401, text="invalid api key")])
    client = groq(s)
    assert FAKE_KEY not in repr(client)
    assert FAKE_KEY not in str(vars(client))  # e.g. if a client ever gets dumped into run metadata
    assert FAKE_KEY not in f"{client._api_key}"
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS, temperature=0, max_tokens=5)
    assert FAKE_KEY not in str(exc.value)


def test_groq_uses_limiter_and_settles_actual_usage(clock: FakeClock) -> None:
    limiter = RateLimiter(None, 1000, clock=clock, sleep=clock.sleep)
    s = Scripted([groq_ok(total=40), groq_ok(total=40)])
    client = groq(s, limiter=limiter)
    client.chat(MSGS, temperature=0, max_tokens=900)  # estimate ~900, settled to 40
    client.chat(MSGS, temperature=0, max_tokens=900)  # would wait if not settled
    assert clock.sleeps == []


def test_estimate_tokens_includes_output_cap() -> None:
    est = GroqClient.estimate_tokens([{"role": "user", "content": "x" * 400}], 300)
    assert est >= 400 // 4 + 300


# --- CachedClient ---------------------------------------------------------------


def test_cache_hit_skips_call_and_marks_cached(tmp_path: Path) -> None:
    s = Scripted([groq_ok("first")])
    c = CachedClient(groq(s), ResponseCache(tmp_path / "c.sqlite"))
    r1 = c.chat(MSGS, temperature=0.9, max_tokens=50, seed=1)
    r2 = c.chat(MSGS, temperature=0.9, max_tokens=50, seed=1)
    assert len(s.requests) == 1
    assert (r1.cached, r2.cached) == (False, True)
    assert r2.text == "first"
    assert r2.latency_s == 0.0


def test_cache_key_depends_on_seed_and_params(tmp_path: Path) -> None:
    s = Scripted([groq_ok("a"), groq_ok("b"), groq_ok("c")])
    c = CachedClient(groq(s), ResponseCache(tmp_path / "c.sqlite"))
    c.chat(MSGS, temperature=0.9, max_tokens=50, seed=1)
    c.chat(MSGS, temperature=0.9, max_tokens=50, seed=2)
    c.chat(MSGS, temperature=0.5, max_tokens=50, seed=1)
    assert len(s.requests) == 3


def test_cache_does_not_store_failures(tmp_path: Path) -> None:
    s = Scripted([httpx.Response(400, text="bad"), groq_ok("ok")])
    c = CachedClient(groq(s), ResponseCache(tmp_path / "c.sqlite"))
    with pytest.raises(LLMError):
        c.chat(MSGS, temperature=0, max_tokens=5)
    assert c.chat(MSGS, temperature=0, max_tokens=5).text == "ok"


def test_chat_result_roundtrip() -> None:
    r = ChatResult("t", "raw", None, "groq", "g", 1.5, 1, 2, "length", None)
    d = r.to_dict()
    assert d["truncated"] is True
    assert ChatResult.from_dict(d) == r


# --- FallbackClient --------------------------------------------------------------


class StubClient:
    """Minimal ChatClient that fails a set number of times, then answers."""

    def __init__(self, name: str, fail_times: int = 0) -> None:
        self.provider, self.model = name, name
        self.fail_times = fail_times
        self.calls = 0

    def chat(self, messages, *, temperature, max_tokens, seed=None) -> ChatResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise LLMError(f"{self.provider} failed")
        return ChatResult(self.provider, self.provider, None, self.provider, self.model, 0.0)


def call(c: FallbackClient) -> ChatResult:
    return c.chat(MSGS, temperature=0, max_tokens=5)


def test_fallback_used_when_primary_fails(clock: FakeClock) -> None:
    fb = FallbackClient(StubClient("p", fail_times=1), StubClient("f"), clock=clock)
    assert call(fb).provider == "f"
    assert call(fb).provider == "p"  # recovered
    assert fb.last_error == "p failed"
    assert (fb.primary_calls, fb.fallback_calls) == (1, 1)


def test_primary_skipped_during_cooldown_then_retried(clock: FakeClock) -> None:
    primary = StubClient("p", fail_times=3)
    fb = FallbackClient(primary, StubClient("f"), max_failures=3, cooldown_s=300, clock=clock)
    for _ in range(3):
        assert call(fb).provider == "f"
    assert not fb.primary_available
    clock.t = 299
    assert call(fb).provider == "f"
    assert primary.calls == 3  # not tried during cooldown
    clock.t = 300
    assert call(fb).provider == "p"  # tried again after cooldown, and works
    assert fb.primary_available


def test_fallback_failure_propagates(clock: FakeClock) -> None:
    fb = FallbackClient(StubClient("p", 99), StubClient("f", 99), clock=clock)
    with pytest.raises(LLMError, match="f failed"):
        call(fb)


# --- load_groq_key ---------------------------------------------------------------


def test_load_groq_key_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    assert load_groq_key(env) == FAKE_KEY


def test_load_groq_key_env_var_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "from_env")
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from_file\n", encoding="utf-8")
    assert load_groq_key(env) == "from_env"


@pytest.mark.parametrize("content", ["GROQ_API_KEY=\n", "OTHER=1\n", None])
def test_load_groq_key_missing_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env = tmp_path / ".env"
    if content is not None:
        env.write_text(content, encoding="utf-8")
    assert load_groq_key(env) is None
