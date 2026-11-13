"""Tests for the sliding-window rate limiter (fake clock, no real sleeping)."""

import pytest

from redevolve.ratelimit import RateLimiter
from tests.helpers import FakeClock


def make(clock: FakeClock, rpm=None, tpm=None) -> RateLimiter:
    return RateLimiter(rpm, tpm, window_s=60, clock=clock, sleep=clock.sleep)


def test_under_limits_never_waits(clock: FakeClock) -> None:
    rl = make(clock, rpm=5, tpm=1000)
    for _ in range(5):
        rl.acquire(100)
    assert clock.sleeps == []


def test_request_cap_waits_for_window(clock: FakeClock) -> None:
    rl = make(clock, rpm=2)
    rl.acquire()
    clock.t = 10
    rl.acquire()
    clock.t = 20
    rl.acquire()  # third request must wait until the first leaves the window (t=60)
    assert clock.t == pytest.approx(60)
    assert rl.total_wait_s == pytest.approx(40)


def test_token_cap_waits(clock: FakeClock) -> None:
    rl = make(clock, tpm=1000)
    rl.acquire(700)
    rl.acquire(400)  # 700 + 400 > 1000, so it waits a full window
    assert clock.t == pytest.approx(60)


def test_settle_with_actual_usage_frees_budget(clock: FakeClock) -> None:
    rl = make(clock, tpm=1000)
    slot = rl.acquire(900)  # estimate was pessimistic
    RateLimiter.settle(slot, 200)
    rl.acquire(700)  # 200 + 700 fits now
    assert clock.sleeps == []


def test_settle_none_keeps_estimate(clock: FakeClock) -> None:
    rl = make(clock, tpm=1000)
    slot = rl.acquire(900)
    RateLimiter.settle(slot, None)
    assert slot.tokens == 900


def test_oversized_request_is_clamped_not_blocked_forever(clock: FakeClock) -> None:
    rl = make(clock, tpm=1000)
    rl.acquire(5000)  # alone in the window: allowed immediately
    rl.acquire(5000)  # waits one window, then proceeds
    assert clock.t == pytest.approx(60)


def test_no_limits_is_passthrough(clock: FakeClock) -> None:
    rl = make(clock)
    for _ in range(1000):
        rl.acquire(10**6)
    assert clock.sleeps == []


@pytest.mark.parametrize("kwargs", [{"rpm": 0, "tpm": None}, {"rpm": None, "tpm": -1}])
def test_rejects_non_positive_limits(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RateLimiter(**kwargs)
