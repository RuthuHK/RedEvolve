"""Client-side sliding-window rate limiter (requests/min and tokens/min).

Groq's free tier enforces per-minute request and token caps server-side. Waiting
*before* a call is cheaper and more predictable than hitting 429s and backing off,
so the Groq client asks this limiter for permission first.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Slot:
    """One granted request in the window; ``tokens`` can be corrected after the call."""

    t: float
    tokens: int


class RateLimiter:
    """Blocks until a request of ``est_tokens`` fits under both per-window caps.

    Args:
        rpm: Max requests per window, or None for no request cap.
        tpm: Max tokens per window, or None for no token cap.
        window_s: Window length in seconds (60 for per-minute limits).
        clock: Monotonic clock, injectable for tests.
        sleep: Sleep function, injectable for tests.
    """

    def __init__(
        self,
        rpm: int | None,
        tpm: int | None,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm is not None and rpm <= 0:
            raise ValueError("rpm must be positive or None")
        if tpm is not None and tpm <= 0:
            raise ValueError("tpm must be positive or None")
        self.rpm = rpm
        self.tpm = tpm
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._slots: deque[Slot] = deque()
        self.total_wait_s = 0.0

    def _prune(self, now: float) -> None:
        while self._slots and now - self._slots[0].t >= self.window_s:
            self._slots.popleft()

    def _fits(self, tokens: int) -> bool:
        if self.rpm is not None and len(self._slots) + 1 > self.rpm:
            return False
        used = sum(s.tokens for s in self._slots)
        return self.tpm is None or used + tokens <= self.tpm

    def acquire(self, est_tokens: int = 0) -> Slot:
        """Wait until the request fits, record it, and return its slot.

        A single request larger than the whole token cap is clamped to the cap, so
        it waits for an empty window instead of blocking forever.
        """
        tokens = max(0, est_tokens)
        if self.tpm is not None:
            tokens = min(tokens, self.tpm)
        while True:
            now = self._clock()
            self._prune(now)
            if not self._slots or self._fits(tokens):
                slot = Slot(now, tokens)
                self._slots.append(slot)
                return slot
            # Wait until the oldest slot leaves the window, then re-check.
            wait = self.window_s - (now - self._slots[0].t)
            wait = max(wait, 0.01)
            self.total_wait_s += wait
            self._sleep(wait)

    @staticmethod
    def settle(slot: Slot, actual_tokens: int | None) -> None:
        """Replace a slot's estimated token count with the actual usage, when known."""
        if actual_tokens is not None:
            slot.tokens = max(0, actual_tokens)
