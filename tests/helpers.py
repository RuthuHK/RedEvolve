"""Shared test helpers: a fake clock and a scripted httpx transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx


class FakeClock:
    """Deterministic clock whose ``sleep`` advances time instantly."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        self.t += dt


class Scripted:
    """An httpx transport that replays a list of responses/exceptions and records requests."""

    def __init__(self, steps: list[httpx.Response | Exception | Callable[..., Any]]) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("unexpected extra request")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def body(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)
