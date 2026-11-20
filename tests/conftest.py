"""Pytest fixtures shared across the test suite."""

from pathlib import Path

import pytest

from redevolve import cli
from redevolve.config import load_config
from tests.helpers import FakeClock

_REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture
def clock() -> FakeClock:
    """A fresh fake clock starting at t=0."""
    return FakeClock()


@pytest.fixture(autouse=True)
def _never_read_heldout_benign(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owner's held-out benign file is read only by the real evaluation run, never by tests."""
    configured = load_config(_REPO_CONFIG).get("defend", {}).get("heldout_benign_file")
    if not configured:
        return
    forbidden = Path(configured).resolve()
    real = cli.load_benign

    def guarded(path):  # type: ignore[no-untyped-def]
        if Path(path).resolve() == forbidden:
            raise AssertionError("a test tried to load the owner's held-out benign file")
        return real(path)

    monkeypatch.setattr(cli, "load_benign", guarded)
