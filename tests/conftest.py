"""Pytest fixtures shared across the test suite."""

from pathlib import Path

import pytest

from redevolve import cli
from redevolve.config import load_config
from tests.helpers import FakeClock

_REPO = Path(__file__).resolve().parent.parent
_CONFIGS = [_REPO / "config.yaml", *sorted((_REPO / "configs").glob("*.yaml"))]


@pytest.fixture
def clock() -> FakeClock:
    """A fresh fake clock starting at t=0."""
    return FakeClock()


def _owner_files() -> set[Path]:
    """The owner's private files named in any repo config (resolved, never opened)."""
    out: set[Path] = set()
    for c in _CONFIGS:
        d = load_config(c).get("defend", {})
        for key in ("heldout_benign_file", "heldout_custom"):
            if d.get(key):
                out.add(Path(d[key]).resolve())
    return out


@pytest.fixture(autouse=True)
def _never_read_owner_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owner's held-out benign file and custom attack set are read only by the real
    evaluation runs, never by tests."""
    forbidden = _owner_files()
    if not forbidden:
        return
    for name in ("load_benign", "load_seeds"):
        real = getattr(cli, name)

        def guarded(path, _real=real):  # type: ignore[no-untyped-def]
            if Path(path).resolve() in forbidden:
                raise AssertionError("a test tried to load one of the owner's private files")
            return _real(path)

        monkeypatch.setattr(cli, name, guarded)
