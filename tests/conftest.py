"""Pytest fixtures shared across the test suite."""

import pytest

from tests.helpers import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    """A fresh fake clock starting at t=0."""
    return FakeClock()
