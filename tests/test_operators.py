"""Tests for the operator interface contract (redevolve/operators.py)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from redevolve.mutator import MutationOutcome, MutationStatus
from redevolve.operators import (
    Operator,
    OperatorError,
    OperatorResult,
    OperatorSet,
    Parent,
    check_result,
)


@dataclass
class FakeOp:
    """Minimal Operator implementation; returns a fixed result."""

    name: str
    arity: int = 1
    llm: bool = False
    result: OperatorResult = field(default_factory=lambda: OperatorResult("child"))

    def apply(self, parents: Sequence[Parent], *, seed: int) -> OperatorResult:
        return self.result


def outcome(status: MutationStatus, prompt: str | None) -> MutationOutcome:
    return MutationOutcome("rewrite", status, prompt, "groq/test-model")


def test_fake_op_satisfies_protocol() -> None:
    assert isinstance(FakeOp("x"), Operator)


# --- check_result ---------------------------------------------------------------------


def test_deterministic_ok() -> None:
    check_result(FakeOp("d"), OperatorResult("a prompt"))


@pytest.mark.parametrize(
    "result",
    [
        OperatorResult(None),  # deterministic operators never fail
        OperatorResult("   \n"),  # empty prompt
        OperatorResult("p", outcome(MutationStatus.OK, "p")),  # outcome on deterministic op
    ],
)
def test_deterministic_violations(result: OperatorResult) -> None:
    with pytest.raises(OperatorError):
        check_result(FakeOp("d"), result)


def test_non_result_rejected() -> None:
    with pytest.raises(OperatorError, match="returned str"):
        check_result(FakeOp("d"), "a prompt")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "result",
    [
        OperatorResult("new", outcome(MutationStatus.OK, "new")),
        OperatorResult(None, outcome(MutationStatus.REFUSED, None)),
        OperatorResult(None, outcome(MutationStatus.INVALID, None)),
        OperatorResult(None, outcome(MutationStatus.ERROR, None)),
    ],
)
def test_llm_ok(result: OperatorResult) -> None:
    check_result(FakeOp("m", llm=True), result)


@pytest.mark.parametrize(
    "result",
    [
        OperatorResult("new"),  # no outcome
        OperatorResult(None, outcome(MutationStatus.OK, None)),  # ok but no prompt
        OperatorResult("new", outcome(MutationStatus.REFUSED, None)),  # failed but prompt set
        OperatorResult(" ", outcome(MutationStatus.OK, " ")),  # empty prompt
    ],
)
def test_llm_violations(result: OperatorResult) -> None:
    with pytest.raises(OperatorError):
        check_result(FakeOp("m", llm=True), result)


# --- OperatorSet ----------------------------------------------------------------------


def full_set() -> OperatorSet:
    return OperatorSet(
        [
            FakeOp("mut-det"),
            FakeOp("mut-llm", llm=True),
            FakeOp("cx-det", arity=2),
            FakeOp("cx-llm", arity=2, llm=True),
        ]
    )


def test_set_grouping_and_metadata() -> None:
    s = full_set()
    assert s.names == ["mut-det", "mut-llm", "cx-det", "cx-llm"]
    assert [o.name for o in s.pick(1, llm=False)] == ["mut-det"]
    assert [o.name for o in s.pick(2, llm=True)] == ["cx-llm"]
    assert s.describe()[3] == {"name": "cx-llm", "arity": 2, "llm": True}


@pytest.mark.parametrize(
    ("ops", "match"),
    [
        ([FakeOp("a"), FakeOp("a", arity=2)], "duplicate"),
        ([FakeOp("a", arity=3)], "arity"),
        ([FakeOp("")], "name"),
        ([object()], "protocol"),
    ],
)
def test_set_rejects_bad_operators(ops: list[object], match: str) -> None:
    with pytest.raises(OperatorError, match=match):
        OperatorSet(ops)  # type: ignore[arg-type]


def test_validate_for_full_set() -> None:
    full_set().validate_for(p_crossover=0.5, p_llm=0.5)


def test_validate_for_requires_deterministic_fallback() -> None:
    s = OperatorSet([FakeOp("mut-llm", llm=True)])
    with pytest.raises(OperatorError, match="deterministic mutation"):
        s.validate_for(p_crossover=0.0, p_llm=1.0)


def test_validate_for_requires_llm_op_when_p_llm_positive() -> None:
    s = OperatorSet([FakeOp("mut-det")])
    s.validate_for(p_crossover=0.0, p_llm=0.0)
    with pytest.raises(OperatorError, match="no LLM mutation"):
        s.validate_for(p_crossover=0.0, p_llm=0.3)


def test_validate_for_crossover_only_when_used() -> None:
    s = OperatorSet([FakeOp("mut-det")])
    s.validate_for(p_crossover=0.0, p_llm=0.0)  # no crossover drawn: fine
    with pytest.raises(OperatorError, match="deterministic crossover"):
        s.validate_for(p_crossover=0.2, p_llm=0.0)
