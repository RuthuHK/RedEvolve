"""The pluggable mutation/crossover operator interface used by the GA engine (``ga.py``).

This module defines the *contract* only. It contains no attack-mutation logic: real
operators are added separately and are registered by name in an :class:`OperatorSet`.

Contract (enforced by :class:`OperatorSet` and the engine):

- An operator has a unique ``name`` (logged as provenance in every child record; if an
  operator's behaviour changes, give it a new name, e.g. ``foo-v2``, so old runs stay
  interpretable and can't be resumed with different behaviour), an ``arity`` (1 = mutation,
  2 = crossover) and an ``llm`` flag.
- ``apply(parents, seed=...)`` receives exactly ``arity`` parents and an integer seed derived
  from the run seed, the GA seed and the child id (stable across resume).
- A **deterministic** operator (``llm = False``) must be a pure function of
  ``(parents, seed)``, must not do I/O, and must always return a non-empty prompt. It never
  fails, which is why it can serve as the fallback for a failed LLM operator.
- An **LLM** operator (``llm = True``) returns its :class:`~redevolve.mutator.MutationOutcome`
  in ``OperatorResult.outcome``. If the outcome is not ``ok`` it returns ``prompt=None``, and
  the engine applies a deterministic operator of the same arity instead, recorded as a
  fallback. If the mutator service is unavailable it raises
  :class:`~redevolve.mutator.MutatorUnavailable`; the engine checkpoints and pauses the run.
  It must not switch models: the engine checks ``outcome.model`` against the run's mutator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .mutator import MutationOutcome, MutationStatus


class OperatorError(ValueError):
    """An operator or operator set violates the contract above."""


@dataclass(frozen=True)
class Parent:
    """What an operator sees of a parent: its id, prompt and attack family."""

    id: str
    prompt: str
    family: str


@dataclass(frozen=True)
class OperatorResult:
    """The output of one operator application.

    Attributes:
        prompt: The child prompt, or None if an LLM operator failed (refused/invalid/error).
        outcome: The LLM mutation outcome (LLM operators only; None for deterministic ones).
    """

    prompt: str | None
    outcome: MutationOutcome | None = None


@runtime_checkable
class Operator(Protocol):
    """A mutation (arity 1) or crossover (arity 2) operator. See the module docstring."""

    name: str
    arity: int
    llm: bool

    def apply(self, parents: Sequence[Parent], *, seed: int) -> OperatorResult:
        """Produce one child prompt from ``arity`` parents."""
        ...


def check_result(op: Operator, result: OperatorResult) -> None:
    """Validate an operator's result against the contract.

    Raises:
        OperatorError: On any violation (a programming error in the operator).
    """
    if not isinstance(result, OperatorResult):
        raise OperatorError(f"operator {op.name!r} returned {type(result).__name__}")
    if op.llm:
        if result.outcome is None:
            raise OperatorError(f"LLM operator {op.name!r} returned no MutationOutcome")
        ok = result.outcome.status == MutationStatus.OK
        if ok != (result.prompt is not None):
            raise OperatorError(
                f"LLM operator {op.name!r}: prompt must be set iff status is ok "
                f"(status={result.outcome.status}, prompt set={result.prompt is not None})"
            )
    elif result.outcome is not None:
        raise OperatorError(f"deterministic operator {op.name!r} returned a MutationOutcome")
    if result.prompt is not None and not result.prompt.strip():
        raise OperatorError(f"operator {op.name!r} returned an empty prompt")
    if not op.llm and result.prompt is None:
        raise OperatorError(f"deterministic operator {op.name!r} failed to produce a prompt")


class OperatorSet:
    """A validated, name-unique collection of operators, grouped by arity and kind.

    Raises:
        OperatorError: If names repeat, an arity isn't 1 or 2, or an object doesn't
            implement :class:`Operator`.
    """

    def __init__(self, operators: Iterable[Operator]) -> None:
        ops = list(operators)
        names = [getattr(o, "name", None) for o in ops]
        for o in ops:
            if not isinstance(o, Operator):
                raise OperatorError(f"{o!r} does not implement the Operator protocol")
            if o.arity not in (1, 2):
                raise OperatorError(f"operator {o.name!r}: arity must be 1 or 2, got {o.arity}")
            if not o.name or not isinstance(o.name, str):
                raise OperatorError(f"operator {o!r} needs a non-empty string name")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise OperatorError(f"duplicate operator names: {dupes}")
        self.operators: tuple[Operator, ...] = tuple(ops)

    def pick(self, arity: int, *, llm: bool) -> tuple[Operator, ...]:
        """Operators of one arity and kind, in registration order."""
        return tuple(o for o in self.operators if o.arity == arity and o.llm == llm)

    @property
    def names(self) -> list[str]:
        """Operator names in registration order (recorded in run metadata)."""
        return [o.name for o in self.operators]

    def describe(self) -> list[dict[str, object]]:
        """Name, arity and kind of every operator (recorded in run metadata)."""
        return [{"name": o.name, "arity": o.arity, "llm": o.llm} for o in self.operators]

    def validate_for(self, *, p_crossover: float, p_llm: float) -> None:
        """Check that every operator kind the GA can draw with these probabilities exists.

        A deterministic operator of each arity used is always required (it is the fallback
        for a failed LLM operator).

        Raises:
            OperatorError: If a needed kind is missing.
        """
        arities = [a for a, p in ((1, 1.0 - p_crossover), (2, p_crossover)) if p > 0]
        for a in arities:
            what = "mutation" if a == 1 else "crossover"
            if not self.pick(a, llm=False):
                raise OperatorError(f"no deterministic {what} operator (needed as fallback)")
            if p_llm > 0 and not self.pick(a, llm=True):
                raise OperatorError(f"p_llm_mutation > 0 but no LLM {what} operator")
