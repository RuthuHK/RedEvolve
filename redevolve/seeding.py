"""Deterministic derivation of random seeds and canaries from the run seed.

Everything random in an experiment derives from ``cfg["seed"]`` through
:func:`derive_seed`, keyed by *what* the randomness is for (e.g.
``("target", attack_id, trial)``). Two consequences:

- re-running with the same config reproduces every target sampling seed exactly;
- the seed of one call doesn't depend on how many other calls came before it, so
  skipping completed work on resume can't shift the randomness of the rest.
"""

from __future__ import annotations

import hashlib
import random

from .judge import make_canary


def derive_seed(*parts: object) -> int:
    """A stable non-negative 31-bit seed from any sequence of labels.

    Stable across processes, platforms and Python versions (it uses SHA-256, not ``hash()``).
    """
    blob = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "big") & 0x7FFFFFFF


def derive_canary(run_seed: int) -> str:
    """The dummy canary for a run, determined by the run seed."""
    return make_canary(random.Random(derive_seed("canary", run_seed)))


def target_seed(run_seed: int, attack_id: str, trial: int) -> int:
    """Sampling seed for the target's reply to ``attack_id`` on trial ``trial``."""
    return derive_seed("target", run_seed, attack_id, trial)
