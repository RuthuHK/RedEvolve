"""Tests for deterministic seed and canary derivation."""

from redevolve.judge import canary_core
from redevolve.seeding import derive_canary, derive_seed, target_seed


def test_derive_seed_is_stable_across_processes() -> None:
    # Pinned values: if these change, every past run's sampling seeds would silently change.
    assert derive_seed("target", 42, "io-01", 0) == 1687854977
    assert derive_canary(42) == "CANARY-d38b5fda"


def test_derive_seed_range_and_sensitivity() -> None:
    seeds = {derive_seed("x", i) for i in range(1000)}
    assert len(seeds) == 1000
    assert all(0 <= s < 2**31 for s in seeds)
    assert derive_seed("a", 1) != derive_seed("a1")  # parts are delimited, not concatenated


def test_target_seed_varies_by_attack_trial_and_run() -> None:
    base = target_seed(42, "io-01", 0)
    assert target_seed(42, "io-01", 0) == base
    assert len({base, target_seed(42, "io-01", 1), target_seed(42, "io-02", 0)}) == 3
    assert target_seed(43, "io-01", 0) != base


def test_canary_per_run_seed() -> None:
    assert derive_canary(1) != derive_canary(2)
    assert len(canary_core(derive_canary(7))) == 8
