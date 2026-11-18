"""Tests for the statistics helpers."""

from __future__ import annotations

import pytest

from redevolve.stats import Rate, cluster_bootstrap_ci, cohen_kappa, mcnemar_exact, wilson


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.27753),  # upper bound = z^2 / (n + z^2)
        (5, 10, 0.23659, 0.76341),
        (10, 10, 0.72247, 1.0),
        (1, 300, 0.000589, 0.018637),
    ],
)
def test_wilson_known_values(k: int, n: int, lo: float, hi: float) -> None:
    got = wilson(k, n)
    assert got == pytest.approx((lo, hi), abs=1e-5)


def test_wilson_is_symmetric_and_contains_the_estimate() -> None:
    for n in (1, 7, 10, 60):
        for k in range(n + 1):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0
            lo2, hi2 = wilson(n - k, n)
            assert (lo, hi) == pytest.approx((1 - hi2, 1 - lo2))


def test_wilson_endpoints_exact_at_zero_and_all() -> None:
    for n in (1, 7, 10, 60, 300):
        assert wilson(0, n)[0] == 0.0
        assert wilson(n, n)[1] == 1.0


def test_wilson_empty_and_invalid() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    for k, n in [(-1, 5), (6, 5), (0, -1)]:
        with pytest.raises(ValueError):
            wilson(k, n)


def test_rate_formats_and_serialises() -> None:
    r = Rate(3, 10)
    assert r.p == 0.3
    d = r.to_dict()
    assert (d["k"], d["n"], d["rate"]) == (3, 10, 0.3)
    assert d["ci_low"] < 0.3 < d["ci_high"]
    assert r.fmt().startswith("30.0% [") and r.fmt().endswith("(3/10)")
    assert Rate(0, 0).p == 0.0


def test_cluster_bootstrap_deterministic_and_bounded() -> None:
    rates = [0.0, 0.0, 0.1, 0.9, 1.0, 0.2]
    a = cluster_bootstrap_ci(rates, seed=1, n_boot=2000)
    assert a == cluster_bootstrap_ci(rates, seed=1, n_boot=2000)
    mean = sum(rates) / len(rates)
    assert 0.0 <= a[0] < mean < a[1] <= 1.0


def test_cluster_bootstrap_wider_than_pooled_wilson_when_attacks_differ() -> None:
    # 6 attacks x 10 trials, all-or-nothing: pooled Wilson ignores the clustering.
    rates = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    lo, hi = cluster_bootstrap_ci(rates, seed=3, n_boot=4000)
    wlo, whi = wilson(20, 60)
    assert hi - lo > whi - wlo


def test_cluster_bootstrap_constant_and_errors() -> None:
    assert cluster_bootstrap_ci([0.0] * 6, seed=0, n_boot=100) == (0.0, 0.0)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([], seed=0)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([0.5], seed=0, level=1.0)


def test_cohen_kappa() -> None:
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "n", "n"]) == pytest.approx(0.5)
    assert cohen_kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert cohen_kappa(["n"] * 5, ["n"] * 5) == 1.0  # chance agreement total, raters agree
    assert cohen_kappa(["y", "n"], ["n", "y"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 0) == pytest.approx(0.25)  # 2 * (1/2)^3
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)
