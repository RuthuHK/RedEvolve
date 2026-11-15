"""Small statistics helpers: Wilson intervals, cluster bootstrap, rate summaries, Cohen's kappa."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default).

    Better than the normal approximation at small n and at rates near 0 or 1, which is
    exactly where per-seed attack success rates live. Returns (0, 1) for n == 0.
    """
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Pin the endpoints exactly at k = 0 and k = n (rounding otherwise leaves ~1e-17).
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


@dataclass(frozen=True)
class Rate:
    """k successes out of n, with a Wilson 95% interval."""

    k: int
    n: int

    @property
    def p(self) -> float:
        """The observed rate k/n (0 when n is 0)."""
        return self.k / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        """Wilson 95% interval."""
        return wilson(self.k, self.n)

    def to_dict(self) -> dict[str, float | int]:
        """JSON-friendly form: k, n, rate, ci_low, ci_high."""
        lo, hi = self.ci
        return {"k": self.k, "n": self.n, "rate": self.p, "ci_low": lo, "ci_high": hi}

    def fmt(self) -> str:
        """Human-readable form, e.g. ``30.0% [10.8%, 60.3%] (3/10)``."""
        lo, hi = self.ci
        return f"{self.p:.1%} [{lo:.1%}, {hi:.1%}] ({self.k}/{self.n})"


def cluster_bootstrap_ci(
    rates: Sequence[float],
    *,
    seed: int,
    n_boot: int = 10_000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-cluster rates.

    Used for family and overall ASR: trials of the same attack are not independent of each
    other (one prompt can leak 10/10 while another leaks 0/10), so a pooled Wilson interval
    over all evaluations is too narrow. Resampling whole attacks (clusters) accounts for the
    between-attack spread. With few clusters (6 per family) the interval is itself rough.

    Args:
        rates: One rate per cluster (e.g. per attack, over its K trials).
        seed: RNG seed, so the interval is reproducible.
        n_boot: Number of bootstrap resamples.
        level: Coverage, e.g. 0.95.

    Raises:
        ValueError: If ``rates`` is empty or ``level`` is not in (0, 1).
    """
    if not rates:
        raise ValueError("need at least one cluster")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    rng = random.Random(seed)
    m = len(rates)
    means = sorted(sum(rng.choices(rates, k=m)) / m for _ in range(n_boot))
    alpha = (1 - level) / 2
    lo = means[math.floor(alpha * (n_boot - 1))]
    hi = means[math.ceil((1 - alpha) * (n_boot - 1))]
    return lo, hi


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two raters' labels over the same items.

    Returns 1.0 when chance agreement is already total and the raters agree.
    """
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be non-empty and of equal length")
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)
