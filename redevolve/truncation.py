"""Truncation follow-up: does a larger output cap reveal leaks the baseline cap cut off?

The baseline caps the target at ``target.max_tokens`` (200). For some attacks most replies hit
that cap, so a leak that would have come later is invisible. This follow-up re-runs chosen
attacks at **both** caps, back to back in the same session: for every attack x trial, one call
at the baseline cap, then one at the larger cap, with the same target seed.

Why both caps here rather than pairing with the baseline log: Ollama is deterministic for a
fixed seed *within a session*, but replies did not reproduce across sessions (measured
2026-11-18: an identical replay of a baseline call diverged after ~20 characters, while two
replays in one session were identical). So the baseline log can't be extended exactly.
Pairs made in the same session can be, and :func:`compare` checks it: an uncapped reply
should start with its capped partner (``prefix_match``). Pairs split across a pause/resume may
not match; the count shows how many did.

It is a separate run kind (``truncation``), reported on its own. It never changes the baseline
headline, which stays at the baseline cap. The baseline's own capped rate for the same attacks
is shown alongside as an unpaired reference.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .baseline import WorkItem, eval_key, rejudge, verdict_label
from .seeds import SeedSet
from .stats import Rate

KIND = "truncation"


class FollowupError(ValueError):
    """The follow-up config or its baseline run doesn't fit."""


def pair_key(attack_id: str, trial: int, cap: int) -> str:
    """Log key of one follow-up evaluation at output cap ``cap``."""
    return f"{eval_key(attack_id, trial)}:cap{cap}"


def work_items(
    seeds: SeedSet, attacks: list[str], trials: int, *, capped: int, uncapped: int
) -> list[WorkItem]:
    """Capped then uncapped call for each attack x trial, trial-major, pairs adjacent.

    Raises:
        FollowupError: If an attack id is not in the seed set, or the caps are not increasing.
    """
    by_id = {s.id: s for s in seeds.seeds}
    unknown = [a for a in attacks if a not in by_id]
    if unknown:
        raise FollowupError(f"unknown attack ids in truncation_followup.attacks: {unknown}")
    if uncapped <= capped:
        raise FollowupError(f"max_tokens ({uncapped}) must exceed the baseline cap ({capped})")
    return [
        WorkItem(t, a, by_id[a].family, by_id[a].prompt, pair_key(a, t, cap), cap)
        for t in range(trials)
        for a in attacks
        for cap in (capped, uncapped)
    ]


@dataclass(frozen=True)
class AttackComparison:
    """Capped vs uncapped results for one attack, paired within the follow-up run."""

    attack_id: str
    pairs: int
    capped_full: Rate
    uncapped_full: Rate
    transitions: dict[str, int]  # "none->full" etc. (capped label -> uncapped label)
    still_truncated: int  # uncapped replies that hit the larger cap too
    prefix_match: int  # pairs whose uncapped reply starts with the capped one
    baseline_capped_full: Rate | None = None  # unpaired reference from the baseline log

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "attack_id": self.attack_id,
            "pairs": self.pairs,
            "capped_full": self.capped_full.to_dict(),
            "uncapped_full": self.uncapped_full.to_dict(),
            "transitions": self.transitions,
            "still_truncated": self.still_truncated,
            "prefix_match": self.prefix_match,
            "baseline_capped_full": (
                self.baseline_capped_full.to_dict() if self.baseline_capped_full else None
            ),
        }


def compare(
    followup_records: Iterable[dict[str, Any]],
    *,
    canary: str,
    attacks: list[str],
    capped: int,
    uncapped: int,
    trials: int,
    baseline_records: Iterable[dict[str, Any]] | None = None,
) -> list[AttackComparison]:
    """Pair each attack x trial's capped and uncapped replies and compare verdicts.

    Everything is re-judged with the current judge. Only complete pairs are counted.
    """
    follow = {r["key"]: r for r in rejudge(followup_records, canary)}
    base = {r["key"]: r for r in rejudge(baseline_records, canary)} if baseline_records else {}
    out = []
    for a in attacks:
        trans: Counter[str] = Counter()
        n = cf = uf = trunc = prefix = 0
        for t in range(trials):
            c = follow.get(pair_key(a, t, capped))
            u = follow.get(pair_key(a, t, uncapped))
            if c is None or u is None:
                continue
            n += 1
            trans[f"{verdict_label(c['verdict'])}->{verdict_label(u['verdict'])}"] += 1
            cf += c["verdict"].leaked
            uf += u["verdict"].leaked
            trunc += bool(u.get("truncated"))
            prefix += u["reply"].startswith(c["reply"])
        ref = None
        if base:
            brs = [base[k] for t in range(trials) if (k := eval_key(a, t)) in base]
            ref = Rate(sum(r["verdict"].leaked for r in brs), len(brs))
        out.append(
            AttackComparison(
                attack_id=a,
                pairs=n,
                capped_full=Rate(cf, n),
                uncapped_full=Rate(uf, n),
                transitions=dict(sorted(trans.items())),
                still_truncated=trunc,
                prefix_match=prefix,
                baseline_capped_full=ref,
            )
        )
    return out


def format_comparison(rows: list[AttackComparison], *, capped: int, uncapped: int) -> str:
    """Plain-text table for the console."""
    lines = [
        f"Truncation follow-up: max_tokens {capped} vs {uncapped}, both in this run, same seeds",
        "",
        f"{'attack':<8} {'pairs':>5}  {'full @' + str(capped):>9}  {'full @' + str(uncapped):>10}  "
        f"{'still trunc':>11}  {'prefix match':>12}  {'baseline @' + str(capped):>13}  transitions",
    ]
    for r in rows:
        ref = r.baseline_capped_full
        ref_s = f"{ref.k}/{ref.n}" if ref else "-"
        lines.append(
            f"{r.attack_id:<8} {r.pairs:>5}  {r.capped_full.k:>5}/{r.capped_full.n:<3}  "
            f"{r.uncapped_full.k:>6}/{r.uncapped_full.n:<3}  {r.still_truncated:>11}  "
            f"{r.prefix_match:>8}/{r.pairs:<3}  {ref_s:>13}  {r.transitions}"
        )
    lines += [
        "",
        "prefix match = the uncapped reply starts with its capped partner (exact pairing).",
        f"baseline @{capped} = the baseline run's own rate for these attacks (unpaired reference:",
        "replies don't reproduce across sessions).",
        "This run is reported separately; the baseline headline stays at the baseline cap.",
    ]
    return "\n".join(lines)
