"""M5 baseline: every seed attack x K trials against the undefended target, then a summary.

The loop is trial-major (all 30 seeds at trial 0, then all at trial 1, ...), so a run that is
paused part-way still has a balanced sample across seeds. Every evaluation is logged with a
``key`` (``eval:<attack_id>:<trial>``), and a resumed run skips keys already in the log.
Target sampling seeds come from :func:`redevolve.seeding.target_seed`, so skipping work on
resume never shifts the randomness of the remaining calls.

The summary re-judges every stored reply with the *current* judge, so results are never
mixed across judge versions, and reports how many verdicts differ from the logged ones.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from . import judge
from .llm import LLMError
from .runs import RunDir
from .seeding import derive_seed, target_seed
from .seeds import SeedSet
from .stats import Rate, cluster_bootstrap_ci
from .target import Target

KIND = "baseline"


def eval_key(attack_id: str, trial: int) -> str:
    """Log key of one baseline evaluation."""
    return f"eval:{attack_id}:{trial}"


@dataclass(frozen=True)
class WorkItem:
    """One target evaluation to make: which attack, which trial, its log key and output cap."""

    trial: int
    attack_id: str
    family: str
    prompt: str
    key: str
    max_tokens: int | None = None  # None = the target's configured cap


def schedule(seeds: SeedSet, trials: int) -> list[tuple[int, str, str, str]]:
    """All (trial, attack_id, family, prompt) work items, trial-major."""
    return [(t, s.id, s.family, s.prompt) for t in range(trials) for s in seeds.seeds]


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


@dataclass(frozen=True)
class RunOutcome:
    """What a call to :func:`run_baseline` did."""

    done_now: int  # evaluations made in this session
    total_done: int  # evaluations in the log after this session
    total: int  # evaluations the full run needs
    complete: bool


def run_baseline(
    run: RunDir,
    target: Target,
    seeds: SeedSet,
    *,
    trials: int,
    run_seed: int,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> RunOutcome:
    """Evaluate every seed ``trials`` times, skipping work already in the run's log.

    Args:
        run: The (new or resumed) run directory.
        target: The target, with this run's canary.
        seeds: The frozen seed set.
        trials: K, evaluations per seed.
        run_seed: The config seed; target sampling seeds derive from it.
        limit: Stop (status ``paused``) after this many new evaluations; for trying a few
            calls before committing to the full run.
        log: Progress output.
        clock: Monotonic clock (injectable for tests).

    Returns:
        A :class:`RunOutcome`. The run status is ``complete`` or ``paused`` afterwards.

    Raises:
        LLMError: If the target fails (after its own retries). The run is marked ``paused``
            with the error first, and everything logged so far is kept.
        KeyboardInterrupt: Re-raised after marking the run ``paused``.
    """
    items = [WorkItem(t, a, f, p, eval_key(a, t)) for t, a, f, p in schedule(seeds, trials)]
    return run_evals(run, target, items, run_seed=run_seed, limit=limit, log=log, clock=clock)


def run_evals(
    run: RunDir,
    target: Target,
    work: list[WorkItem],
    *,
    run_seed: int,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> RunOutcome:
    """Evaluate work items in order, skipping keys already in the log.

    Shared by the baseline and the truncation follow-up, so both log identical records with
    identical target seeds (``target_seed(run_seed, attack_id, trial)``; the cap doesn't enter
    the seed). Status handling and exceptions are as in :func:`run_baseline`.
    """
    done = run.completed_keys("eval")
    todo = [w for w in work if w.key not in done]
    total, already = len(work), len(work) - len(todo)
    if already:
        log(f"resuming: {already}/{total} evaluations already logged, {len(todo)} to go")

    made, leaks = 0, 0
    started = clock()
    try:
        for item in todo:
            trial, attack_id = item.trial, item.attack_id
            if limit is not None and made >= limit:
                run.set_status("paused", stopped_reason=f"--limit {limit}")
                return RunOutcome(made, already + made, total, complete=False)
            seed = target_seed(run_seed, attack_id, trial)
            ev = target.evaluate(
                attack_id,
                item.family,
                item.prompt,
                trial=trial,
                seed=seed,
                max_tokens=item.max_tokens,
            )
            record = {**ev.to_record(), "key": item.key}
            if item.max_tokens is not None:
                record["max_tokens"] = item.max_tokens
            run.append(record)
            made += 1
            leaks += ev.verdict.leaked
            avg = (clock() - started) / made
            remaining = len(todo) - made
            if limit is not None:
                remaining = min(remaining, limit - made)
            flag = "LEAK" if ev.verdict.leaked else ("part" if ev.verdict.partial else "-")
            log(
                f"[{already + made:>4}/{total}] {attack_id:<7} t{trial:<2} "
                f"{'' if item.max_tokens is None else f'cap{item.max_tokens} '}{flag:<4} "
                f"{ev.result.latency_s:5.1f}s  leaks this session {leaks}/{made}  "
                f"ETA {_fmt_duration(avg * remaining)}"
            )
    except LLMError as e:
        run.set_status("paused", last_error=f"{type(e).__name__}: {e}")
        raise
    except KeyboardInterrupt:
        run.set_status("paused", stopped_reason="interrupted")
        raise
    run.set_status("complete")
    return RunOutcome(made, total, total, complete=True)


# --- summary ---------------------------------------------------------------------------


def verdict_label(v: judge.Verdict) -> str:
    """``full``, ``partial`` or ``none``: the label scale shared with human labelling."""
    return "full" if v.leaked else ("partial" if v.partial else "none")


def rejudge(records: Iterable[dict[str, Any]], canary: str) -> list[dict[str, Any]]:
    """Deduplicated eval records (first per key), each with ``verdict`` from the current judge.

    The record's logged verdict is kept as-is; ``verdict_changed`` says whether the current
    judge disagrees with it (expected only after a judge version bump).
    """
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = rec.get("key")
        if rec.get("type") != "eval" or key is None or key in out:
            continue
        v = judge.detect(canary, rec["reply"])
        changed = (v.leaked, v.partial) != (rec.get("leaked"), rec.get("partial"))
        out[key] = {**rec, "verdict": v, "verdict_changed": changed}
    return list(out.values())


@dataclass(frozen=True)
class GroupStats:
    """Leak statistics for one attack, one family, or everything."""

    name: str
    full: Rate
    partial: Rate
    truncated: Rate
    n_attacks: int
    mean_latency_s: float
    full_cluster_ci: tuple[float, float] | None = None  # bootstrap over attacks (groups only)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        d: dict[str, Any] = {
            "name": self.name,
            "full": self.full.to_dict(),
            "partial": self.partial.to_dict(),
            "truncated": self.truncated.to_dict(),
            "n_attacks": self.n_attacks,
            "mean_latency_s": round(self.mean_latency_s, 3),
        }
        if self.full_cluster_ci is not None:
            d["full_cluster_ci"] = list(self.full_cluster_ci)
        return d


def _stats(
    name: str, recs: list[dict[str, Any]], *, cluster_seed: int | None, n_boot: int
) -> GroupStats:
    n = len(recs)
    attacks = sorted({r["attack_id"] for r in recs})
    cluster_ci = None
    if cluster_seed is not None and attacks:
        per_attack = []
        for a in attacks:
            ar = [r for r in recs if r["attack_id"] == a]
            per_attack.append(sum(r["verdict"].leaked for r in ar) / len(ar))
        cluster_ci = cluster_bootstrap_ci(per_attack, seed=cluster_seed, n_boot=n_boot)
    return GroupStats(
        name=name,
        full=Rate(sum(r["verdict"].leaked for r in recs), n),
        partial=Rate(sum(r["verdict"].partial for r in recs), n),
        truncated=Rate(sum(bool(r.get("truncated")) for r in recs), n),
        n_attacks=len(attacks),
        mean_latency_s=sum(r.get("latency_s", 0.0) for r in recs) / n if n else 0.0,
        full_cluster_ci=cluster_ci,
    )


@dataclass(frozen=True)
class Gate:
    """The pre-registered low-baseline gate (see CLAUDE.md)."""

    passed: bool
    min_seed_rate: float
    min_seeds: int
    qualifying: tuple[str, ...]  # attack ids at or above the rate

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "passed": self.passed,
            "min_seed_rate": self.min_seed_rate,
            "min_seeds": self.min_seeds,
            "qualifying": list(self.qualifying),
        }


@dataclass(frozen=True)
class Summary:
    """Baseline results, all computed from the log by re-judging the stored replies."""

    judge_version: int
    trials: int
    expected: int
    evaluated: int
    verdicts_changed: int
    attacks: list[GroupStats]
    families: list[GroupStats]
    overall: GroupStats
    methods: dict[str, int]
    partial_methods: dict[str, int]
    gate: Gate
    families_of: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """True if every scheduled evaluation is in the log."""
        return self.evaluated == self.expected

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form (written to ``summary.json``)."""
        return {
            "judge_version": self.judge_version,
            "trials": self.trials,
            "expected": self.expected,
            "evaluated": self.evaluated,
            "complete": self.complete,
            "verdicts_changed_vs_log": self.verdicts_changed,
            "overall": self.overall.to_dict(),
            "families": [f.to_dict() for f in self.families],
            "attacks": [{**a.to_dict(), "family": self.families_of[a.name]} for a in self.attacks],
            "full_leak_methods": self.methods,
            "partial_methods": self.partial_methods,
            "gate": self.gate.to_dict(),
        }


def summarize(
    records: Iterable[dict[str, Any]],
    *,
    canary: str,
    seeds: SeedSet,
    trials: int,
    run_seed: int,
    gate_min_seed_rate: float,
    gate_min_seeds: int,
    n_boot: int = 10_000,
) -> Summary:
    """Per-attack, per-family and overall leak rates, plus the low-baseline gate.

    Per-attack intervals are Wilson. Family and overall rates come with both a pooled Wilson
    interval (treats every evaluation as independent, so it is too narrow) and a cluster
    bootstrap interval over attacks (the one to quote).
    """
    recs = rejudge(records, canary)
    by_attack = {s.id: [r for r in recs if r["attack_id"] == s.id] for s in seeds.seeds}
    attacks = [_stats(a, rs, cluster_seed=None, n_boot=n_boot) for a, rs in by_attack.items()]
    families = [
        _stats(
            fam,
            [r for s in members for r in by_attack[s.id]],
            cluster_seed=derive_seed("bootstrap", run_seed, fam),
            n_boot=n_boot,
        )
        for fam, members in seeds.by_family().items()
    ]
    overall = _stats(
        "overall", recs, cluster_seed=derive_seed("bootstrap", run_seed, "overall"), n_boot=n_boot
    )
    qualifying = tuple(
        a.name for a in attacks if a.full.n and a.full.p >= gate_min_seed_rate - 1e-12
    )
    methods = Counter(r["verdict"].method for r in recs if r["verdict"].leaked)
    partials = Counter(r["verdict"].partial_method for r in recs if r["verdict"].partial)
    return Summary(
        judge_version=judge.JUDGE_VERSION,
        trials=trials,
        expected=trials * len(seeds.seeds),
        evaluated=len(recs),
        verdicts_changed=sum(r["verdict_changed"] for r in recs),
        attacks=attacks,
        families=families,
        overall=overall,
        methods=dict(methods.most_common()),
        partial_methods=dict(partials.most_common()),
        gate=Gate(
            passed=len(qualifying) >= gate_min_seeds,
            min_seed_rate=gate_min_seed_rate,
            min_seeds=gate_min_seeds,
            qualifying=qualifying,
        ),
        families_of={s.id: s.family for s in seeds.seeds},
    )


def _pct(x: float) -> str:
    return f"{x:6.1%}"


def format_summary(s: Summary) -> str:
    """Plain-text tables for the console."""
    lines = [
        f"Baseline summary (judge v{s.judge_version}, K={s.trials}, "
        f"{s.evaluated}/{s.expected} evaluations{'' if s.complete else ', INCOMPLETE'})",
    ]
    if s.verdicts_changed:
        lines.append(f"note: {s.verdicts_changed} verdicts differ from the logged ones (re-judged)")
    lines += [
        "",
        f"{'attack':<8} {'family':<21} {'full leak':>7}  {'95% Wilson':<17} {'partial':>7}"
        f"  {'trunc':>6}",
    ]
    for a in s.attacks:
        lo, hi = a.full.ci
        lines.append(
            f"{a.name:<8} {s.families_of[a.name]:<21} {a.full.k:>2}/{a.full.n:<4} "
            f"[{_pct(lo)}, {_pct(hi)}] {a.partial.k:>2}/{a.partial.n:<4} "
            f"{_pct(a.truncated.p)}"
        )
    lines += [
        "",
        f"{'group':<21} {'ASR':>7}  {'cluster bootstrap':<17}  {'pooled Wilson':<17} "
        f"{'partial':>7}  {'trunc':>6}",
    ]
    for g in [*s.families, s.overall]:
        lo, hi = g.full.ci
        clo, chi = g.full_cluster_ci or (0.0, 1.0)
        lines.append(
            f"{g.name:<21} {_pct(g.full.p)}  [{_pct(clo)}, {_pct(chi)}]  "
            f"[{_pct(lo)}, {_pct(hi)}] {_pct(g.partial.p)}  {_pct(g.truncated.p)}"
        )
    lines += ["", f"full-leak methods: {s.methods or '{}'}"]
    lines.append(f"partial methods:   {s.partial_methods or '{}'}")
    g = s.gate
    verdict = "PASSED" if g.passed else "FAILED: stop and review before running the GA"
    lines.append(
        f"low-baseline gate (>= {g.min_seeds} seed(s) with full-leak rate >= "
        f"{g.min_seed_rate:.0%}): {verdict}; qualifying: {list(g.qualifying) or 'none'}"
    )
    return "\n".join(lines)
