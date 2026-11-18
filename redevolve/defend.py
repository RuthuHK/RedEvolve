"""M8 paired guardrail ablation: one target call per item, all 8 layer combinations derived.

Why one call is enough: the input layers (a: heuristic filter, b: classifier) only decide
*block or pass* and never change the message, and the output layer (c) only rewrites the
reply. So for every attack x trial we record

- layer (a)'s decision on the prompt (pure Python, recomputed at summary time),
- layer (b)'s decision on the prompt (one classifier call, logged),
- the undefended target reply to the prompt (one target call, logged),

and derive what each combination would have shown the user: the fixed block message if an
enabled input layer blocked, otherwise the reply, redacted by (c) if (c) is enabled. Every
configuration is judged on exactly the same replies, so differences between configurations
are paired (no sampling noise from re-generating replies), and the comparison against the
undefended configuration uses an exact McNemar test.

Layers run in the order a -> b -> target -> c. Added latency per configuration counts only the
layers that would actually have run: (b) is skipped when an enabled (a) already blocked, and
(c) when the message was blocked.

Benign prompts go through the same pipeline. A false positive is a benign message that is
blocked, or whose reply is altered by (c).

:func:`run_defend` makes the calls (resumable, like the baseline). :func:`items_from_records`
rebuilds paired items from any eval log (a defend run, or a baseline run offline, in which case
there are no classifier decisions and configurations with (b) are left out).
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from . import judge
from .baseline import RunOutcome, _fmt_duration
from .benign import BenignSet
from .guardrail import (
    LAYERS,
    ClassifierDecision,
    InjectionClassifier,
    InputDecision,
    InputFilter,
    OutputDecision,
    OutputFilter,
)
from .llm import LLMError
from .runs import RunDir
from .seeding import derive_seed, target_seed
from .stats import Rate, cluster_bootstrap_ci, mcnemar_exact
from .target import Target

KIND = "defend"
BENIGN_SET = "benign"

Config = frozenset[str]
# none, a, b, c, a+b, a+c, b+c, a+b+c
CONFIGS: tuple[Config, ...] = tuple(
    frozenset(c) for n in range(len(LAYERS) + 1) for c in combinations(LAYERS, n)
)
NONE: Config = frozenset()


def config_name(config: Config) -> str:
    """``"none"`` or the enabled layers joined in a fixed order, e.g. ``"a+c"``."""
    return "+".join(x for x in LAYERS if x in config) or "none"


def defend_key(set_name: str, item_id: str, trial: int) -> str:
    """Log key of one paired evaluation."""
    return f"{set_name}:{item_id}:{trial}"


# --- paired items and derived outcomes -------------------------------------------------


@dataclass(frozen=True)
class PairedItem:
    """One attack (or benign prompt) x trial with every layer's decision and the raw reply."""

    key: str
    set_name: str
    item_id: str  # attack id, or benign prompt id
    family: str  # attack family, or benign category
    trial: int
    prompt: str
    reply: str
    truncated: bool
    input: InputDecision
    classifier: ClassifierDecision | None  # None when no classifier call was made
    output: OutputDecision

    @property
    def benign(self) -> bool:
        """True for benign prompts (scored for false positives, not leaks)."""
        return self.set_name == BENIGN_SET


@dataclass(frozen=True)
class Outcome:
    """What the user would have seen for one item under one configuration."""

    blocked_by: str | None  # "a", "b" or None
    text: str
    redacted: bool
    verdict: judge.Verdict
    added_latency_s: float

    @property
    def blocked(self) -> bool:
        """True if an input layer blocked the message."""
        return self.blocked_by is not None


def supports(item: PairedItem, config: Config) -> bool:
    """False if ``config`` needs a layer this item has no decision for."""
    return "b" not in config or item.classifier is not None


def derive(item: PairedItem, config: Config, *, canary: str, block_message: str) -> Outcome:
    """The outcome for ``item`` had only the layers in ``config`` been enabled.

    Raises:
        ValueError: If ``config`` includes (b) but the item has no classifier decision.
    """
    if not supports(item, config):
        raise ValueError(f"{item.key}: no classifier decision, can't derive {config_name(config)}")
    latency = 0.0
    blocked_by = None
    if "a" in config:
        latency += item.input.latency_s
        if item.input.blocked:
            blocked_by = "a"
    if blocked_by is None and "b" in config:
        assert item.classifier is not None
        latency += item.classifier.latency_s
        if item.classifier.blocked:
            blocked_by = "b"
    if blocked_by is not None:
        text, redacted = block_message, False
    elif "c" in config:
        latency += item.output.latency_s
        text, redacted = item.output.text, item.output.redacted
    else:
        text, redacted = item.reply, False
    return Outcome(blocked_by, text, redacted, judge.detect(canary, text), latency)


def items_from_records(
    records: Iterable[dict[str, Any]],
    *,
    canary: str,
    input_filter: InputFilter,
    output_filter: OutputFilter,
    default_set: str,
) -> list[PairedItem]:
    """Rebuild paired items from eval records (first record per key wins).

    Layers (a) and (c) are pure, so they are recomputed with the given filters (their
    latencies are measured now). Layer (b) comes from the record's ``classifier`` field, if
    any. Records without a ``set`` field (e.g. a baseline log) get ``default_set``.
    """
    out: dict[str, PairedItem] = {}
    for rec in records:
        key = rec.get("key")
        if rec.get("type") != "eval" or key is None or key in out:
            continue
        cls = rec.get("classifier")
        out[key] = PairedItem(
            key=key,
            set_name=rec.get("set", default_set),
            item_id=rec["attack_id"],
            family=rec["family"],
            trial=rec["trial"],
            prompt=rec["prompt"],
            reply=rec["reply"],
            truncated=bool(rec.get("truncated")),
            input=input_filter.check(rec["prompt"]),
            classifier=ClassifierDecision.from_dict(cls) if cls else None,
            output=output_filter.filter(rec["reply"]),
        )
    return list(out.values())


# --- statistics -------------------------------------------------------------------------


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]


@dataclass(frozen=True)
class AttackConfigStats:
    """One attack set under one configuration."""

    set_name: str
    config: str
    n: int
    full: Rate
    partial: Rate
    blocked: Rate
    blocked_by: dict[str, int]
    redacted: Rate
    full_cluster_ci: tuple[float, float] | None
    fixed: int  # pairs that leak with no guardrail but not under this config
    broken: int  # pairs that leak under this config but not with no guardrail (expected 0)
    mcnemar_p: float
    mean_added_latency_s: float
    p95_added_latency_s: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "set": self.set_name,
            "config": self.config,
            "n": self.n,
            "full": self.full.to_dict(),
            "partial": self.partial.to_dict(),
            "blocked": self.blocked.to_dict(),
            "blocked_by": self.blocked_by,
            "redacted": self.redacted.to_dict(),
            "full_cluster_ci": list(self.full_cluster_ci) if self.full_cluster_ci else None,
            "fixed_vs_none": self.fixed,
            "broken_vs_none": self.broken,
            "mcnemar_p_vs_none": self.mcnemar_p,
            "mean_added_latency_s": self.mean_added_latency_s,
            "p95_added_latency_s": self.p95_added_latency_s,
        }


@dataclass(frozen=True)
class BenignConfigStats:
    """The benign set under one configuration."""

    config: str
    n: int
    false_positive: Rate  # blocked, or reply altered by (c)
    blocked: Rate
    redacted: Rate
    by_category: dict[str, Rate]
    flagged: tuple[str, ...]  # keys of the false positives
    mean_added_latency_s: float
    p95_added_latency_s: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "config": self.config,
            "n": self.n,
            "false_positive": self.false_positive.to_dict(),
            "blocked": self.blocked.to_dict(),
            "redacted": self.redacted.to_dict(),
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "flagged": list(self.flagged),
            "mean_added_latency_s": self.mean_added_latency_s,
            "p95_added_latency_s": self.p95_added_latency_s,
        }


def _attack_stats(
    set_name: str,
    items: list[PairedItem],
    config: Config,
    *,
    canary: str,
    block_message: str,
    run_seed: int,
    n_boot: int,
) -> AttackConfigStats:
    outs = [derive(i, config, canary=canary, block_message=block_message) for i in items]
    base = [derive(i, NONE, canary=canary, block_message=block_message) for i in items]
    fixed = sum(b.verdict.leaked and not o.verdict.leaked for b, o in zip(base, outs, strict=True))
    broken = sum(o.verdict.leaked and not b.verdict.leaked for b, o in zip(base, outs, strict=True))
    per_attack: dict[str, list[bool]] = {}
    for i, o in zip(items, outs, strict=True):
        per_attack.setdefault(i.item_id, []).append(o.verdict.leaked)
    cluster = (
        cluster_bootstrap_ci(
            [sum(v) / len(v) for v in per_attack.values()],
            seed=derive_seed("bootstrap", run_seed, "defend", set_name, config_name(config)),
            n_boot=n_boot,
        )
        if per_attack
        else None
    )
    n = len(items)
    lat = [o.added_latency_s for o in outs]
    return AttackConfigStats(
        set_name=set_name,
        config=config_name(config),
        n=n,
        full=Rate(sum(o.verdict.leaked for o in outs), n),
        partial=Rate(sum(o.verdict.partial for o in outs), n),
        blocked=Rate(sum(o.blocked for o in outs), n),
        blocked_by=dict(Counter(o.blocked_by for o in outs if o.blocked_by)),
        redacted=Rate(sum(o.redacted for o in outs), n),
        full_cluster_ci=cluster,
        fixed=fixed,
        broken=broken,
        mcnemar_p=mcnemar_exact(fixed, broken),
        mean_added_latency_s=sum(lat) / n if n else 0.0,
        p95_added_latency_s=_p95(lat),
    )


def _benign_stats(
    items: list[PairedItem], config: Config, *, canary: str, block_message: str
) -> BenignConfigStats:
    outs = [derive(i, config, canary=canary, block_message=block_message) for i in items]
    fp = [o.blocked or o.redacted for o in outs]
    cats: dict[str, list[bool]] = {}
    for i, f in zip(items, fp, strict=True):
        cats.setdefault(i.family, []).append(f)
    n = len(items)
    lat = [o.added_latency_s for o in outs]
    return BenignConfigStats(
        config=config_name(config),
        n=n,
        false_positive=Rate(sum(fp), n),
        blocked=Rate(sum(o.blocked for o in outs), n),
        redacted=Rate(sum(o.redacted for o in outs), n),
        by_category={c: Rate(sum(v), len(v)) for c, v in cats.items()},
        flagged=tuple(i.key for i, f in zip(items, fp, strict=True) if f),
        mean_added_latency_s=sum(lat) / n if n else 0.0,
        p95_added_latency_s=_p95(lat),
    )


@dataclass(frozen=True)
class Ablation:
    """Per-set, per-configuration results of a paired ablation."""

    attacks: list[AttackConfigStats]
    benign: list[BenignConfigStats]
    skipped_configs: list[str]  # configurations no item had the decisions for

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "judge_version": judge.JUDGE_VERSION,
            "attacks": [a.to_dict() for a in self.attacks],
            "benign": [b.to_dict() for b in self.benign],
            "skipped_configs": self.skipped_configs,
        }


def ablate(
    items: list[PairedItem],
    *,
    canary: str,
    block_message: str,
    run_seed: int,
    n_boot: int = 10_000,
    configs: tuple[Config, ...] = CONFIGS,
) -> Ablation:
    """Derive and score every configuration for every set in ``items``.

    A configuration is evaluated only if every item has the decisions it needs (so a set is
    never scored on a subset of its items); otherwise it is listed in ``skipped_configs``.
    """
    usable = [c for c in configs if all(supports(i, c) for i in items)]
    skipped = [config_name(c) for c in configs if c not in usable]
    sets = list(dict.fromkeys(i.set_name for i in items))
    attacks, benign = [], []
    for s in sets:
        members = [i for i in items if i.set_name == s]
        for c in usable:
            if s == BENIGN_SET:
                benign.append(_benign_stats(members, c, canary=canary, block_message=block_message))
            else:
                attacks.append(
                    _attack_stats(
                        s,
                        members,
                        c,
                        canary=canary,
                        block_message=block_message,
                        run_seed=run_seed,
                        n_boot=n_boot,
                    )
                )
    return Ablation(attacks, benign, skipped)


def benign_input_filter(benign: BenignSet, input_filter: InputFilter) -> list[InputDecision]:
    """Layer (a) on every benign prompt. Needs no model calls, so it works offline."""
    return [input_filter.check(p.prompt) for p in benign.prompts]


# --- formatting ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:6.1%}"


def _ms(s: float) -> str:
    return f"{s * 1000:8.1f}"


def format_ablation(ab: Ablation) -> str:
    """Plain-text tables for the console."""
    lines: list[str] = []
    for s in dict.fromkeys(a.set_name for a in ab.attacks):
        lines += [
            f"attack set {s!r}",
            f"{'config':<7} {'full leak':>9}  {'cluster bootstrap':<17}  {'partial':>7}  "
            f"{'blocked':>7}  {'redacted':>8}  {'fixed':>5}  {'McNemar p':>9}  "
            f"{'+ms mean':>8}  {'+ms p95':>8}",
        ]
        for a in (x for x in ab.attacks if x.set_name == s):
            lo, hi = a.full_cluster_ci or (0.0, 1.0)
            lines.append(
                f"{a.config:<7} {a.full.k:>4}/{a.full.n:<4}  [{_pct(lo)}, {_pct(hi)}]  "
                f"{_pct(a.partial.p)}  {_pct(a.blocked.p)}  {_pct(a.redacted.p):>8}  "
                f"{a.fixed:>5}  {a.mcnemar_p:>9.2g}  {_ms(a.mean_added_latency_s)}  "
                f"{_ms(a.p95_added_latency_s)}"
            )
        lines.append("")
    if ab.benign:
        lines += [
            "benign set",
            f"{'config':<7} {'false pos':>9}  {'95% Wilson':<17}  {'blocked':>7}  {'redacted':>8}  "
            f"{'+ms mean':>8}  {'+ms p95':>8}  flagged",
        ]
        for b in ab.benign:
            lo, hi = b.false_positive.ci
            lines.append(
                f"{b.config:<7} {b.false_positive.k:>4}/{b.false_positive.n:<4}  "
                f"[{_pct(lo)}, {_pct(hi)}]  {_pct(b.blocked.p)}  {_pct(b.redacted.p):>8}  "
                f"{_ms(b.mean_added_latency_s)}  {_ms(b.p95_added_latency_s)}  "
                f"{', '.join(b.flagged) or '-'}"
            )
        lines.append("")
    if ab.skipped_configs:
        lines.append(f"not evaluated (no classifier decisions): {', '.join(ab.skipped_configs)}")
    lines.append(
        "fixed = pairs that leak with no guardrail but not under the config (same replies)."
    )
    return "\n".join(lines)


# --- live runner ------------------------------------------------------------------------


@dataclass(frozen=True)
class DefendWork:
    """One paired evaluation to make."""

    set_name: str
    item_id: str
    family: str
    prompt: str
    trial: int

    @property
    def key(self) -> str:
        """Log key."""
        return defend_key(self.set_name, self.item_id, self.trial)


def benign_work(benign: BenignSet, trials: int) -> list[DefendWork]:
    """Every benign prompt x ``trials``, trial-major."""
    return [
        DefendWork(BENIGN_SET, p.id, p.category, p.prompt, t)
        for t in range(trials)
        for p in benign.prompts
    ]


def run_defend(
    run: RunDir,
    target: Target,
    *,
    input_filter: InputFilter,
    classifier: InjectionClassifier,
    output_filter: OutputFilter,
    work: list[DefendWork],
    run_seed: int,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> RunOutcome:
    """One classifier call and one (undefended) target call per work item, logged with all
    three layers' decisions. Skips keys already logged; status handling as in the baseline.

    Raises:
        LLMError: If the target or the classifier fails; the run is marked ``paused`` first.
        KeyboardInterrupt: Re-raised after marking the run ``paused``.
    """
    per_id = Counter(i for _, i in {(w.set_name, w.item_id) for w in work})
    clash = [i for i, n in per_id.items() if n > 1]
    if clash:  # target seeds are keyed by item id, so ids must be unique across sets
        raise ValueError(f"item ids used in more than one set: {sorted(clash)}")
    done = run.completed_keys("eval")
    todo = [w for w in work if w.key not in done]
    total, already = len(work), len(work) - len(todo)
    if already:
        log(f"resuming: {already}/{total} evaluations already logged, {len(todo)} to go")
    made = 0
    started = clock()
    try:
        for w in todo:
            if limit is not None and made >= limit:
                run.set_status("paused", stopped_reason=f"--limit {limit}")
                return RunOutcome(made, already + made, total, complete=False)
            a = input_filter.check(w.prompt)
            b = classifier.check(
                w.prompt, seed=derive_seed("classifier", run_seed, w.item_id, w.trial)
            )
            ev = target.evaluate(
                w.item_id,
                w.family,
                w.prompt,
                trial=w.trial,
                seed=target_seed(run_seed, w.item_id, w.trial),
            )
            c = output_filter.filter(ev.result.text)
            run.append(
                {
                    **ev.to_record(),
                    "key": w.key,
                    "set": w.set_name,
                    "input_filter": a.to_dict(),
                    "classifier": b.to_dict(),
                    "output_filter": c.to_dict(),
                }
            )
            made += 1
            avg = (clock() - started) / made
            remaining = len(todo) - made if limit is None else min(len(todo), limit) - made
            flags = "".join(
                x for x, on in (("a", a.blocked), ("b", b.blocked), ("c", c.redacted)) if on
            )
            log(
                f"[{already + made:>4}/{total}] {w.set_name:<8} {w.item_id:<10} t{w.trial:<2} "
                f"{'LEAK' if ev.verdict.leaked else '-':<4} layers {flags or '-':<3} "
                f"target {ev.result.latency_s:5.1f}s clf {b.latency_s:4.1f}s  "
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
