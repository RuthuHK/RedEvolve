"""Reuse of input-side guardrail verdicts across target models (v1.1).

Layers (b) (the local classifier) and (p) (Prompt Guard) only see the attack prompt, never the
target's reply, so their decision on a prompt doesn't depend on which target model is behind
them. A defend run against a second target can therefore take those decisions from earlier
defend runs instead of calling the classifiers again. (Layer (a) is pure and is always
recomputed.) A decision is reused only on an exact match of

    layer, classifier model, guardrail fingerprint sha256, sha256(prompt), classifier seed

where the seed is the one the source run used (``derive_seed("classifier", run_seed, id,
trial)`` for (b); Prompt Guard takes none). Reused decisions keep the latency measured in the
source run, and every reused record names its source run (``reused``).

A source run is accepted only after a **determinism check** (:func:`check_determinism`,
``python -m redevolve verdicts check RUN_ID``): a fixed sample of its items is re-classified
live and must reproduce every logged decision (same (b) label; same (p) block decision and a
score within :data:`SCORE_TOLERANCE`). The result is written to the source run's
``verdict_check.json``; :meth:`VerdictStore.from_runs` refuses a run without a passing one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .guardrail import ClassifierDecision, InjectionClassifier, PromptGuardClassifier
from .runs import RunDir
from .seeding import derive_seed

CHECK_FILE = "verdict_check.json"
CHECK_N = 20  # items re-classified per source run (half blocked by (b), half passed)
SCORE_TOLERANCE = 0.01  # max |score difference| for a Prompt Guard decision to count as equal
LAYER_FIELDS = {"b": "classifier", "p": "comparison_classifier"}  # layer -> log record field


class VerdictError(RuntimeError):
    """A source run can't be used for verdict reuse."""


def prompt_sha256(prompt: str) -> str:
    """SHA-256 of a prompt's UTF-8 text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def classifier_seed(run_seed: int, item_id: str, trial: int) -> int:
    """The (b) classifier's seed for one item (as used by ``defend.run_defend``)."""
    return derive_seed("classifier", run_seed, item_id, trial)


def layer_models(cfg: dict[str, Any]) -> dict[str, str]:
    """``provider/model`` of each input-side classifier layer configured in ``cfg``."""
    out = {"b": f"{cfg['classifier']['provider']}/{cfg['classifier']['model']}"}
    if cfg.get("comparison_classifier"):
        c = cfg["comparison_classifier"]
        out["p"] = f"{c['provider']}/{c['model']}"
    return out


def _key(layer: str, model: str, guardrail_sha: str, prompt: str, seed: int | None) -> str:
    return "|".join((layer, model, guardrail_sha, prompt_sha256(prompt), str(seed)))


@dataclass
class VerdictStore:
    """Logged (b)/(p) decisions from accepted source runs, looked up by exact key."""

    guardrail_sha: str
    models: dict[str, str]
    sources: list[str] = field(default_factory=list)
    conflicts: int = 0  # same key, different decision in two records (first one kept)
    _d: dict[str, tuple[dict[str, Any], str]] = field(default_factory=dict)

    def add_records(self, records: Iterable[dict[str, Any]], *, run_seed: int, run_id: str) -> int:
        """Index the (b)/(p) decisions in eval ``records``; returns how many were added."""
        added = 0
        for rec in records:
            if rec.get("type") != "eval":
                continue
            for layer, fld in LAYER_FIELDS.items():
                d = rec.get(fld)
                if not d or d.get("model") != self.models.get(layer):
                    continue
                seed = (
                    classifier_seed(run_seed, rec["attack_id"], rec["trial"])
                    if layer == "b"
                    else None
                )
                k = _key(layer, d["model"], self.guardrail_sha, rec["prompt"], seed)
                if k in self._d:
                    prev = self._d[k][0]
                    self.conflicts += (prev["label"], prev["blocked"]) != (d["label"], d["blocked"])
                    continue
                self._d[k] = (d, run_id)
                added += 1
        return added

    def get(
        self, layer: str, prompt: str, *, seed: int | None
    ) -> tuple[ClassifierDecision, str] | None:
        """The logged decision for this layer/prompt/seed and its source run, or None."""
        model = self.models.get(layer)
        if model is None:
            return None
        hit = self._d.get(_key(layer, model, self.guardrail_sha, prompt, seed))
        return (ClassifierDecision.from_dict(hit[0]), hit[1]) if hit else None

    def __len__(self) -> int:
        return len(self._d)

    @classmethod
    def from_runs(
        cls, runs: list[RunDir], *, guardrail_sha: str, models: dict[str, str]
    ) -> VerdictStore:
        """A store over complete defend runs with this guardrail and a passing check.

        Raises:
            VerdictError: If any run is not a complete defend run, used another guardrail,
                or has no passing ``verdict_check.json`` for this guardrail.
        """
        store = cls(guardrail_sha, dict(models))
        for run in runs:
            problems = []
            if run.meta.get("kind") != "defend" or run.meta.get("status") != "complete":
                problems.append("not a complete defend run")
            if (run.meta.get("guardrail") or {}).get("sha256") != guardrail_sha:
                problems.append("guardrail fingerprint differs")
            chk = load_check(run)
            if chk is None or not chk.get("passed") or chk.get("guardrail_sha256") != guardrail_sha:
                problems.append(
                    f"no passing determinism check (python -m redevolve verdicts check "
                    f"{run.run_id})"
                )
            if problems:
                raise VerdictError(f"{run.run_id}: {'; '.join(problems)}")
            store.add_records(run.records("eval"), run_seed=run.meta["seed"], run_id=run.run_id)
            store.sources.append(run.run_id)
        return store


def load_check(run: RunDir) -> dict[str, Any] | None:
    """The run's ``verdict_check.json``, or None."""
    p = run.path / CHECK_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def check_sample(records: list[dict[str, Any]], n: int = CHECK_N) -> list[dict[str, Any]]:
    """Pre-registered sample: eval records ordered by sha256 of their key; the first ``n/2``
    blocked by (b) and the first ``n - n/2`` passed by it (topped up from the other group)."""
    evs = sorted(
        (r for r in records if r.get("type") == "eval" and r.get("classifier")),
        key=lambda r: hashlib.sha256(r["key"].encode()).hexdigest(),
    )
    blocked = [r for r in evs if r["classifier"]["blocked"]]
    passed = [r for r in evs if not r["classifier"]["blocked"]]
    half = n // 2
    take_b = blocked[: max(half, n - len(passed))]
    take_p = passed[: n - len(take_b)]
    return sorted(take_b + take_p, key=lambda r: hashlib.sha256(r["key"].encode()).hexdigest())


def check_determinism(
    run: RunDir,
    classifier: InjectionClassifier,
    comparison: PromptGuardClassifier | None,
    *,
    guardrail_sha: str,
    n: int = CHECK_N,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Re-classify a fixed sample of ``run``'s items live and compare with the logged
    decisions; writes and returns ``verdict_check.json`` (``passed`` = every decision matched).

    Raises:
        LLMError: If a classifier call fails (nothing is written).
    """
    sample = check_sample(list(run.records("eval")), n)
    rows, ok_all = [], True
    for rec in sample:
        seed = classifier_seed(run.meta["seed"], rec["attack_id"], rec["trial"])
        b_old = rec["classifier"]
        b_new = classifier.check(rec["prompt"], seed=seed)
        row: dict[str, Any] = {
            "key": rec["key"],
            "b_logged": b_old["label"],
            "b_now": b_new.label,
            "b_match": b_new.label == b_old["label"] and b_new.blocked == b_old["blocked"],
        }
        p_old = rec.get("comparison_classifier")
        if comparison is not None and p_old:
            p_new = comparison.check(rec["prompt"])
            diff = abs((p_new.score or 0.0) - (p_old.get("score") or 0.0))
            row |= {
                "p_logged": p_old.get("score"),
                "p_now": p_new.score,
                "p_match": p_new.blocked == p_old["blocked"] and diff <= SCORE_TOLERANCE,
            }
        ok = row["b_match"] and row.get("p_match", True)
        ok_all &= ok
        rows.append(row)
        log(f"  {rec['key']:<28} b {row['b_logged']}->{row['b_now']}"
            + (f"  p {row['p_logged']:.4f}->{row['p_now']:.4f}" if "p_now" in row else "")
            + ("" if ok else "  MISMATCH"))  # fmt: skip
    result = {
        "checked_at": datetime.now(UTC).isoformat(),
        "run_id": run.run_id,
        "guardrail_sha256": guardrail_sha,
        "n": len(rows),
        "score_tolerance": SCORE_TOLERANCE,
        "b_matches": sum(r["b_match"] for r in rows),
        "p_checked": sum("p_match" in r for r in rows),
        "p_matches": sum(r.get("p_match", False) for r in rows),
        "passed": bool(rows) and ok_all,
        "items": rows,
    }
    (run.path / CHECK_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
