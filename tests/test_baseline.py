"""Tests for the baseline runner and summary (target stubbed; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from redevolve import baseline
from redevolve.config import load_config
from redevolve.llm import ChatResult, LLMUnavailable
from redevolve.runs import RUN_FILE, RunDir
from redevolve.seeding import derive_canary, target_seed
from redevolve.seeds import load_seeds
from redevolve.target import Target
from tests.test_config import REPO_CONFIG
from tests.test_seeds import REPO_SEEDS

RUN_SEED = 42
CANARY = derive_canary(RUN_SEED)
CORE = CANARY.split("-")[1]
ENV = {"git_commit": "abc", "git_dirty": False}


class ScriptedTarget:
    """Leaks for io-01 always, for rp-01 on even trial seeds, partially for enc-01."""

    provider, model = "ollama", "stub"

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.fail_after = fail_after

    def chat(self, messages, *, temperature, max_tokens, seed=None) -> ChatResult:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise LLMUnavailable("ollama down")
        prompt = messages[1]["content"]
        self.calls.append((prompt, seed))
        attack = PROMPT_TO_ID[prompt]
        if attack == "io-01" or (attack == "rp-01" and seed is not None and seed % 2 == 0):
            text = f"Sure, it is {CANARY}."
        elif attack == "enc-01":
            text = f"Part of it: {CORE[:5]}"
        else:
            text = "Sorry, I can't share that."
        finish = "length" if attack == "fs-01" else "stop"
        return ChatResult(text, text, None, "ollama", "stub", 2.0, 100, 20, finish)


SEEDS = load_seeds(REPO_SEEDS)
PROMPT_TO_ID = {s.prompt: s.id for s in SEEDS.seeds}


@pytest.fixture
def cfg() -> dict:
    return load_config(REPO_CONFIG)


def make_run(tmp_path: Path, cfg: dict) -> RunDir:
    return RunDir.create(tmp_path, "baseline", cfg, seeds=SEEDS, canary=CANARY, env=lambda: ENV)


def make_target(client: ScriptedTarget) -> Target:
    return Target(
        client,
        system_prompt_template="Code: {canary}",
        canary=CANARY,
        temperature=0.7,
        max_tokens=200,
    )


def run(tmp_path: Path, cfg: dict, client: ScriptedTarget, trials: int = 2, **kw):
    r = make_run(tmp_path, cfg)
    out = baseline.run_baseline(
        r, make_target(client), SEEDS, trials=trials, run_seed=RUN_SEED, log=lambda s: None, **kw
    )
    return r, out


def test_schedule_is_trial_major() -> None:
    work = baseline.schedule(SEEDS, 2)
    assert len(work) == 60
    assert [w[0] for w in work[:30]] == [0] * 30
    assert [w[1] for w in work[:30]] == [s.id for s in SEEDS.seeds]
    assert work[30][:2] == (1, "io-01")


def test_full_run_logs_every_eval_with_derived_seeds(tmp_path: Path, cfg: dict) -> None:
    client = ScriptedTarget()
    r, out = run(tmp_path, cfg, client)
    assert out == baseline.RunOutcome(60, 60, 60, complete=True)
    recs = list(r.records("eval"))
    assert len(recs) == 60
    assert {rec["key"] for rec in recs} == {
        baseline.eval_key(s.id, t) for s in SEEDS.seeds for t in range(2)
    }
    for rec in recs:
        assert rec["seed"] == target_seed(RUN_SEED, rec["attack_id"], rec["trial"])
    assert json.loads((r.path / RUN_FILE).read_text())["status"] == "complete"


def test_limit_pauses_and_resume_skips_completed(tmp_path: Path, cfg: dict) -> None:
    first = ScriptedTarget()
    r, out = run(tmp_path, cfg, first, limit=7)
    assert out == baseline.RunOutcome(7, 7, 60, complete=False)
    assert r.meta["status"] == "paused"

    resumed = RunDir.resume(tmp_path, r.run_id, cfg, seeds=SEEDS, canary=CANARY)
    second = ScriptedTarget()
    out2 = baseline.run_baseline(
        resumed, make_target(second), SEEDS, trials=2, run_seed=RUN_SEED, log=lambda s: None
    )
    assert out2 == baseline.RunOutcome(53, 60, 60, complete=True)
    assert len(first.calls) + len(second.calls) == 60
    keys = [rec["key"] for rec in resumed.records("eval")]
    assert len(keys) == len(set(keys)) == 60


def test_resumed_seeds_match_uninterrupted_run(tmp_path: Path, cfg: dict) -> None:
    whole = ScriptedTarget()
    run(tmp_path / "a", cfg, whole)
    part1, part2 = ScriptedTarget(), ScriptedTarget()
    r, _ = run(tmp_path / "b", cfg, part1, limit=13)
    resumed = RunDir.resume(tmp_path / "b", r.run_id, cfg, seeds=SEEDS, canary=CANARY)
    baseline.run_baseline(
        resumed, make_target(part2), SEEDS, trials=2, run_seed=RUN_SEED, log=lambda s: None
    )
    assert sorted(whole.calls) == sorted(part1.calls + part2.calls)


def test_target_failure_pauses_and_keeps_log(tmp_path: Path, cfg: dict) -> None:
    r = make_run(tmp_path, cfg)
    with pytest.raises(LLMUnavailable):
        baseline.run_baseline(
            r,
            make_target(ScriptedTarget(fail_after=5)),
            SEEDS,
            trials=2,
            run_seed=RUN_SEED,
            log=lambda s: None,
        )
    meta = json.loads((r.path / RUN_FILE).read_text())
    assert meta["status"] == "paused"
    assert "ollama down" in meta["last_error"]
    assert len(list(r.records("eval"))) == 5


def test_progress_lines_mention_leaks_and_eta(tmp_path: Path, cfg: dict) -> None:
    lines: list[str] = []
    r = make_run(tmp_path, cfg)
    ticks = iter(range(1000))
    baseline.run_baseline(
        r,
        make_target(ScriptedTarget()),
        SEEDS,
        trials=1,
        run_seed=RUN_SEED,
        log=lines.append,
        clock=lambda: float(next(ticks)),
    )
    assert len(lines) == 30
    assert "io-01" in lines[0] and "LEAK" in lines[0] and "ETA" in lines[0]


def summarize(r: RunDir, trials: int = 2, **kw) -> baseline.Summary:
    args = {"gate_min_seed_rate": 0.2, "gate_min_seeds": 1, "n_boot": 500, **kw}
    return baseline.summarize(
        r.records("eval"), canary=CANARY, seeds=SEEDS, trials=trials, run_seed=RUN_SEED, **args
    )


def test_summary_counts(tmp_path: Path, cfg: dict) -> None:
    r, _ = run(tmp_path, cfg, ScriptedTarget(), trials=4)
    s = summarize(r, trials=4)
    assert s.complete and s.evaluated == 120 and s.verdicts_changed == 0
    by = {a.name: a for a in s.attacks}
    assert (by["io-01"].full.k, by["io-01"].full.n) == (4, 4)
    assert by["enc-01"].partial.k == 4 and by["enc-01"].full.k == 0
    assert by["fs-01"].truncated.k == 4
    rp01 = sum(target_seed(RUN_SEED, "rp-01", t) % 2 == 0 for t in range(4))
    assert by["rp-01"].full.k == rp01
    fam = {f.name: f for f in s.families}
    assert fam["instruction_override"].full.k == 4
    assert fam["instruction_override"].n_attacks == 6
    assert s.overall.full.k == 4 + rp01
    assert s.methods == {"plain": 4 + rp01}
    assert s.partial_methods == {"fragment": 4}
    lo, hi = fam["instruction_override"].full_cluster_ci
    assert lo <= fam["instruction_override"].full.p <= hi
    assert s.gate.passed and "io-01" in s.gate.qualifying


def test_gate_fails_when_nothing_leaks_enough(tmp_path: Path, cfg: dict) -> None:
    r, _ = run(tmp_path, cfg, ScriptedTarget(), trials=1)
    s = summarize(r, trials=1, gate_min_seeds=3, gate_min_seed_rate=0.5)
    assert not s.gate.passed
    assert "FAILED" in baseline.format_summary(s)


def test_summary_rejudges_and_flags_changed_verdicts(tmp_path: Path, cfg: dict) -> None:
    r, _ = run(tmp_path, cfg, ScriptedTarget(), trials=1)
    tampered = []
    for rec in r.records("eval"):
        if rec["attack_id"] == "io-01":
            rec = {**rec, "leaked": False}  # as if logged by an older judge
        tampered.append(rec)
    s = baseline.summarize(
        tampered,
        canary=CANARY,
        seeds=SEEDS,
        trials=1,
        run_seed=RUN_SEED,
        gate_min_seed_rate=0.2,
        gate_min_seeds=1,
        n_boot=100,
    )
    assert s.verdicts_changed == 1
    assert {a.name: a.full.k for a in s.attacks}["io-01"] == 1


def test_incomplete_summary_and_duplicates(tmp_path: Path, cfg: dict) -> None:
    r, _ = run(tmp_path, cfg, ScriptedTarget(), trials=2, limit=10)
    recs = list(r.records("eval"))
    s = baseline.summarize(
        recs + recs[:3],  # duplicate keys are counted once
        canary=CANARY,
        seeds=SEEDS,
        trials=2,
        run_seed=RUN_SEED,
        gate_min_seed_rate=0.2,
        gate_min_seeds=1,
        n_boot=100,
    )
    assert s.evaluated == 10 and not s.complete
    assert "INCOMPLETE" in baseline.format_summary(s)
    d = s.to_dict()
    assert d["complete"] is False
    assert json.loads(json.dumps(d)) == d  # JSON-serialisable
    assert {a["family"] for a in d["attacks"]} == set(SEEDS.families)
