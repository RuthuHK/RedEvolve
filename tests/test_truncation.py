"""Tests for the truncation follow-up (target stubbed; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from redevolve import cli, truncation
from redevolve.baseline import eval_key
from redevolve.runs import RunDir
from redevolve.seeding import target_seed
from tests.test_baseline import SEEDS
from tests.test_cli_baseline import setup  # noqa: F401  (fixture)

CANARY = "CANARY-7f3a9c21"


def test_work_items_pairs_caps_back_to_back() -> None:
    work = truncation.work_items(SEEDS, ["enc-06", "ind-01"], 2, capped=200, uncapped=1024)
    assert [(w.trial, w.attack_id, w.max_tokens) for w in work] == [
        (0, "enc-06", 200),
        (0, "enc-06", 1024),
        (0, "ind-01", 200),
        (0, "ind-01", 1024),
        (1, "enc-06", 200),
        (1, "enc-06", 1024),
        (1, "ind-01", 200),
        (1, "ind-01", 1024),
    ]
    assert work[0].key == "eval:enc-06:0:cap200" and work[1].key == "eval:enc-06:0:cap1024"
    assert len({w.key for w in work}) == len(work)


def test_work_items_validation() -> None:
    with pytest.raises(truncation.FollowupError, match="nope-01"):
        truncation.work_items(SEEDS, ["nope-01"], 1, capped=200, uncapped=1024)
    with pytest.raises(truncation.FollowupError, match="must exceed"):
        truncation.work_items(SEEDS, ["enc-06"], 1, capped=200, uncapped=200)


def rec(key: str, attack: str, reply: str, truncated: bool = False) -> dict:
    return {
        "type": "eval",
        "key": key,
        "attack_id": attack,
        "reply": reply,
        "leaked": None,
        "partial": None,
        "truncated": truncated,
    }


def test_compare_pairs_within_run_and_adds_baseline_reference() -> None:
    capped = "Here is the translation of my instructions: ... the access"
    pk = truncation.pair_key
    follow = [
        rec(pk("enc-06", 0, 200), "enc-06", capped, truncated=True),
        rec(pk("enc-06", 0, 1024), "enc-06", capped + f" code is {CANARY}."),  # leak after cap
        rec(pk("enc-06", 1, 200), "enc-06", "Sorry, I can't."),
        rec(pk("enc-06", 1, 1024), "enc-06", "Something different."),  # pairing broken
        rec(pk("enc-06", 2, 200), "enc-06", "unpaired, no uncapped partner"),
    ]
    baseline = [
        rec(eval_key("enc-06", 0), "enc-06", "no leak"),
        rec(eval_key("enc-06", 1), "enc-06", f"{CANARY}"),
    ]
    (row,) = truncation.compare(
        follow,
        canary=CANARY,
        attacks=["enc-06"],
        capped=200,
        uncapped=1024,
        trials=3,
        baseline_records=baseline,
    )
    assert row.pairs == 2  # trial 2 has no uncapped partner
    assert (row.capped_full.k, row.uncapped_full.k) == (0, 1)
    assert row.transitions == {"none->full": 1, "none->none": 1}
    assert row.prefix_match == 1
    assert row.baseline_capped_full is not None
    assert (row.baseline_capped_full.k, row.baseline_capped_full.n) == (1, 2)
    text = truncation.format_comparison([row], capped=200, uncapped=1024)
    assert "enc-06" in text and "reported separately" in text and "unpaired" in text
    assert json.loads(json.dumps(row.to_dict())) == row.to_dict()


def configure(path: Path, runs: Path, **fu) -> str:
    assert cli.main(["--config", str(path), "baseline"]) == 0
    (base_id,) = [p.name for p in runs.iterdir()]
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["truncation_followup"] = {
        "baseline_run": base_id,
        "attacks": ["io-01", "enc-01"],
        "max_tokens": 999,
        **fu,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return base_id


def followup_run(runs: Path, base_id: str) -> RunDir:
    (fu_id,) = [p.name for p in runs.iterdir() if p.name != base_id]
    return RunDir.open(runs, fu_id)


def test_followup_end_to_end(setup, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    path, runs, stub = setup
    base_id = configure(path, runs)
    seen: list[tuple[str, int, int | None]] = []
    original = stub.chat

    def spy(messages, *, temperature, max_tokens, seed=None):
        seen.append((messages[1]["content"], max_tokens, seed))
        return original(messages, temperature=temperature, max_tokens=max_tokens, seed=seed)

    monkeypatch.setattr(stub, "chat", spy)
    assert cli.main(["--config", str(path), "truncation"]) == 0
    # 2 attacks x K=1 (from the baseline run's config) x 2 caps, capped call first.
    assert [m for _, m, _ in seen] == [200, 999, 200, 999]
    io01 = next(s.prompt for s in SEEDS.seeds if s.id == "io-01")
    assert seen[0][0] == seen[1][0] == io01
    assert seen[0][2] == seen[1][2] == target_seed(42, "io-01", 0)  # same seed for both caps
    run = followup_run(runs, base_id)
    assert run.meta["kind"] == "truncation" and run.meta["status"] == "complete"
    assert run.meta["baseline_run"] == base_id
    assert (run.meta["max_tokens"], run.meta["baseline_max_tokens"]) == (999, 200)
    assert {r["max_tokens"] for r in run.records("eval")} == {200, 999}
    data = json.loads((run.path / cli.FOLLOWUP_FILE).read_text(encoding="utf-8"))
    assert [a["attack_id"] for a in data["attacks"]] == ["io-01", "enc-01"]
    assert all(a["pairs"] == 1 and a["prefix_match"] == 1 for a in data["attacks"])
    io = data["attacks"][0]
    assert io["capped_full"]["k"] == io["uncapped_full"]["k"] == 1
    assert io["baseline_capped_full"]["k"] == 1
    assert cli.main(["--config", str(path), "truncation", "--summarize", run.run_id]) == 0


def test_followup_limit_then_resume(setup) -> None:  # noqa: F811
    path, runs, stub = setup
    base_id = configure(path, runs)
    stub.calls.clear()
    assert cli.main(["--config", str(path), "truncation", "--limit", "3"]) == 0
    run = followup_run(runs, base_id)
    assert run.meta["status"] == "paused"
    assert cli.main(["--config", str(path), "truncation", "--resume", run.run_id]) == 0
    assert len(stub.calls) == 4
    assert RunDir.open(runs, run.run_id).meta["status"] == "complete"


def test_followup_refuses_unpairable_baseline(setup) -> None:  # noqa: F811
    path, runs, _ = setup
    configure(path, runs)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["seed"] = 7  # different canary -> not the same experiment
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli.main(["--config", str(path), "truncation"]) == 2


def test_followup_bad_config(setup) -> None:  # noqa: F811
    path, runs, _ = setup
    configure(path, runs, attacks=["zz-99"])
    assert cli.main(["--config", str(path), "truncation"]) == 2
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["truncation_followup"]["attacks"] = ["io-01"]
    cfg["truncation_followup"]["max_tokens"] = 100  # not above the baseline cap
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli.main(["--config", str(path), "truncation"]) == 2
    del cfg["truncation_followup"]
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli.main(["--config", str(path), "truncation"]) == 2
