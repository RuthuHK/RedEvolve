"""Tests for run directories: metadata, durable logging, crash recovery and resume checks."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from redevolve.config import load_config
from redevolve.runs import CONFIG_FILE, LOG_FILE, RUN_FILE, RunDir, RunError, fingerprint
from redevolve.seeding import derive_canary
from redevolve.seeds import load_seeds
from tests.test_config import REPO_CONFIG
from tests.test_seeds import REPO_SEEDS

FIXED_NOW = datetime(2026, 11, 13, 9, 30, 0, tzinfo=UTC)
ENV = {"git_commit": "abc123", "git_dirty": False}


@pytest.fixture
def cfg() -> dict:
    return load_config(REPO_CONFIG)


@pytest.fixture
def seeds():
    return load_seeds(REPO_SEEDS)


def create(tmp_path: Path, cfg: dict, seeds, **kw) -> RunDir:
    return RunDir.create(
        tmp_path,
        "baseline",
        cfg,
        seeds=seeds,
        canary=derive_canary(cfg["seed"]),
        now=lambda: FIXED_NOW,
        env=lambda: ENV,
        **kw,
    )


def resume(tmp_path: Path, run_id: str, cfg: dict, seeds) -> RunDir:
    return RunDir.resume(tmp_path, run_id, cfg, seeds=seeds, canary=derive_canary(cfg["seed"]))


def test_create_writes_metadata_config_and_empty_log(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds, extra={"trials": 10})
    assert run.run_id == "baseline-20261113-093000-s42"
    meta = json.loads((run.path / RUN_FILE).read_text(encoding="utf-8"))
    fp = meta["fingerprint"]
    assert fp["seeds_sha256"] == seeds.sha256
    assert fp["canary"] == derive_canary(42)
    assert fp["mutator_model"] == cfg["mutator"]["model"]
    assert fp["target_model"] == cfg["target"]["model"]
    assert meta["environment"] == ENV
    assert meta["status"] == "running"
    assert meta["trials"] == 10
    assert yaml.safe_load((run.path / CONFIG_FILE).read_text(encoding="utf-8")) == cfg
    assert (run.path / LOG_FILE).read_text() == ""


def test_run_ids_never_collide(tmp_path: Path, cfg: dict, seeds) -> None:
    a = create(tmp_path, cfg, seeds)
    b = create(tmp_path, cfg, seeds)
    assert b.run_id == a.run_id + "-2"


def test_append_records_and_completed_keys(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds)
    run.append({"type": "eval", "key": "io-01:0", "leaked": False})
    run.append({"type": "eval", "key": "io-01:1", "leaked": True})
    run.append({"type": "note", "text": "héllo"})
    assert [r.get("key") for r in run.records("eval")] == ["io-01:0", "io-01:1"]
    assert run.completed_keys() == {"io-01:0", "io-01:1"}
    assert len(list(run.records())) == 3


def test_torn_last_line_is_ignored_and_repaired_on_resume(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds)
    run.append({"type": "eval", "key": "a"})
    with run.log_path.open("a", encoding="utf-8") as f:
        f.write('{"type": "eval", "key": "b", "rep')  # crash mid-write
    assert run.completed_keys() == {"a"}
    resumed = resume(tmp_path, run.run_id, cfg, seeds)
    resumed.append({"type": "eval", "key": "b"})
    assert resumed.completed_keys() == {"a", "b"}
    lines = resumed.log_path.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line) for line in lines)  # file is clean JSONL again


def test_resume_records_session_and_status(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds)
    run.set_status("paused", pause_reason="groq daily quota")
    resumed = resume(tmp_path, run.run_id, cfg, seeds)
    assert resumed.meta["status"] == "running"
    assert resumed.meta["pause_reason"] == "groq daily quota"
    assert len(resumed.meta["sessions"]) == 2
    assert resumed.meta["sessions"][0]["status"] == "paused"
    assert resumed.meta["sessions"][1]["resumed"] is True


def _mutate(cfg: dict, path: str, value) -> dict:
    new = copy.deepcopy(cfg)
    node = new
    *parents, leaf = path.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = value
    return new


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        ("mutator.model", "qwen/qwen3.8-27b", "mutator_model"),
        ("target.model", "other:1b", "target_model"),
        ("target.system_prompt", "Changed. {canary}", "system_prompt_sha256"),
        ("ga.population", 99, "config_sha256"),
        ("target.temperature", 0.1, "config_sha256"),
    ],
)
def test_resume_refused_when_setup_differs(
    tmp_path: Path, cfg: dict, seeds, path: str, value, field: str
) -> None:
    run = create(tmp_path, cfg, seeds)
    with pytest.raises(RunError, match=field):
        resume(tmp_path, run.run_id, _mutate(cfg, path, value), seeds)


def test_resume_refused_when_seed_set_differs(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds)
    other = seeds.__class__(seeds.version, seeds.families, seeds.seeds[:-1], "f" * 64)
    with pytest.raises(RunError, match="seeds_sha256"):
        resume(tmp_path, run.run_id, cfg, other)


def test_resume_refused_when_canary_differs(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds)
    with pytest.raises(RunError, match="canary"):
        RunDir.resume(tmp_path, run.run_id, cfg, seeds=seeds, canary="CANARY-00000000")


def test_resume_refused_when_complete_or_missing(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds)
    run.set_status("complete")
    with pytest.raises(RunError, match="already complete"):
        resume(tmp_path, run.run_id, cfg, seeds)
    with pytest.raises(RunError, match="no run named"):
        resume(tmp_path, "nope", cfg, seeds)


def test_fingerprint_is_insensitive_to_key_order(cfg: dict, seeds) -> None:
    reordered = dict(reversed(list(cfg.items())))
    canary = derive_canary(42)
    assert fingerprint(reordered, seeds, canary) == fingerprint(cfg, seeds, canary)


def test_refuses_to_write_api_key_anywhere(tmp_path: Path, cfg: dict, seeds) -> None:
    key = "gsk_" + "A1b2" * 10
    with pytest.raises(RunError, match="API key"):
        create(tmp_path, _mutate(cfg, "mutator.api_key", key), seeds)
    run = create(tmp_path, cfg, seeds)
    with pytest.raises(RunError, match="API key"):
        run.append({"type": "eval", "reply": f"my key is {key}"})
    with pytest.raises(RunError, match="API key"):
        run.set_status("failed", error=f"401 for {key}")


def test_open_is_read_only_and_exposes_canary(tmp_path: Path, cfg: dict, seeds) -> None:
    run = create(tmp_path, cfg, seeds)
    run.set_status("complete")
    before = (run.path / RUN_FILE).read_text(encoding="utf-8")
    opened = RunDir.open(tmp_path, run.run_id)  # works on complete runs, unlike resume
    assert opened.canary == derive_canary(cfg["seed"])
    assert opened.meta["status"] == "complete"
    assert (run.path / RUN_FILE).read_text(encoding="utf-8") == before
    with pytest.raises(RunError, match="no run named"):
        RunDir.open(tmp_path, "missing")
