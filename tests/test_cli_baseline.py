"""End-to-end tests of the baseline / summary / labels commands (target stubbed; no network)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from redevolve import cli, labelling, target
from redevolve.config import load_config
from redevolve.runs import RUN_FILE, RunDir
from tests.test_baseline import ScriptedTarget
from tests.test_config import REPO_CONFIG
from tests.test_seeds import REPO_SEEDS


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, ScriptedTarget]:
    cfg = load_config(REPO_CONFIG)
    cfg["seeds_file"] = str(REPO_SEEDS)
    cfg["runs_dir"] = str(tmp_path / "runs")
    cfg["baseline"]["trials"] = 1
    cfg["baseline"]["bootstrap_resamples"] = 200
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    stub = ScriptedTarget()
    monkeypatch.setattr(target, "build_target", lambda cfg: stub)
    monkeypatch.setattr(cli, "environment_info", lambda: {"git_dirty": False})
    return path, tmp_path / "runs", stub


def only_run(runs: Path) -> RunDir:
    (run_id,) = [p.name for p in runs.iterdir()]
    return RunDir.open(runs, run_id)


def test_baseline_end_to_end(setup, capsys: pytest.CaptureFixture[str]) -> None:
    path, runs, stub = setup
    assert cli.main(["--config", str(path), "baseline"]) == 0
    out = capsys.readouterr().out
    assert "30 evaluations" in out and "Baseline summary" in out and "gate" in out
    run = only_run(runs)
    assert run.meta["status"] == "complete" and run.meta["trials"] == 1
    summary = json.loads((run.path / cli.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["evaluated"] == 30 and summary["complete"]
    assert (run.path / labelling.DIR / labelling.CSV_FILE).is_file()
    assert len(stub.calls) == 30

    # Re-summarising works; re-exporting refuses to overwrite labels.
    assert cli.main(["--config", str(path), "summary", run.run_id]) == 0
    assert cli.main(["--config", str(path), "labels", "export", run.run_id]) == 2


def test_limit_then_resume(setup, capsys: pytest.CaptureFixture[str]) -> None:
    path, runs, stub = setup
    assert cli.main(["--config", str(path), "baseline", "--limit", "4"]) == 0
    run = only_run(runs)
    assert run.meta["status"] == "paused"
    assert f"--resume {run.run_id}" in capsys.readouterr().out
    assert cli.main(["--config", str(path), "baseline", "--resume", run.run_id]) == 0
    assert len(stub.calls) == 30
    assert RunDir.open(runs, run.run_id).meta["status"] == "complete"
    # A complete run can't be resumed.
    assert cli.main(["--config", str(path), "baseline", "--resume", run.run_id]) == 2


def test_target_down_pauses_with_hint(setup, capsys: pytest.CaptureFixture[str]) -> None:
    path, runs, stub = setup
    stub.fail_after = 3
    assert cli.main(["--config", str(path), "baseline"]) == 1
    err = capsys.readouterr().err
    assert "run paused" in err and "--resume" in err
    assert only_run(runs).meta["status"] == "paused"


def test_dirty_tree_refused(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    path, runs, _ = setup
    monkeypatch.setattr(cli, "environment_info", lambda: {"git_dirty": True})
    assert cli.main(["--config", str(path), "baseline"]) == 2
    assert not runs.exists()
    assert cli.main(["--config", str(path), "baseline", "--allow-dirty", "--limit", "1"]) == 0


def test_bad_limit_and_unknown_run(setup) -> None:
    path, _, _ = setup
    assert cli.main(["--config", str(path), "baseline", "--limit", "0"]) == 2
    assert cli.main(["--config", str(path), "summary", "nope"]) == 2
    assert cli.main(["--config", str(path), "labels", "score", "nope"]) == 2


def test_labels_score(setup, capsys: pytest.CaptureFixture[str]) -> None:
    path, runs, _ = setup
    assert cli.main(["--config", str(path), "baseline"]) == 0
    run = only_run(runs)
    d = run.path / labelling.DIR
    assert cli.main(["--config", str(path), "labels", "score", run.run_id]) == 2  # unlabelled
    key = {i["item_id"]: i for i in json.loads((d / labelling.KEY_FILE).read_text())["items"]}
    with (d / labelling.CSV_FILE).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = key[r["item_id"]]["judge"]
    with (d / labelling.CSV_FILE).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=labelling.CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    capsys.readouterr()
    assert cli.main(["--config", str(path), "labels", "score", run.run_id]) == 0
    assert "kappa" in capsys.readouterr().out
    agreement = json.loads((d / labelling.AGREEMENT_FILE).read_text(encoding="utf-8"))
    assert agreement["accuracy_full_leak"] == 1.0
    assert json.loads((run.path / RUN_FILE).read_text())["status"] == "complete"


def test_summary_archives_older_judge_version_and_uses_run_config(setup) -> None:
    path, runs, _ = setup
    assert cli.main(["--config", str(path), "baseline"]) == 0
    run = only_run(runs)
    summary_path = run.path / cli.SUMMARY_FILE
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    old["judge_version"] = 1
    summary_path.write_text(json.dumps(old), encoding="utf-8")
    # A later edit of the current config must not change how this run is summarised.
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["baseline"]["trials"] = 99
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert cli.main(["--config", str(path), "summary", run.run_id]) == 0
    assert json.loads((run.path / "summary.judge-v1.json").read_text())["judge_version"] == 1
    new = json.loads(summary_path.read_text(encoding="utf-8"))
    assert new["trials"] == 1 and new["complete"]


def test_labels_guide_command(setup, capsys: pytest.CaptureFixture[str]) -> None:
    path, runs, _ = setup
    assert cli.main(["--config", str(path), "baseline"]) == 0
    run = only_run(runs)
    capsys.readouterr()
    assert cli.main(["--config", str(path), "labels", "guide", run.run_id]) == 0
    assert "untouched" in capsys.readouterr().out
