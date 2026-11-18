"""Tests for README generation: marker replacement and the judge sections."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from redevolve import cli, judge, labelling, report
from redevolve.runs import RunDir
from redevolve.stats import Rate
from tests.test_cli_baseline import only_run, setup  # noqa: F401  (fixture)

README = """# T

prose stays
<!-- BEGIN GENERATED: a -->
old a
<!-- END GENERATED: a -->
middle
<!-- BEGIN GENERATED: b -->
<!-- END GENERATED: b -->
tail
"""


def test_update_readme_replaces_only_marked_blocks(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_text(README, encoding="utf-8")
    report.update_readme(p, {"a": "new a with \\1 and $1", "b": "new b"})
    text = p.read_text(encoding="utf-8")
    assert "old a" not in text
    assert "new a with \\1 and $1" in text  # replacement text is literal, not a regex template
    assert "prose stays" in text and "middle" in text and "tail" in text
    report.update_readme(p, {"a": "again"})  # idempotent on regenerated files
    assert p.read_text(encoding="utf-8").count("BEGIN GENERATED: a") == 1


def test_update_readme_missing_marker(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_text("no markers", encoding="utf-8")
    with pytest.raises(report.ReportError, match="marker"):
        report.update_readme(p, {"a": "x"})


def scored_run(setup):  # noqa: F811
    path, runs, _ = setup
    assert cli.main(["--config", str(path), "baseline"]) == 0
    run = only_run(runs)
    d = run.path / labelling.DIR
    key = {i["item_id"]: i for i in json.loads((d / labelling.KEY_FILE).read_text())["items"]}
    with (d / labelling.CSV_FILE).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = key[r["item_id"]]["judge"]
    with (d / labelling.CSV_FILE).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=labelling.CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert cli.main(["--config", str(path), "labels", "score", run.run_id]) == 0
    return path, run


def test_judge_validation_section_numbers_come_from_files(setup) -> None:  # noqa: F811
    _, run = scored_run(setup)
    text = report.judge_validation_section(run)
    agreement = json.loads((run.path / labelling.DIR / labelling.AGREEMENT_FILE).read_text())
    n = agreement["n"]
    assert f"**{n}**" in text and f"({n}/{n};" in text
    assert "One annotator" in text and "Truncation." in text and "Wilson 95% CI" in text


def test_judge_validation_refuses_stale_scoring(setup) -> None:  # noqa: F811
    _, run = scored_run(setup)
    p = run.path / labelling.DIR / labelling.AGREEMENT_FILE
    a = json.loads(p.read_text())
    a["judge_version"] = judge.JUDGE_VERSION - 1
    p.write_text(json.dumps(a))
    with pytest.raises(report.ReportError, match="labels score"):
        report.judge_validation_section(run)


def test_judge_fp_section(tmp_path: Path) -> None:
    fp = {
        "judge_version": judge.JUDGE_VERSION,
        "n_canaries": 10,
        "checks": 20,
        "corpus": "data/x.yaml",
        "sections": {
            "general": {
                "texts": 2,
                "checks": 20,
                "full_hits": 0,
                "partial_rate_v3": 0.05,
                "partial_rate_v2_rule": 0.0,
            }
        },
    }
    p = tmp_path / "fp.json"
    p.write_text(json.dumps(fp))
    text = report.judge_fp_section(p)
    assert "| general (2) | 20 | 0 | 5.000% | 0.000% |" in text
    fp["judge_version"] -= 1
    p.write_text(json.dumps(fp))
    with pytest.raises(report.ReportError, match=r"judge_fp.py"):
        report.judge_fp_section(p)


def test_report_command_end_to_end(setup, tmp_path: Path) -> None:  # noqa: F811
    path, run = scored_run(setup)
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- BEGIN GENERATED: judge-validation -->\n<!-- END GENERATED: judge-validation -->\n"
        "<!-- BEGIN GENERATED: judge-fp -->\n<!-- END GENERATED: judge-fp -->\n",
        encoding="utf-8",
    )

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["report"] = {
        "baseline_run": run.run_id,
        "readme": str(readme),
        "judge_fp": str(Path(__file__).resolve().parent.parent / "results" / "judge_fp.json"),
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli.main(["--config", str(path), "report"]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "Cohen's kappa" in text and "full-leak hits" in text
    assert cli.main(["--config", str(path), "report", "--baseline", "nope"]) == 2


def _rate(k: int, n: int) -> dict:
    return Rate(k, n).to_dict()


def test_truncation_section(tmp_path: Path) -> None:
    d = tmp_path / "truncation-x"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"run_id": "truncation-x"}), encoding="utf-8")
    fu = {
        "baseline_max_tokens": 200,
        "max_tokens": 1024,
        "judge_version": judge.JUDGE_VERSION,
        "attacks": [
            {
                "attack_id": "enc-06",
                "pairs": 10,
                "capped_full": _rate(3, 10),
                "uncapped_full": _rate(5, 10),
                "transitions": {"full->full": 3, "none->full": 2, "none->none": 5},
                "still_truncated": 1,
                "prefix_match": 7,
                "baseline_capped_full": _rate(0, 10),
            }
        ],
    }
    (d / "followup.json").write_text(json.dumps(fu), encoding="utf-8")
    text = report.truncation_section(RunDir.open(tmp_path, "truncation-x"))
    assert "| enc-06 | 0/10 (0.0% to 27.8%) | 3/10 (10.8% to 60.3%) | 5/10" in text
    assert "none->full 2" in text and "| 1/10 | 7/10 |" in text
    fu["judge_version"] -= 1
    (d / "followup.json").write_text(json.dumps(fu), encoding="utf-8")
    with pytest.raises(report.ReportError, match="--summarize truncation-x"):
        report.truncation_section(RunDir.open(tmp_path, "truncation-x"))
