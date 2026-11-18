"""Tests for the M9 README sections: baseline, guardrail ablation, placeholders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from redevolve import cli, defend, judge, report
from redevolve.config import load_config
from redevolve.guardrail import ClassifierDecision, guardrail_fingerprint
from tests.test_cli_baseline import setup  # noqa: F401  (fixture)
from tests.test_config import REPO_CONFIG
from tests.test_defend import CLEAN, item
from tests.test_report import scored_run


def test_baseline_section_from_summary(setup) -> None:  # noqa: F811
    _, run = scored_run(setup)
    s = json.loads((run.path / "summary.json").read_text(encoding="utf-8"))
    text = report.baseline_section(run)
    o = s["overall"]["full"]
    assert f"({o['k']}/{o['n']};" in text and "cluster bootstrap" in text
    assert text.count("| instruction_override | 6 |") == 1  # family table row
    assert "Low-baseline gate" in text and "<details>" in text and "| io-01 |" in text

    s["judge_version"] -= 1
    (run.path / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    with pytest.raises(report.ReportError, match="summary"):
        report.baseline_section(run)
    s["judge_version"] += 1
    s["complete"] = False
    (run.path / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    with pytest.raises(report.ReportError, match="incomplete"):
        report.baseline_section(run)


def write_ablation(tmp_path: Path) -> Path:
    items = [
        item("seeds:io-01:0", set_name="seeds", a=True),
        item("seeds:io-02:0", set_name="seeds"),
        item("ga:g1:0", set_name="ga"),
        item("ga:g2:0", set_name="ga", b=True),
        item("benign:bn-01:0", reply=CLEAN, set_name="benign", family="plain"),
        item("benign_heldout:bh-01:0", reply=CLEAN, set_name="benign_heldout", family="x", b=True),
    ]
    ab = defend.ablate(items, canary=item().reply.split()[-1].rstrip("."), block_message="no",
                       run_seed=1, n_boot=100).to_dict()  # fmt: skip
    ab["guardrail"] = guardrail_fingerprint(load_config(REPO_CONFIG))
    p = tmp_path / "ablation.json"
    p.write_text(json.dumps(ab), encoding="utf-8")
    return p


def test_guardrail_section_puts_heldout_first_and_separates_benign(tmp_path: Path) -> None:
    p = write_ablation(tmp_path)
    text = report.guardrail_section(p, dev_sets=["seeds"], heldout_sets=["ga", "control"])
    held = text.index("Held-out attacks (headline")
    dev = text.index("Dev attacks")
    assert held < dev and "`ga`" in text[held:dev] and "`seeds`" in text[dev:]
    assert text.index("Held-out benign") < text.index("Dev benign")
    assert "Guardrail v2" in text and "| a+b+c |" in text
    assert "Comparison classifier" not in text  # no p decisions in this ablation
    # every attack set must be declared, so nothing is silently reported as the headline
    with pytest.raises(report.ReportError, match="not declared"):
        report.guardrail_section(p, dev_sets=["seeds"], heldout_sets=[])
    d = json.loads(p.read_text(encoding="utf-8"))
    d["judge_version"] = judge.JUDGE_VERSION - 1
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(report.ReportError, match="judge"):
        report.guardrail_section(p, dev_sets=["seeds"], heldout_sets=["ga"])


def test_update_readme_optional_blocks(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_text("<!-- BEGIN GENERATED: a -->\n<!-- END GENERATED: a -->\n", encoding="utf-8")
    report.update_readme(p, {"a": "x", "b": "y"}, optional={"b"})
    assert "x" in p.read_text(encoding="utf-8")
    with pytest.raises(report.ReportError):
        report.update_readme(p, {"b": "y"})


def test_report_renders_placeholders_and_guardrail(setup, tmp_path: Path) -> None:  # noqa: F811
    path, run = scored_run(setup)
    names = ["baseline", "judge-validation", "judge-fp", "evolve", "guardrail"]
    readme = tmp_path / "README.md"
    readme.write_text(
        "".join(f"<!-- BEGIN GENERATED: {n} -->\n<!-- END GENERATED: {n} -->\n" for n in names),
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
    assert text.count(report.PENDING) == 2 and "Overall full-leak ASR" in text

    cfg["report"]["guardrail_ablation"] = str(write_ablation(tmp_path))
    cfg["defend"]["heldout_sets"] = ["ga"]
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli.main(["--config", str(path), "report"]) == 0
    text = readme.read_text(encoding="utf-8")
    assert text.count(report.PENDING) == 1 and "Held-out attacks (headline" in text


def test_guardrail_section_reports_comparison_separately(tmp_path: Path) -> None:
    from dataclasses import replace

    pg = ClassifierDecision("injection", True, "0.9", 0.2, "groq/pg", 0.9)
    items = [
        replace(item("ga:g1:0", set_name="ga"), comparison=pg),
        replace(item("benign:bn-01:0", reply=CLEAN, set_name="benign"), comparison=pg),
    ]
    ab = defend.ablate(items, canary=item().reply.split()[-1].rstrip("."), block_message="no",
                       run_seed=1, n_boot=100).to_dict()  # fmt: skip
    p = tmp_path / "ablation.json"
    p.write_text(json.dumps(ab), encoding="utf-8")
    text = report.guardrail_section(p, dev_sets=[], heldout_sets=["ga"])
    main, _, cmp_part = text.partition("*Comparison classifier on `ga`")
    assert "| a+b+c |" in main and "| a+p+c |" not in main
    assert "| a+p+c |" in cmp_part and "| none |" in cmp_part
    assert "*Comparison classifier: Llama Prompt Guard" in text
