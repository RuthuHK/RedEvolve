"""Generated README sections. Every number in the README comes from here, computed from logs.

The README contains marker pairs::

    <!-- BEGIN GENERATED: judge-validation -->
    ...replaced on every `python -m redevolve report` ...
    <!-- END GENERATED: judge-validation -->

and :func:`update_readme` replaces what is between them. Text outside the markers is
hand-written prose and must not contain result numbers.

This module grows in M9 (baseline, GA, control and guardrail sections). It currently renders
the judge's validation against human labels and its benign false-positive rates.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from . import judge, labelling
from .baseline import rejudge, verdict_label
from .runs import RunDir
from .stats import wilson

BEGIN = "<!-- BEGIN GENERATED: {name} -->"
END = "<!-- END GENERATED: {name} -->"


class ReportError(RuntimeError):
    """An input file is missing, or the README lacks a marker pair."""


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def judge_validation_section(run: RunDir) -> str:
    """Markdown for the judge-vs-human agreement on a run's labelled sample, with caveats.

    Re-reads ``agreement.json`` and ``key.json`` and re-judges the labelled replies from the
    run log, so the method and truncation counts reflect the current judge.

    Raises:
        ReportError: If the run has no scored labels, or they were scored by another judge
            version (re-run ``labels score`` first).
    """
    d = run.path / labelling.DIR
    agreement = _read_json(d / labelling.AGREEMENT_FILE)
    key = _read_json(d / labelling.KEY_FILE)
    if agreement["judge_version"] != judge.JUDGE_VERSION:
        raise ReportError(
            f"agreement.json was scored with judge v{agreement['judge_version']}, current is "
            f"v{judge.JUDGE_VERSION}: run `labels score {run.run_id}` again"
        )
    conf = agreement["confusion_human_by_judge"]
    n = agreement["n"]
    agree = sum(conf[x][x] for x in labelling.LABELS)
    lo, hi = wilson(agree, n)

    keys = {it["key"] for it in key["items"]}
    recs = [r for r in rejudge(run.records("eval"), run.canary) if r["key"] in keys]
    by_label = Counter(verdict_label(r["verdict"]) for r in recs)
    methods = Counter(r["verdict"].method for r in recs if r["verdict"].leaked)
    partial_methods = Counter(r["verdict"].partial_method for r in recs if r["verdict"].partial)
    truncated = sum(bool(r.get("truncated")) for r in recs)
    strata = Counter(it["stratum"] for it in key["items"])
    always = sorted({it["attack_id"] for it in key["items"]} & set(_always_include(run)))
    n_always = sum(it["attack_id"] in always for it in key["items"])

    rows = [
        "| human \\ judge | " + " | ".join(labelling.LABELS) + " |",
        "|---|" + "---|" * len(labelling.LABELS),
    ]
    rows += [
        f"| **{h}** | " + " | ".join(str(conf[h][m]) for m in labelling.LABELS) + " |"
        for h in labelling.LABELS
    ]
    fmt_counts = ", ".join(f"{k} {v}" for k, v in sorted(by_label.items()))
    fmt_strata = ", ".join(f"{k} {v}" for k, v in sorted(strata.items()))
    fmt_methods = ", ".join(f"`{k}` {v}" for k, v in methods.most_common()) or "none"
    fmt_pmethods = ", ".join(f"`{k}` {v}" for k, v in partial_methods.most_common()) or "none"
    lines = [
        f"Judge v{agreement['judge_version']} against hand labels on **{n}** real target "
        f"replies from baseline run `{run.run_id}`:",
        "",
        f"- Agreement (full / partial / none): **{_pct(agreement['accuracy_3class'])}** "
        f"({agree}/{n}; Wilson 95% CI {_pct(lo)} to {_pct(hi)}), "
        f"Cohen's kappa **{agreement['kappa_3class']:.3f}**.",
        f"- Full leak vs not (the headline metric): accuracy "
        f"{_pct(agreement['accuracy_full_leak'])}, kappa {agreement['kappa_full_leak']:.3f}.",
        f"- Disagreements: {len(agreement['disagreements'])}.",
        "",
        *rows,
        "",
        "Caveats:",
        "",
        "- **One annotator**, the project owner, who also wrote the leak definitions and the "
        "judge's design. There is no second labeller and no inter-annotator agreement, so the "
        "labels are not independent of the judge's intent.",
        f"- **Small, non-random sample.** Items were stratified by the judge's own verdict "
        f"({fmt_strata}), plus every reply to {', '.join(always) or 'no attack'} "
        f"({n_always} items). So the agreement is not an estimate of accuracy over all replies, "
        f"and with n={n} the interval above is wide.",
        f"- **Narrow coverage of leak forms.** Judge verdicts in the sample: {fmt_counts}. "
        f"Full leaks by method: {fmt_methods}; partials by method: {fmt_pmethods}. Encoded "
        "forms (base64, hex, acrostics, ...) didn't occur in real replies, so they are "
        "validated only by the synthetic tests in `tests/test_judge.py`.",
        "- **Partial definition fixed before scoring.** The partial rule (judge v3) was "
        "settled after the baseline replies had been seen but before the labels were scored, "
        "and was not changed afterwards. Part of the set was labelled under an earlier guide "
        "wording that had no anchor requirement.",
        f"- **Truncation.** {truncated} of the {n} labelled replies hit the output-token cap. "
        "The labeller and the judge both saw only the visible text, so a leak that would have "
        "come after the cap is missed by both and doesn't show up as a disagreement.",
        f"- The sample was exported with judge v{agreement['export_judge_version']} and "
        f"re-judged with v{agreement['judge_version']}; "
        f"{agreement['verdicts_changed_since_export']} verdicts changed.",
    ]
    return "\n".join(lines)


def _always_include(run: RunDir) -> list[str]:
    cfg = yaml.safe_load((run.path / "config.yaml").read_text(encoding="utf-8"))
    return list(cfg.get("labelling", {}).get("always_include", []))


def judge_fp_section(fp_path: Path) -> str:
    """Markdown for the judge's benign false-positive rates (``scripts/judge_fp.py`` output)."""
    fp = _read_json(fp_path)
    if fp["judge_version"] != judge.JUDGE_VERSION:
        raise ReportError(
            f"{fp_path} is for judge v{fp['judge_version']}, current is "
            f"v{judge.JUDGE_VERSION}: run `python scripts/judge_fp.py` again"
        )
    lines = [
        f"Benign texts checked against {fp['n_canaries']:,} random canaries each "
        f"({fp['checks']:,} checks, judge v{fp['judge_version']}, corpus `{fp['corpus']}`):",
        "",
        "| texts | checks | full-leak hits | partial rate | partial rate, v2 rule |",
        "|---|---|---|---|---|",
    ]
    for name, s in fp["sections"].items():
        lines.append(
            f"| {name} ({s['texts']}) | {s['checks']:,} | {s['full_hits']} | "
            f"{s['partial_rate_v3']:.3%} | {s['partial_rate_v2_rule']:.3%} |"
        )
    lines += [
        "",
        "`anchored` texts are customer-support replies that put a code-like token right after "
        '"code is" / "code:", the worst case for the anchored-partial rule.',
    ]
    return "\n".join(lines)


def update_readme(readme: Path, blocks: dict[str, str]) -> None:
    """Replace each generated block in ``readme`` (marker pairs must already exist).

    Raises:
        ReportError: If a marker pair is missing.
    """
    text = readme.read_text(encoding="utf-8")
    for name, body in blocks.items():
        begin, end = BEGIN.format(name=name), END.format(name=name)
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(text):
            raise ReportError(f"{readme} has no '{begin}' ... '{end}' marker pair")
        text = pattern.sub(lambda _m, b=begin, e=end, x=body: f"{b}\n{x}\n{e}", text)
    readme.write_text(text, encoding="utf-8")
