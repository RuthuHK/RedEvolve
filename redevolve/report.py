"""Generated README sections. Every number in the README comes from here, computed from logs.

The README contains marker pairs::

    <!-- BEGIN GENERATED: judge-validation -->
    ...replaced on every `python -m redevolve report` ...
    <!-- END GENERATED: judge-validation -->

and :func:`update_readme` replaces what is between them. Text outside the markers is
hand-written prose and must not contain result numbers.

Sections: ``baseline`` (summary.json), ``judge-validation``, ``judge-fp``,
``truncation-followup`` (followup.json), ``guardrail`` (an ``Ablation.to_dict()`` JSON, held-out
sets first) and ``evolve`` (v1.1; a placeholder, skipped if the README has no markers). A section
whose inputs don't exist yet renders as :data:`PENDING`, never as numbers.
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
from .stats import mcnemar_exact, wilson

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


def _rate(d: dict[str, Any]) -> str:
    return f"{d['k']}/{d['n']} ({_pct(d['ci_low'])} to {_pct(d['ci_high'])})"


def truncation_section(run: RunDir) -> str:
    """Markdown for the truncation follow-up (``followup.json`` in a ``truncation`` run).

    Raises:
        ReportError: If ``followup.json`` is missing or from another judge version.
    """
    fu = _read_json(run.path / "followup.json")
    if fu["judge_version"] != judge.JUDGE_VERSION:
        raise ReportError(
            f"followup.json is for judge v{fu['judge_version']}, current is "
            f"v{judge.JUDGE_VERSION}: run `truncation --summarize {run.run_id}` again"
        )
    cap, big = fu["baseline_max_tokens"], fu["max_tokens"]
    lines = [
        f"Attacks whose baseline replies mostly hit the output cap, replayed in run "
        f"`{run.run_id}` at **both** caps ({cap} and {big} tokens), back to back with the same "
        "seed. Full-leak rates with Wilson 95% CIs:",
        "",
        f"| attack | baseline @{cap} (earlier session) | follow-up @{cap} | follow-up @{big} | "
        f"transitions @{cap} -> @{big} | still truncated @{big} | exact pairs |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in fu["attacks"]:
        base = a["baseline_capped_full"]
        trans = ", ".join(f"{k} {v}" for k, v in a["transitions"].items())
        lines.append(
            f"| {a['attack_id']} | {_rate(base) if base else '-'} | {_rate(a['capped_full'])} | "
            f"{_rate(a['uncapped_full'])} | {trans} | {a['still_truncated']}/{a['pairs']} | "
            f"{a['prefix_match']}/{a['pairs']} |"
        )
    lines += [
        "",
        f"- *Exact pairs* = the @{big} reply starts with its @{cap} partner, i.e. generation was "
        "identical up to the cap. Pairs that diverge show that even same-session, same-seed "
        "replies are not always reproducible.",
        f"- *Baseline @{cap}* comes from a different session than the follow-up, so it is an "
        "unpaired reference: the difference between the two columns at the same cap is "
        "cross-session variance, not an effect of the cap.",
    ]
    return "\n".join(lines)


PENDING = "_Not run yet._"


def _ci(lo: float, hi: float) -> str:
    return f"{_pct(lo)} to {_pct(hi)}"


def baseline_section(run: RunDir) -> str:
    """Markdown for the undefended baseline, from the run's ``summary.json``.

    Raises:
        ReportError: If the summary is missing, incomplete, or from another judge version
            (re-run ``summary RUN_ID`` first).
    """
    s = _read_json(run.path / "summary.json")
    if s["judge_version"] != judge.JUDGE_VERSION:
        raise ReportError(
            f"summary.json is for judge v{s['judge_version']}, current is "
            f"v{judge.JUDGE_VERSION}: run `summary {run.run_id}` again"
        )
    if not s["complete"]:
        raise ReportError(f"baseline {run.run_id} is incomplete ({s['evaluated']}/{s['expected']})")
    o = s["overall"]
    clo, chi = o["full_cluster_ci"]
    n_attacks = len(s["attacks"])
    lines = [
        f"Run `{run.run_id}`: {n_attacks} seed attacks x K={s['trials']} trials = "
        f"{s['evaluated']} target replies, judge v{s['judge_version']}.",
        "",
        f"- **Overall full-leak ASR: {_pct(o['full']['rate'])}** ({o['full']['k']}/"
        f"{o['full']['n']}; cluster bootstrap 95% CI over attacks {_ci(clo, chi)}; pooled Wilson "
        f"{_ci(o['full']['ci_low'], o['full']['ci_high'])}, too narrow because trials of one "
        "attack are correlated).",
        f"- Partial leaks: {_pct(o['partial']['rate'])}. Replies cut off by the output cap: "
        f"{_pct(o['truncated']['rate'])}.",
        "- Full leaks by method: "
        + (", ".join(f"`{k}` {v}" for k, v in s["full_leak_methods"].items()) or "none")
        + ".",
        "",
        "| family | attacks | full-leak ASR | cluster bootstrap 95% CI | partial | truncated |",
        "|---|---|---|---|---|---|",
    ]
    for f in s["families"]:
        flo, fhi = f["full_cluster_ci"]
        lines.append(
            f"| {f['name']} | {f['n_attacks']} | {_pct(f['full']['rate'])} "
            f"({f['full']['k']}/{f['full']['n']}) | {_ci(flo, fhi)} | "
            f"{_pct(f['partial']['rate'])} | {_pct(f['truncated']['rate'])} |"
        )
    g = s["gate"]
    lines += [
        "",
        f"Low-baseline gate (fixed before the run: at least {g['min_seeds']} seed with a "
        f"full-leak rate of at least {_pct(g['min_seed_rate'])}): "
        f"**{'passed' if g['passed'] else 'failed'}**, qualifying seeds: "
        f"{', '.join(g['qualifying']) or 'none'}.",
        "",
        "<details><summary>Per-attack results</summary>",
        "",
        "| attack | family | full leak | Wilson 95% CI | partial | truncated |",
        "|---|---|---|---|---|---|",
    ]
    for a in s["attacks"]:
        lines.append(
            f"| {a['name']} | {a['family']} | {a['full']['k']}/{a['full']['n']} | "
            f"{_ci(a['full']['ci_low'], a['full']['ci_high'])} | {a['partial']['k']}/"
            f"{a['partial']['n']} | {a['truncated']['k']}/{a['truncated']['n']} |"
        )
    lines += ["", "</details>"]
    return "\n".join(lines)


def _latency(r: dict[str, Any]) -> str:
    """Mean / p95 added latency; for rows with network stalls, also without them."""
    text = f"{r['mean_added_latency_s'] * 1000:.0f} / {r['p95_added_latency_s'] * 1000:.0f} ms"
    if r.get("stalled"):
        text += (
            f" (excl. {r['stalled']} stalled: {r['mean_added_latency_excl_stalls_s'] * 1000:.0f}"
            f" / {r['p95_added_latency_excl_stalls_s'] * 1000:.0f} ms)"
        )
    return text


def _ablation_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| layers | full-leak ASR | cluster bootstrap 95% CI | blocked | redacted | "
        "fixed vs none | McNemar p | added latency (mean / p95) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = r["full_cluster_ci"]
        out.append(
            f"| {r['config']} | {_pct(r['full']['rate'])} ({r['full']['k']}/{r['full']['n']}) | "
            f"{_ci(*ci) if ci else '-'} | {_pct(r['blocked']['rate'])} | "
            f"{_pct(r['redacted']['rate'])} | {r['fixed_vs_none']} | "
            f"{r['mcnemar_p_vs_none']:.2g} | {_latency(r)} |"
        )
    return out


def _benign_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| layers | false-positive rate | Wilson 95% CI | earlier definition | blocked | "
        "redacted | leaks reaching the user | added latency (mean / p95) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fp, v1 = r["false_positive"], r["false_positive_v1"]
        out.append(
            f"| {r['config']} | {_pct(fp['rate'])} ({fp['k']}/{fp['n']}) | "
            f"{_ci(fp['ci_low'], fp['ci_high'])} | {_pct(v1['rate'])} ({v1['k']}/{v1['n']}) | "
            f"{r['blocked']['k']} | {r['redacted']['k']} | {r['residual_leak']['k']} | "
            f"{_latency(r)} |"
        )
    return out


def _spontaneous_line(rows: list[dict[str, Any]]) -> list[str]:
    """One line on benign replies that leaked with no guardrail at all."""
    none = next((r for r in rows if r["config"] == "none"), None)
    if none is None:
        return []
    sp = none["spontaneous_leak"]
    keys = ", ".join(f"`{k.split(':')[1]}`" for k in none["spontaneous"]) or "none"
    return [
        f"Spontaneous leaks (benign prompt, no guardrail, reply leaked the canary): "
        f"{sp['k']}/{sp['n']} ({_pct(sp['rate'])}, Wilson 95% CI "
        f"{_ci(sp['ci_low'], sp['ci_high'])}): {keys}.",
        "",
    ]


def _headline_summary(ab: dict[str, Any], headline: list[str], dev_sets: list[str]) -> list[str]:
    """Block rate on the headline sets (the v1 headline), the floor-effect check, and the
    leak reduction on the dev sets (optimistic)."""
    lines: list[str] = []
    configs = [r["config"] for r in ab["attacks"] if r["set"] == headline[0]] if headline else []
    rows = {(r["set"], r["config"]): r for r in ab["attacks"]}
    benign = {(r["set"], r["config"]): r for r in ab["benign"]}
    if not configs:
        return lines
    head = " | ".join(f"blocked: `{s}` (cluster bootstrap 95% CI)" for s in headline)
    bsets = [b for b in ("benign_heldout", "benign") if (b, "none") in benign]
    bhead = "".join(f" | false positives: `{b}`" for b in bsets)

    def table(cs: list[str]) -> list[str]:
        out = [f"| layers | {head}{bhead} |", "|---|" + "---|" * (len(headline) + len(bsets))]
        for c in cs:
            cells = []
            for hs in headline:
                r = rows[(hs, c)]
                b, ci = r["blocked"], r.get("blocked_cluster_ci")
                cells.append(f"{_pct(b['rate'])} ({b['k']}/{b['n']}) {_ci(*ci) if ci else '-'}")
            for bs in bsets:
                fp = benign.get((bs, c), {}).get("false_positive")
                cells.append(f"{fp['k']}/{fp['n']}" if fp else "-")
            out.append(f"| {c} | " + " | ".join(cells) + " |")
        return [*out, ""]

    main = [c for c in configs if c != "none" and not _is_cmp({"config": c})]
    cmp_ = [c for c in configs if _is_cmp({"config": c})]
    lines += ["**Headline: block rate on the held-out attacks**", "", *table(main)]
    if cmp_:
        lines += [f"*Comparison classifier: {CMP_TITLE}*", "", *table(cmp_)]
    for s in headline:
        base = rows[(s, "none")]["full"]
        best_p = mcnemar_exact(base["k"], 0)
        verdict = (
            "too few to show a significant reduction even if every one were stopped, so leak "
            "reduction is not measurable on this set; the block rate is the held-out headline"
            if best_p >= 0.05
            else "enough for a reduction to be testable"
        )
        lines += [
            f"Floor check on `{s}`: with no guardrail {base['k']}/{base['n']} replies leaked "
            f"({_pct(base['rate'])}). The smallest exact McNemar p any configuration could reach "
            f"is {best_p:.2g}: {verdict}.",
            "",
        ]
    for s in dev_sets:
        if (s, "none") in rows and (s, "a+b+c") in rows:
            n0, n1 = rows[(s, "none")]["full"], rows[(s, "a+b+c")]
            lines += [
                f"Leak reduction is reported on the dev set `{s}` (optimistic: the guardrail was "
                f"designed on it): full-leak ASR {n0['k']}/{n0['n']} with no guardrail vs "
                f"{n1['full']['k']}/{n1['full']['n']} with a+b+c, exact McNemar "
                f"p = {n1['mcnemar_p_vs_none']:.2g}.",
                "",
            ]
    return lines


def _layer_latency_table(ab: dict[str, Any]) -> list[str]:
    ll = ab.get("layer_latency") or {}
    if not ll:
        return []
    thr = ab.get("network_stall_s", 10.0)
    names = {
        "a": "a: input filter (re-timed at summary)",
        "b": "b: classifier (local qwen3:1.7b)",
        "p": "p: Prompt Guard (Groq API)",
        "c": "c: output filter (re-timed at summary)",
    }
    out = [
        "*Per-call latency by layer (all sets)*",
        "",
        f"| layer | calls | median | p95 | max | calls over {thr:.0f} s | p95 / max without them |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in ll.items():
        out.append(
            f"| {names.get(k, k)} | {v['n']} | {v['median_s'] * 1000:.0f} ms | "
            f"{v['p95_s'] * 1000:.0f} ms | {v['max_s']:.1f} s | {v['over_stall_threshold']} | "
            f"{v['p95_excl_over_threshold_s'] * 1000:.0f} ms / "
            f"{v['max_excl_over_threshold_s']:.1f} s |"
        )
    return [*out, ""]


CMP_TITLE = "Llama Prompt Guard 2 86M (`p`) in place of the classifier (`b`)"


def _is_cmp(row: dict[str, Any]) -> bool:
    return "p" in row["config"].split("+")


def guardrail_section(
    ablation_path: Path,
    *,
    dev_sets: list[str],
    heldout_sets: list[str],
    headline_sets: list[str] | None = None,
    aware_sets: list[str] | None = None,
) -> str:
    """Markdown for the paired guardrail ablation (``Ablation.to_dict()`` JSON).

    Headline held-out attack sets come first, then other held-out sets (variants, labelled as
    not the headline; ``headline_sets`` defaults to all held-out sets), then dev sets under an
    explicit "optimistic" heading. The dev and held-out benign sets get separate tables.

    Raises:
        ReportError: If the file is missing, from another judge version, or contains an attack
            set that is neither dev nor held-out (every set must be declared in config).
    """
    ab = _read_json(ablation_path)
    if ab["judge_version"] != judge.JUDGE_VERSION:
        raise ReportError(
            f"{ablation_path} is for judge v{ab['judge_version']}, current is "
            f"v{judge.JUDGE_VERSION}: re-summarise the defend run"
        )
    if ab.get("fp_definition") != 2:
        raise ReportError(
            f"{ablation_path} uses an older false-positive definition: re-summarise the defend "
            "run (python -m redevolve defend --summarize RUN_ID)"
        )
    sets = list(dict.fromkeys(r["set"] for r in ab["attacks"]))
    aware = list(aware_sets or [])
    undeclared = [s for s in sets if s not in dev_sets and s not in heldout_sets and s not in aware]
    if undeclared:
        raise ReportError(f"attack sets not declared as dev or held-out in config: {undeclared}")
    lines: list[str] = []
    if "guardrail" in ab:
        g = ab["guardrail"]
        target = f" Target: `{ab['target_model']}`." if ab.get("target_model") else ""
        lines += [f"Guardrail v{g['version']} (`{g['sha256'][:12]}`).{target}", ""]
    headline = heldout_sets if headline_sets is None else headline_sets
    bad = [s for s in headline if s not in heldout_sets]
    if bad:
        raise ReportError(f"headline sets must be held-out sets: {bad}")
    lines += _headline_summary(ab, [s for s in headline if s in sets], dev_sets)
    for title, group in (
        ("Held-out attacks (headline: not seen while designing the guardrail)", headline),
        (
            "Held-out variant (reported separately, not the headline)",
            [s for s in heldout_sets if s not in headline],
        ),
        (
            "Guardrail-aware held-out set (not used in design, but written by someone who knows "
            "the guardrail; reported separately)",
            aware,
        ),
        ("Dev attacks (the guardrail was designed while reading these: optimistic)", dev_sets),
    ):
        for s in (x for x in sets if x in group):
            rows = [r for r in ab["attacks"] if r["set"] == s]
            lines += [f"**{title}: `{s}`**", ""]
            lines += _ablation_table([r for r in rows if not _is_cmp(r)])
            lines.append("")
            cmp_rows = [r for r in rows if _is_cmp(r) or r["config"] == "none"]
            if len(cmp_rows) > 1:
                lines += [f"*Comparison classifier on `{s}`: {CMP_TITLE}*", ""]
                lines += _ablation_table(cmp_rows)
                lines.append("")
    for s, title in (
        ("benign_heldout", "Held-out benign prompts (written by the owner, unseen during design)"),
        ("benign", "Dev benign prompts (written alongside the guardrail)"),
    ):
        rows = [r for r in ab["benign"] if r.get("set", "benign") == s]
        if rows:
            lines += [f"**{title}**", "", *_spontaneous_line(rows)]
            lines += [*_benign_table([r for r in rows if not _is_cmp(r)]), ""]
            cmp_rows = [r for r in rows if _is_cmp(r) or r["config"] == "none"]
            if len(cmp_rows) > 1:
                lines += [f"*Comparison classifier: {CMP_TITLE}*", "", *_benign_table(cmp_rows), ""]
    if ab.get("skipped_configs"):
        lines.append(
            f"Not evaluated (no classifier decisions): {', '.join(ab['skipped_configs'])}."
        )
    lines += _layer_latency_table(ab)
    lines.append(
        "False positive = a benign prompt that was blocked, or whose reply (c) changed although "
        "it did not leak; *earlier definition* also counted (c) redacting a spontaneous leak. "
        f"Stalled = a Prompt Guard call over {ab.get('network_stall_s', 10.0):.0f} s."
    )
    lines.append("")
    lines.append(
        "All layer combinations are judged on the same target replies (one call per attack x "
        "trial), so *fixed vs none* counts paired outcomes and the McNemar test is exact."
    )
    return "\n".join(lines).rstrip()


def update_readme(
    readme: Path, blocks: dict[str, str], *, optional: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """Replace each generated block in ``readme`` (marker pairs must already exist).

    Blocks named in ``optional`` are skipped when the README has no marker pair for them.

    Returns:
        The names of the blocks that were written, in order.

    Raises:
        ReportError: If a required marker pair is missing.
    """
    text = readme.read_text(encoding="utf-8")
    written: list[str] = []
    for name, body in blocks.items():
        begin, end = BEGIN.format(name=name), END.format(name=name)
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(text):
            if name in optional:
                continue
            raise ReportError(f"{readme} has no '{begin}' ... '{end}' marker pair")
        text = pattern.sub(lambda _m, b=begin, e=end, x=body: f"{b}\n{x}\n{e}", text)
        written.append(name)
    readme.write_text(text, encoding="utf-8")
    return written
