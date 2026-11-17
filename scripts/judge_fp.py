"""Measure the judge's false-positive rates on benign text (data/judge_benign.yaml).

Every benign text is checked against N random canaries. Reports full-leak hits (must be 0)
and partial hits by method, per corpus section and per text. Because judge v3 only *adds* the
``anchored`` method after ``split``/``fragment`` fail, the v2 partial rule is exactly
"v3 partials minus anchored", so both rules are measured on the same checks.

Usage: python scripts/judge_fp.py [--canaries 5000] [--seed 2026] [--out results/judge_fp.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from redevolve import judge

CORPUS = Path(__file__).resolve().parent.parent / "data" / "judge_benign.yaml"


def load_corpus(path: Path = CORPUS) -> dict[str, list[str]]:
    """Section name -> benign texts."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {k: [str(t) for t in v] for k, v in data.items()}


def measure(corpus: dict[str, list[str]], n_canaries: int, seed: int) -> dict:
    """Run every text against ``n_canaries`` canaries drawn from ``seed``."""
    rng = random.Random(seed)
    canaries = [judge.make_canary(rng) for _ in range(n_canaries)]
    sections, per_text = {}, []
    started = time.perf_counter()
    for name, texts in corpus.items():
        methods: Counter[str] = Counter()
        full = 0
        for text in texts:
            t_methods: Counter[str] = Counter()
            for c in canaries:
                v = judge.detect(c, text)
                full += v.leaked
                if v.partial:
                    t_methods[v.partial_method] += 1
            methods.update(t_methods)
            if t_methods:
                per_text.append({"section": name, "text": text, "partial_hits": dict(t_methods)})
        checks = len(texts) * n_canaries
        anchored = methods.get("anchored", 0)
        total = sum(methods.values())
        sections[name] = {
            "texts": len(texts),
            "checks": checks,
            "full_hits": full,
            "partial_hits_by_method": dict(methods),
            "partial_rate_v3": total / checks,
            "partial_rate_v2_rule": (total - anchored) / checks,
        }
    checks = sum(s["checks"] for s in sections.values())
    total = sum(sum(s["partial_hits_by_method"].values()) for s in sections.values())
    anchored = sum(s["partial_hits_by_method"].get("anchored", 0) for s in sections.values())
    return {
        "judge_version": judge.JUDGE_VERSION,
        "corpus": str(CORPUS.relative_to(CORPUS.parent.parent)).replace("\\", "/"),
        "n_canaries": n_canaries,
        "seed": seed,
        "checks": checks,
        "full_hits": sum(s["full_hits"] for s in sections.values()),
        "partial_rate_v3": total / checks,
        "partial_rate_v2_rule": (total - anchored) / checks,
        "sections": sections,
        "texts_with_partial_hits": per_text,
        "ms_per_check": 1000 * (time.perf_counter() - started) / checks,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns an exit code (1 if any benign text is judged a full leak)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--canaries", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results/judge_fp.json")
    args = p.parse_args(argv)
    res = measure(load_corpus(), args.canaries, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"judge v{res['judge_version']}: {res['checks']:,} checks, full hits {res['full_hits']}")
    print(
        f"partial rate: v3 {res['partial_rate_v3']:.4%}  "
        f"(v2 rule {res['partial_rate_v2_rule']:.4%})"
    )
    for name, s in res["sections"].items():
        print(
            f"  {name:<9} {s['checks']:>7,} checks  full {s['full_hits']}  "
            f"partial v3 {s['partial_rate_v3']:.4%}  v2 rule {s['partial_rate_v2_rule']:.4%}  "
            f"{s['partial_hits_by_method']}"
        )
    for t in res["texts_with_partial_hits"]:
        print(f"    {t['partial_hits']}  {t['text'][:70]!r}")
    print(f"written to {out}")
    return 1 if res["full_hits"] else 0


if __name__ == "__main__":
    sys.exit(main())
