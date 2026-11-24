"""The v1.1 configs may differ from config.yaml only in the keys decided for v1.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from redevolve.config import load_config
from redevolve.guardrail import guardrail_fingerprint
from redevolve.heldout import load_heldout

REPO = Path(__file__).resolve().parents[1]
BASE = load_config(REPO / "config.yaml")


def flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out |= flat(v, f"{prefix}{k}.")
        return out
    return {prefix.rstrip("."): d}


def diff(cfg: dict[str, Any]) -> set[str]:
    a, b = flat(BASE), flat(cfg)
    return {k for k in a.keys() | b.keys() if a.get(k, "<missing>") != b.get(k, "<missing>")}


# defend.heldout_custom: the owner points it at their (private) file before the run
DEFEND = {"defend.dev_sets", "defend.heldout_sets", "defend.headline_sets", "defend.heldout_custom"}
FILES = {"defend.heldout_attack_files.gandalf", "defend.heldout_attack_files.gandalf_code",
         "defend.heldout_attack_files.tensortrust"}  # fmt: skip
REPORT = {f"report.{k}" for k in ("baseline_run", "truncation_run", "guardrail_ablation",
          "judge_fp", "readme", "enabled", "reason")}  # fmt: skip
ALLOWED = {
    "v1.1-qwen.yaml": DEFEND | FILES | REPORT,
    "v1.1-llama.yaml": DEFEND
    | FILES
    | REPORT
    | {"target.model", "target.think", "labelling.sample"}
    | {f"labelling.strata.{k}" for k in ("full", "partial", "tricky", "none")},
}


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_v11_config_differs_only_in_decided_keys(name: str) -> None:
    cfg = load_config(REPO / "configs" / name)
    extra = diff(cfg) - ALLOWED[name]
    assert not extra, f"{name} differs from config.yaml in undeclared keys: {sorted(extra)}"
    assert cfg["report"]["enabled"] is False  # never overwrites the v1 README blocks
    # same guardrail, seed (canary + per-item seeds) and system prompt as v1
    assert guardrail_fingerprint(cfg) == guardrail_fingerprint(BASE)
    assert cfg["seed"] == BASE["seed"] and cfg["defend"]["trials"] == 3
    assert cfg["defend"]["headline_sets"] == ["tensortrust"]


def test_llama_config_target() -> None:
    cfg = load_config(REPO / "configs" / "v1.1-llama.yaml")
    assert cfg["target"]["model"] == "llama3.2:3b" and cfg["target"]["think"] is None
    assert cfg["defend"]["heldout_sets"] == ["gandalf", "tensortrust"]  # headline variant only
    assert cfg["labelling"]["sample"] + 10 == 30  # + the 10 enc-06 replies
    assert cfg["labelling"]["always_include"] == ["enc-06"]


def test_v11_attack_files_are_frozen_sets() -> None:
    for name in ALLOWED:
        files = load_config(REPO / "configs" / name)["defend"]["heldout_attack_files"]
        for set_name, f in files.items():
            if set_name != "tensortrust":  # manifest: needs the local source copy
                assert load_heldout(REPO / f).name == set_name
            else:
                assert f == "data/heldout/tensortrust_ids.yaml"
