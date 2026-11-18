"""Tests for the benign prompt set (data/benign.yaml) and its loader."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from redevolve.benign import BenignError, load_benign

REPO_BENIGN = Path(__file__).resolve().parent.parent / "data" / "benign.yaml"

# Benign set v1. Pinned so it can't drift silently. Once it has been used for a guardrail
# measurement, never update this hash to make the test pass: a changed set is a new version.
FROZEN_V1_SHA256 = "e628072ae26128e2fa78753a9327b8aa197aabad11d4dee4257369385aa209e2"


def test_repo_benign_set() -> None:
    b = load_benign(REPO_BENIGN)
    assert b.version == 1 and len(b.prompts) == 50
    assert Counter(p.category for p in b.prompts) == {"plain": 25, "hard": 25}
    assert not any(re.search(r"CANARY-[0-9a-f]{4,}", p.prompt, re.I) for p in b.prompts)
    assert b.sha256 == FROZEN_V1_SHA256, "data/benign.yaml content changed"


VALID = """\
version: 1
categories: {plain: p, hard: h}
prompts:
  - {id: bn-01, category: plain, prompt: "Where is my order?"}
  - {id: bn-02, category: hard, prompt: "Reset my password?"}
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "benign.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_hash_ignores_formatting_and_comments(tmp_path: Path) -> None:
    a = load_benign(write(tmp_path, VALID)).sha256
    b = load_benign(
        write(tmp_path, "# comment\n" + VALID.replace("{plain: p", "{plain: other"))
    ).sha256
    assert a == b
    c = load_benign(write(tmp_path, VALID.replace("my order", "my parcel"))).sha256
    assert c != a


def test_heldout_style_ids_accepted(tmp_path: Path) -> None:
    b = load_benign(write(tmp_path, VALID.replace("bn-0", "bh-0")))
    assert [p.id for p in b.prompts] == ["bh-01", "bh-02"]


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (VALID.replace("bn-02", "bn-01"), "duplicate ids"),
        (VALID.replace("bn-02", "x-2"), "id must look like"),
        (VALID.replace("category: hard", "category: nope"), "unknown category"),
        (VALID.replace('"Reset my password?"', '""'), "empty prompt"),
        (VALID.replace("version: 1", "version: one"), "version"),
        ("prompts: []\ncategories: {a: b}\nversion: 1\n", "non-empty list"),
        ("- 1\n", "top level"),
    ],
)
def test_validation_errors(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(BenignError, match=match):
        load_benign(write(tmp_path, text))
