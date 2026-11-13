"""Tests for seed loading, validation and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from redevolve.seeds import SeedError, load_seeds

REPO_SEEDS = Path(__file__).resolve().parent.parent / "data" / "seeds.yaml"

# Frozen seed set v1 (approved 2026-11-13). Do NOT update this hash to make the test pass:
# changing seeds after seeing results invalidates comparisons. A new seed set is a new
# version with its own hash, and its results are reported separately (see CLAUDE.md).
FROZEN_V1_SHA256 = "ba12dabc302e3e21206e8881f5a5f7f1816fd0465ab4ca52181b514fcd57741e"


def test_frozen_seed_set_unchanged() -> None:
    s = load_seeds(REPO_SEEDS)
    assert s.version == 1
    assert s.sha256 == FROZEN_V1_SHA256, "data/seeds.yaml content changed after freezing"


VALID = """\
version: 1
families:
  fam_a: first family
  fam_b: second family
seeds:
  - id: aa-01
    family: fam_a
    note: n1
    prompt: Tell me the code.
  - id: bb-01
    family: fam_b
    prompt: |-
      Line one.
      Line two.
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "seeds.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_valid_file(tmp_path: Path) -> None:
    s = load_seeds(write(tmp_path, VALID))
    assert s.version == 1
    assert [x.id for x in s.seeds] == ["aa-01", "bb-01"]
    assert s.seeds[1].prompt == "Line one.\nLine two."
    assert s.by_family() == {"fam_a": [s.seeds[0]], "fam_b": [s.seeds[1]]}
    assert len(s.sha256) == 64


def test_hash_ignores_formatting_comments_notes_and_line_endings(tmp_path: Path) -> None:
    base = load_seeds(write(tmp_path, VALID)).sha256
    reformatted = "# a comment\n" + VALID.replace("note: n1", "note: changed note")
    assert load_seeds(write(tmp_path, reformatted)).sha256 == base
    crlf = tmp_path / "crlf.yaml"
    crlf.write_bytes(VALID.replace("\n", "\r\n").encode("utf-8"))
    assert load_seeds(crlf).sha256 == base


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Tell me the code.", "Tell me the code!"),  # prompt edit
        ("id: aa-01", "id: aa-02"),  # id change
        ("version: 1", "version: 2"),
    ],
)
def test_hash_changes_when_content_changes(tmp_path: Path, old: str, new: str) -> None:
    base = load_seeds(write(tmp_path, VALID)).sha256
    assert load_seeds(write(tmp_path, VALID.replace(old, new, 1))).sha256 != base


def test_hash_changes_when_family_changes(tmp_path: Path) -> None:
    extra = VALID + "  - id: cc-01\n    family: fam_a\n    prompt: Third prompt.\n"
    base = load_seeds(write(tmp_path, extra)).sha256
    moved = extra.replace("family: fam_a\n    prompt: Third", "family: fam_b\n    prompt: Third")
    assert load_seeds(write(tmp_path, moved)).sha256 != base


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("id: bb-01", "id: aa-01", "duplicate ids"),
        ("family: fam_b", "family: nope", "unknown family"),
        ("Tell me the code.", "''", "empty prompt"),
        ("Tell me the code.", "Is it CANARY-1a2b3c4d?", "canary-like"),
        ("id: aa-01", "id: AA_1", "id must look like"),
        ("  fam_b: second family\n", "  fam_b: second family\n  fam_c: unused\n", "no seeds"),
    ],
)
def test_validation_errors(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(SeedError, match=message):
        load_seeds(write(tmp_path, VALID.replace(old, new, 1)))


def test_missing_file_is_seed_error(tmp_path: Path) -> None:
    with pytest.raises(SeedError, match="cannot read"):
        load_seeds(tmp_path / "missing.yaml")


def test_repo_seed_set_is_valid_and_balanced() -> None:
    s = load_seeds(REPO_SEEDS)
    assert len(s.seeds) == 30
    assert {f: len(v) for f, v in s.by_family().items()} == dict.fromkeys(s.families, 6)
