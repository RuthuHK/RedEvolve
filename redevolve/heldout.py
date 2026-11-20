"""Held-out attack sets for the guardrail evaluation (M8), built from public datasets.

The sets are built once by ``scripts/build_heldout.py`` from a pinned dataset revision and
committed under ``data/heldout/``. Each file records its source (dataset, revision, file,
sha256, licence), the transform applied and how many rows each filter dropped. Its content
hash (version, name, transform and every id/family/prompt) is pinned in
``tests/test_heldout.py`` and recorded in every defend run. Like the seed set, a held-out set
is never edited after results are seen; a change is a new version with a new hash.

v1 (decided 2026-11-19/20, before any held-out evaluation):

- Source: Lakera ``gandalf_ignore_instructions`` (MIT), the official test split, all rows.
- Filters, fixed before download and applied in this order: empty prompt; contains a
  canary-like value; duplicate of an earlier row after :func:`normalise`; shares a run of
  :data:`OVERLAP_WORDS` consecutive normalised words with any dev text (the seeds, the dev benign
  set, the classifier's examples and the classifier check prompts).
- Two variants of the same filtered rows: ``verbatim`` (headline) and ``password-to-code``
  (:func:`password_to_code`), reported separately.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VERSION = 1
FAMILY = "gandalf"
OVERLAP_WORDS = 8
TRANSFORMS = ("verbatim", "password-to-code")
_ID_RE = re.compile(r"^[a-z]+-\d{3}$")
_CANARY_VALUE_RE = re.compile(r"CANARY-[0-9a-f]{4,}", re.I)
_PASSWORD_RE = re.compile(r"\bpassword\b", re.I)
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))


class HeldoutError(ValueError):
    """A held-out attack file is malformed."""


@dataclass(frozen=True)
class HeldoutAttack:
    """One held-out attack prompt, with the dataset row it came from."""

    id: str
    family: str
    prompt: str
    source_row: int


@dataclass(frozen=True)
class HeldoutSet:
    """A validated held-out attack set.

    Attributes:
        name: Set name used in defend logs and config (``defend.heldout_sets``).
        version: Set version.
        transform: One of :data:`TRANSFORMS`.
        attacks: The attacks, in file order.
        source: Provenance (dataset, revision, file, sha256, licence, split).
        filters: Rows dropped per filter (provenance only; not hashed).
        sha256: Content hash, see :func:`content_hash`.
    """

    name: str
    version: int
    transform: str
    attacks: tuple[HeldoutAttack, ...]
    source: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""


def content_hash(name: str, version: int, transform: str, attacks: Iterable[HeldoutAttack]) -> str:
    """SHA-256 over what affects experiments: name, version, transform, id/family/prompt."""
    canonical = {
        "name": name,
        "version": version,
        "transform": transform,
        "attacks": [{"id": a.id, "family": a.family, "prompt": a.prompt} for a in attacks],
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def normalise(text: str) -> str:
    """NFKC, zero-width characters removed, case-folded, runs of whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    return " ".join(t.split())


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", normalise(text))


def shingles(text: str, n: int = OVERLAP_WORDS) -> set[tuple[str, ...]]:
    """All runs of ``n`` consecutive normalised words in ``text``."""
    w = _words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def password_to_code(text: str) -> str:
    """Replace the whole word "password" (any case) with "code", preserving case.

    ``PASSWORD`` -> ``CODE``, a capitalised first letter -> ``Code``, otherwise ``code``.
    Word boundaries are strict: "passwords", "pass word", "pwd" and "passcode" are unchanged.
    """

    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        if w.isupper():
            return "CODE"
        return "Code" if w[0].isupper() else "code"

    return _PASSWORD_RE.sub(repl, text)


def apply_transform(transform: str, text: str) -> str:
    """Apply one of :data:`TRANSFORMS` to a prompt."""
    if transform == "verbatim":
        return text
    if transform == "password-to-code":
        return password_to_code(text)
    raise HeldoutError(f"unknown transform {transform!r}")


@dataclass(frozen=True)
class FilterResult:
    """Rows kept (index, text) and the indices dropped by each filter."""

    kept: list[tuple[int, str]]
    dropped: dict[str, list[int]]


def filter_rows(rows: list[str], dev_texts: Iterable[str]) -> FilterResult:
    """Apply the pre-registered filters (module docstring) in order. Pure; no I/O."""
    dev = set().union(*(shingles(t) for t in dev_texts))
    dropped: dict[str, list[int]] = {"empty": [], "canary_like": [], "duplicate": [], "overlap": []}
    seen: set[str] = set()
    kept: list[tuple[int, str]] = []
    for i, raw in enumerate(rows):
        text = (raw or "").strip()
        if not text:
            dropped["empty"].append(i)
        elif _CANARY_VALUE_RE.search(text):
            dropped["canary_like"].append(i)
        elif normalise(text) in seen:
            dropped["duplicate"].append(i)
        elif shingles(text) & dev:
            seen.add(normalise(text))
            dropped["overlap"].append(i)
        else:
            seen.add(normalise(text))
            kept.append((i, text))
    return FilterResult(kept, dropped)


def build_set(
    name: str,
    id_prefix: str,
    transform: str,
    kept: list[tuple[int, str]],
    *,
    source: dict[str, Any],
    filters: dict[str, Any],
) -> HeldoutSet:
    """A held-out set from filtered rows; ids are ``<prefix>-<source row, 3 digits>``."""
    attacks = tuple(
        HeldoutAttack(f"{id_prefix}-{i:03d}", FAMILY, apply_transform(transform, t), i)
        for i, t in kept
    )
    return HeldoutSet(
        name,
        VERSION,
        transform,
        attacks,
        source,
        filters,
        content_hash(name, VERSION, transform, attacks),
    )


def dump_set(s: HeldoutSet, path: str | Path, *, header: str = "") -> None:
    """Write a held-out set as YAML (prompts as literal blocks where possible)."""
    data = {
        "version": s.version,
        "name": s.name,
        "transform": s.transform,
        "sha256": s.sha256,
        "source": s.source,
        "filters": s.filters,
        "attacks": [
            {"id": a.id, "family": a.family, "source_row": a.source_row, "prompt": a.prompt}
            for a in s.attacks
        ],
    }
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    Path(path).write_text(header + body, encoding="utf-8")


def load_heldout(path: str | Path) -> HeldoutSet:
    """Load and validate a held-out attack file; the stored hash must match the content.

    Raises:
        HeldoutError: On any problem (all listed).
    """
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise HeldoutError(f"cannot read held-out file {p}: {e}") from e
    if not isinstance(data, dict):
        raise HeldoutError(f"{p}: top level must be a mapping")
    problems: list[str] = []
    name, version, transform = data.get("name"), data.get("version"), data.get("transform")
    if not isinstance(name, str) or not name:
        problems.append("'name' must be a non-empty string")
    if not isinstance(version, int):
        problems.append("'version' must be an integer")
    if transform not in TRANSFORMS:
        problems.append(f"'transform' must be one of {TRANSFORMS}")
    raw = data.get("attacks")
    if not isinstance(raw, list) or not raw:
        raise HeldoutError(f"{p}: 'attacks' must be a non-empty list")
    attacks: list[HeldoutAttack] = []
    for k, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"attack #{k}: must be a mapping")
            continue
        aid, prompt = item.get("id"), item.get("prompt")
        if not isinstance(aid, str) or not _ID_RE.match(aid):
            problems.append(f"attack #{k}: id must look like 'gv-001'")
        if not isinstance(prompt, str) or not prompt.strip():
            problems.append(f"attack {aid}: empty prompt")
        attacks.append(
            HeldoutAttack(
                str(aid), str(item.get("family", "")), str(prompt), int(item.get("source_row", -1))
            )
        )
    ids = [a.id for a in attacks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate ids: {dupes}")
    if problems:
        raise HeldoutError(f"{p}: " + "; ".join(problems))
    digest = content_hash(str(name), int(version), str(transform), attacks)  # type: ignore[arg-type]
    if data.get("sha256") != digest:
        raise HeldoutError(f"{p}: stored sha256 does not match the content (edited by hand?)")
    return HeldoutSet(
        str(name),
        int(version),  # type: ignore[arg-type]
        str(transform),
        tuple(attacks),
        dict(data.get("source") or {}),
        dict(data.get("filters") or {}),
        digest,
    )
