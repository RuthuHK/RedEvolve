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

v1.1 (decided 2026-11-23/24, before any v1.1 run): Tensor Trust's extraction-robustness
benchmark has no licence, so its texts are never committed. The repo holds an **id manifest**
(``kind: manifest``): the pinned source, the sampling rule, and per attack its ``sample_id`` and
the sha256 of its text. :func:`load_manifest` rebuilds the texts from a local copy of the pinned
file (``scripts/build_tensortrust.py --fetch``) and checks every hash. Filters: the v1 filters
plus ``too_long`` (more than ``max_chars`` characters) after ``canary_like``, with the Gandalf
prompts added to the overlap texts; then a sample of ``n`` attacks, ordered by
``sha256(salt + "|" + sample_id)``.
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


def filter_rows(
    rows: list[str], dev_texts: Iterable[str], *, max_chars: int | None = None
) -> FilterResult:
    """Apply the pre-registered filters (module docstring) in order. Pure; no I/O.

    With ``max_chars``, a ``too_long`` filter runs after ``canary_like`` (v1.1 sets only).
    """
    dev = set().union(*(shingles(t) for t in dev_texts))
    dropped: dict[str, list[int]] = {"empty": [], "canary_like": []}
    if max_chars is not None:
        dropped["too_long"] = []
    dropped |= {"duplicate": [], "overlap": []}
    seen: set[str] = set()
    kept: list[tuple[int, str]] = []
    for i, raw in enumerate(rows):
        text = (raw or "").strip()
        if not text:
            dropped["empty"].append(i)
        elif _CANARY_VALUE_RE.search(text):
            dropped["canary_like"].append(i)
        elif max_chars is not None and len(text) > max_chars:
            dropped["too_long"].append(i)
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


def _read_yaml(p: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise HeldoutError(f"cannot read held-out file {p}: {e}") from e
    if not isinstance(data, dict):
        raise HeldoutError(f"{p}: top level must be a mapping")
    return data


def load_heldout(path: str | Path) -> HeldoutSet:
    """Load and validate a held-out attack file; the stored hash must match the content.

    An id manifest (``kind: manifest``) is rebuilt from its local source copy instead, see
    :func:`load_manifest`.

    Raises:
        HeldoutError: On any problem (all listed).
    """
    p = Path(path)
    data = _read_yaml(p)
    if data.get("kind") == MANIFEST_KIND:
        return load_manifest(p)
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


# --- id manifests (v1.1: sources whose texts may not be redistributed) ------------------

MANIFEST_KIND = "manifest"


def text_sha256(text: str) -> str:
    """SHA-256 of one prompt's UTF-8 text (as used in the set, after stripping)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_order(salt: str, sample_id: str | int) -> str:
    """The pre-registered sampling order key: ``sha256(salt + "|" + sample_id)``."""
    return hashlib.sha256(f"{salt}|{sample_id}".encode()).hexdigest()


def manifest_hash(
    name: str, version: int, transform: str, entries: Iterable[dict[str, Any]]
) -> str:
    """SHA-256 over a manifest's name, version, transform and id/family/sample_id/text hash."""
    canonical = {
        "name": name,
        "version": version,
        "transform": transform,
        "attacks": [
            {k: e[k] for k in ("id", "family", "sample_id", "prompt_sha256")} for e in entries
        ],
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path, *, id_field: str, text_field: str) -> list[tuple[str, str]]:
    """(sample_id as str, text) per non-empty JSONL line, in file order.

    Raises:
        HeldoutError: If the file can't be read or a line lacks a field.
    """
    out: list[tuple[str, str]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise HeldoutError(f"cannot read {path}: {e}") from e
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out.append((str(row[id_field]), str(row[text_field] or "")))
        except (ValueError, KeyError, TypeError) as e:
            raise HeldoutError(f"{path}: line {n + 1}: {e}") from e
    return out


def build_manifest(
    name: str,
    id_prefix: str,
    family: str,
    rows: list[tuple[str, str]],
    *,
    dev_texts: Iterable[str],
    n: int,
    salt: str,
    max_chars: int,
    source: dict[str, Any],
) -> tuple[dict[str, Any], HeldoutSet]:
    """Filter, sample and freeze ``rows`` (sample_id, text) as an id manifest (+ the set).

    The ``n`` kept rows first in :func:`sample_order` are taken and listed in file order, with
    ids ``<prefix>-000`` onwards. Pure; the caller writes the manifest.
    """
    fr = filter_rows([t for _, t in rows], dev_texts, max_chars=max_chars)
    ranked = sorted(fr.kept, key=lambda it: sample_order(salt, rows[it[0]][0]))
    chosen = sorted(ranked[:n])
    if len(chosen) < n:
        raise HeldoutError(f"only {len(chosen)} rows left after filtering, need {n}")
    attacks = tuple(
        HeldoutAttack(f"{id_prefix}-{k:03d}", family, text, row)
        for k, (row, text) in enumerate(chosen)
    )
    entries = [
        {
            "id": a.id,
            "family": a.family,
            "sample_id": rows[a.source_row][0],
            "source_row": a.source_row,
            "prompt_sha256": text_sha256(a.prompt),
        }
        for a in attacks
    ]
    filters = {
        "rows": len(rows),
        "dropped": {k: len(v) for k, v in fr.dropped.items()},
        "kept": len(fr.kept),
        "sampled": n,
    }
    digest = content_hash(name, VERSION, "verbatim", attacks)
    manifest = {
        "version": VERSION,
        "name": name,
        "kind": MANIFEST_KIND,
        "transform": "verbatim",
        "manifest_sha256": manifest_hash(name, VERSION, "verbatim", entries),
        "set_sha256": digest,
        "source": source,
        "sampling": {
            "n": n,
            "salt": salt,
            "order": "sha256(salt|sample_id)",
            "max_chars": max_chars,
        },
        "filters": filters,
        "attacks": entries,
    }
    hs = HeldoutSet(name, VERSION, "verbatim", attacks, source, filters, digest)
    return manifest, hs


def dump_manifest(manifest: dict[str, Any], path: str | Path, *, header: str = "") -> None:
    """Write an id manifest as YAML."""
    body = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=100)
    Path(path).write_text(header + body, encoding="utf-8")


def load_manifest(path: str | Path, *, source_file: str | Path | None = None) -> HeldoutSet:
    """Rebuild a held-out set from an id manifest and a local copy of its pinned source.

    ``source_file`` defaults to the manifest's ``source.cache_path`` (relative to the working
    directory). Checks, in order: the manifest's own hash, the source file's sha256, that every
    ``sample_id`` exists, every text's sha256, and the rebuilt set's content hash.

    Raises:
        HeldoutError: On any mismatch, or if the source file is missing (with a fetch hint).
    """
    p = Path(path)
    data = _read_yaml(p)
    name, version, transform = data.get("name"), data.get("version"), data.get("transform")
    entries = data.get("attacks")
    src = data.get("source") or {}
    if not isinstance(entries, list) or not entries or not isinstance(name, str):
        raise HeldoutError(f"{p}: a manifest needs 'name' and a non-empty 'attacks' list")
    try:
        if data.get("manifest_sha256") != manifest_hash(name, version, transform, entries):
            raise HeldoutError(f"{p}: stored manifest_sha256 does not match (edited by hand?)")
    except (KeyError, TypeError) as e:
        raise HeldoutError(f"{p}: malformed manifest entry: {e}") from e
    f = Path(source_file or src.get("cache_path", ""))
    if not f.is_file():
        raise HeldoutError(
            f"{p}: source file {f} not found; rebuild it locally with "
            "`python scripts/build_tensortrust.py --fetch` (the texts are not redistributed)"
        )
    if file_sha256(f) != src.get("sha256"):
        raise HeldoutError(f"{f}: sha256 differs from the pinned source in {p}")
    texts = dict(read_jsonl(f, id_field=src["id_field"], text_field=src["text_field"]))
    attacks: list[HeldoutAttack] = []
    for e in entries:
        text = texts.get(str(e["sample_id"]))
        if text is None:
            raise HeldoutError(f"{p}: sample_id {e['sample_id']} not in {f}")
        text = text.strip()
        if text_sha256(text) != e["prompt_sha256"]:
            raise HeldoutError(f"{p}: text of {e['id']} does not match its sha256")
        attacks.append(HeldoutAttack(e["id"], e["family"], text, int(e["source_row"])))
    digest = content_hash(name, int(version), str(transform), attacks)
    if digest != data.get("set_sha256"):
        raise HeldoutError(f"{p}: rebuilt set hash differs from set_sha256")
    return HeldoutSet(
        name,
        int(version),
        str(transform),
        tuple(attacks),
        dict(src),
        dict(data.get("filters") or {}),
        digest,
    )
