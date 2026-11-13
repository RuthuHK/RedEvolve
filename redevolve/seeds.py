"""Loading, validation and content hashing of the seed attack set (data/seeds.yaml)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

_ID_RE = re.compile(r"^[a-z]+-\d{2}$")
_CANARY_VALUE_RE = re.compile(r"CANARY-[0-9a-f]{4,}", re.I)


class SeedError(ValueError):
    """The seed file is malformed or violates a rule."""


@dataclass(frozen=True)
class Seed:
    """One hand-written attack prompt."""

    id: str
    family: str
    prompt: str
    note: str = ""


@dataclass(frozen=True)
class SeedSet:
    """A validated seed set and its content hash.

    Attributes:
        version: Seed-set version from the file.
        families: Family name -> description.
        seeds: The seeds, in file order.
        sha256: Hash of the canonical content (ids, families, prompts and the version),
            independent of YAML formatting, comments and line endings. Recorded in
            every run's metadata.
    """

    version: int
    families: dict[str, str]
    seeds: tuple[Seed, ...]
    sha256: str

    def by_family(self) -> dict[str, list[Seed]]:
        """Seeds grouped by family, in family-declaration order."""
        return {f: [s for s in self.seeds if s.family == f] for f in self.families}


def content_hash(version: int, seeds: list[Seed]) -> str:
    """SHA-256 over the parts of the seed set that affect experiments (notes excluded)."""
    canonical = {
        "version": version,
        "seeds": [{"id": s.id, "family": s.family, "prompt": s.prompt} for s in seeds],
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_seeds(path: str | Path) -> SeedSet:
    """Load and validate a seed file.

    Rules: a ``version`` integer; a ``families`` mapping; every seed has a unique id of
    the form ``abc-01``, a declared family and a non-empty prompt; every declared family
    has at least one seed; no prompt contains a canary-like value (the attacker doesn't
    know it).

    Raises:
        SeedError: On any violation, with every problem listed.
    """
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise SeedError(f"cannot read seed file {p}: {e}") from e
    if not isinstance(data, dict):
        raise SeedError(f"{p}: top level must be a mapping")

    problems: list[str] = []
    version = data.get("version")
    if not isinstance(version, int):
        problems.append("'version' must be an integer")
    families = data.get("families")
    if not isinstance(families, dict) or not families:
        raise SeedError(f"{p}: 'families' must be a non-empty mapping")
    raw = data.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise SeedError(f"{p}: 'seeds' must be a non-empty list")

    seeds: list[Seed] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"seed #{i}: must be a mapping")
            continue
        sid, fam, prompt = item.get("id"), item.get("family"), item.get("prompt")
        label = sid or f"#{i}"
        if not isinstance(sid, str) or not _ID_RE.match(sid):
            problems.append(f"seed {label}: id must look like 'io-01'")
        if fam not in families:
            problems.append(f"seed {label}: unknown family {fam!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            problems.append(f"seed {label}: empty prompt")
        elif _CANARY_VALUE_RE.search(prompt):
            problems.append(f"seed {label}: prompt contains a canary-like value")
        seeds.append(Seed(str(sid), str(fam), (prompt or "").strip(), str(item.get("note", ""))))

    dupes = [k for k, n in Counter(s.id for s in seeds).items() if n > 1]
    if dupes:
        problems.append(f"duplicate ids: {sorted(dupes)}")
    empty = [f for f in families if not any(s.family == f for s in seeds)]
    if empty:
        problems.append(f"families with no seeds: {empty}")
    if problems:
        raise SeedError(f"{p}: " + "; ".join(problems))

    return SeedSet(
        version=version,
        families={str(k): str(v) for k, v in families.items()},
        seeds=tuple(seeds),
        sha256=content_hash(version, seeds),
    )
