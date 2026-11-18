"""Loading, validation and content hashing of the benign prompt set (data/benign.yaml)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

_ID_RE = re.compile(r"^bn-\d{2,3}$")


class BenignError(ValueError):
    """The benign file is malformed."""


@dataclass(frozen=True)
class BenignPrompt:
    """One benign user message."""

    id: str
    category: str
    prompt: str


@dataclass(frozen=True)
class BenignSet:
    """A validated benign set; ``sha256`` covers version, ids, categories and prompts."""

    version: int
    categories: dict[str, str]
    prompts: tuple[BenignPrompt, ...]
    sha256: str


def content_hash(version: int, prompts: list[BenignPrompt]) -> str:
    """SHA-256 of the canonical content (independent of YAML formatting and comments)."""
    canonical = {
        "version": version,
        "prompts": [{"id": p.id, "category": p.category, "prompt": p.prompt} for p in prompts],
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_benign(path: str | Path) -> BenignSet:
    """Load and validate a benign file.

    Raises:
        BenignError: On any problem (all listed): bad version, unknown category, bad or
            duplicate id, empty prompt.
    """
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise BenignError(f"cannot read benign file {p}: {e}") from e
    if not isinstance(data, dict):
        raise BenignError(f"{p}: top level must be a mapping")
    version, categories, raw = data.get("version"), data.get("categories"), data.get("prompts")
    if not isinstance(categories, dict) or not categories:
        raise BenignError(f"{p}: 'categories' must be a non-empty mapping")
    if not isinstance(raw, list) or not raw:
        raise BenignError(f"{p}: 'prompts' must be a non-empty list")
    problems = [] if isinstance(version, int) else ["'version' must be an integer"]
    prompts: list[BenignPrompt] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"prompt #{i}: must be a mapping")
            continue
        pid, cat, text = item.get("id"), item.get("category"), item.get("prompt")
        label = pid or f"#{i}"
        if not isinstance(pid, str) or not _ID_RE.match(pid):
            problems.append(f"prompt {label}: id must look like 'bn-01'")
        if cat not in categories:
            problems.append(f"prompt {label}: unknown category {cat!r}")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"prompt {label}: empty prompt")
        prompts.append(BenignPrompt(str(pid), str(cat), (text or "").strip()))
    dupes = sorted(k for k, n in Counter(x.id for x in prompts).items() if n > 1)
    if dupes:
        problems.append(f"duplicate ids: {dupes}")
    if problems:
        raise BenignError(f"{p}: " + "; ".join(problems))
    return BenignSet(
        version=version,
        categories={str(k): str(v) for k, v in categories.items()},
        prompts=tuple(prompts),
        sha256=content_hash(version, prompts),
    )
