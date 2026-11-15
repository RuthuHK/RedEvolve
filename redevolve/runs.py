"""Run directories: metadata, append-only JSONL logs, and safe resume.

Layout of ``runs/<run_id>/``::

    run.json       metadata: fingerprint, status, environment, canary, extra info
    config.yaml    the exact resolved config used
    log.jsonl      one JSON record per line (evaluations, mutations, generation summaries)

The log is append-only, and every line is flushed and fsynced, so a crash (or Ctrl-C, or a
reboot) loses at most the line being written. A torn last line is detected and dropped
on resume. Work is identified by a ``key`` per record, so a resumed run skips completed keys.

Resume is refused if anything that defines the experiment differs from the original run:
config, seed set, system prompt, target model, mutator model, judge version or canary.
This enforces "one run = one mutator model" and the frozen seed set.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__, judge
from .seeds import SeedSet
from .target import system_prompt_hash

RUN_FILE = "run.json"
LOG_FILE = "log.jsonl"
CONFIG_FILE = "config.yaml"

# Defence in depth: refuse to write anything that looks like an API key into a run directory.
_SECRET_RE = re.compile(r"gsk_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,}")


class RunError(RuntimeError):
    """A run can't be created or resumed (e.g. fingerprint mismatch)."""


def _canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprint(cfg: dict[str, Any], seeds: SeedSet, canary: str) -> dict[str, Any]:
    """Everything that must be identical for two sessions to count as the same run."""
    return {
        "config_sha256": _canonical_hash(cfg),
        "seeds_sha256": seeds.sha256,
        "seeds_version": seeds.version,
        "system_prompt_sha256": system_prompt_hash(cfg["target"]["system_prompt"]),
        "target_model": cfg["target"]["model"],
        "mutator_model": cfg["mutator"]["model"],
        "judge_version": judge.JUDGE_VERSION,
        "canary": canary,
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def environment_info() -> dict[str, Any]:
    """Code version and platform, recorded for reproducibility (no env vars, no secrets)."""
    status = _git("status", "--porcelain")
    return {
        "redevolve_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _now() -> datetime:
    return datetime.now(UTC)


def _check_no_secrets(text: str, where: str) -> None:
    if _SECRET_RE.search(text):
        raise RunError(f"refusing to write {where}: content looks like it contains an API key")


class RunDir:
    """A run directory. Create with :meth:`create`, continue with :meth:`resume`."""

    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        self.path = path
        self.meta = meta

    @property
    def run_id(self) -> str:
        """The run's id (also its directory name)."""
        return self.meta["run_id"]

    @property
    def log_path(self) -> Path:
        return self.path / LOG_FILE

    # --- creation / resume -------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str | Path,
        kind: str,
        cfg: dict[str, Any],
        *,
        seeds: SeedSet,
        canary: str,
        extra: dict[str, Any] | None = None,
        now: Callable[[], datetime] = _now,
        env: Callable[[], dict[str, Any]] = environment_info,
    ) -> RunDir:
        """Create ``root/<kind>-<UTC timestamp>-s<seed>/`` and write its metadata and config."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        created = now()
        base = f"{kind}-{created:%Y%m%d-%H%M%S}-s{cfg['seed']}"
        run_id, n = base, 2
        while (root / run_id).exists():
            run_id, n = f"{base}-{n}", n + 1
        path = root / run_id
        meta = {
            "run_id": run_id,
            "kind": kind,
            "seed": cfg["seed"],
            "created": created.isoformat(),
            "status": "running",
            "sessions": [{"started": created.isoformat()}],
            "fingerprint": fingerprint(cfg, seeds, canary),
            "environment": env(),
            **(extra or {}),
        }
        config_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        _check_no_secrets(config_text, CONFIG_FILE)
        path.mkdir()
        (path / CONFIG_FILE).write_text(config_text, encoding="utf-8")
        run = cls(path, meta)
        run._write_meta()
        run.log_path.touch()
        return run

    @classmethod
    def resume(
        cls,
        root: str | Path,
        run_id: str,
        cfg: dict[str, Any],
        *,
        seeds: SeedSet,
        canary: str,
        now: Callable[[], datetime] = _now,
    ) -> RunDir:
        """Reopen an existing run after checking its fingerprint against the current setup.

        Raises:
            RunError: If the run doesn't exist, is already complete, or anything in its
                fingerprint differs (every mismatch is listed).
        """
        path = Path(root) / run_id
        if not (path / RUN_FILE).is_file():
            raise RunError(f"no run named {run_id!r} in {root}")
        meta = json.loads((path / RUN_FILE).read_text(encoding="utf-8"))
        if meta.get("status") == "complete":
            raise RunError(f"run {run_id} is already complete")
        expected = meta["fingerprint"]
        actual = fingerprint(cfg, seeds, canary)
        diffs = [k for k in expected if expected[k] != actual.get(k)]
        if diffs:
            detail = ", ".join(f"{k}: run={expected[k]!r} now={actual.get(k)!r}" for k in diffs)
            raise RunError(f"cannot resume {run_id}: setup differs ({detail})")
        run = cls(path, meta)
        run._repair_log()
        meta["status"] = "running"
        meta.setdefault("sessions", []).append({"started": now().isoformat(), "resumed": True})
        run._write_meta()
        return run

    @classmethod
    def open(cls, root: str | Path, run_id: str) -> RunDir:
        """Open an existing run read-only (for summaries and labelling; no fingerprint check).

        Raises:
            RunError: If the run doesn't exist.
        """
        path = Path(root) / run_id
        if not (path / RUN_FILE).is_file():
            raise RunError(f"no run named {run_id!r} in {root}")
        return cls(path, json.loads((path / RUN_FILE).read_text(encoding="utf-8")))

    @property
    def canary(self) -> str:
        """The run's canary (from its fingerprint)."""
        return self.meta["fingerprint"]["canary"]

    # --- metadata -----------------------------------------------------------------

    def _write_meta(self) -> None:
        text = json.dumps(self.meta, indent=2, ensure_ascii=False)
        _check_no_secrets(text, RUN_FILE)
        tmp = self.path / (RUN_FILE + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path / RUN_FILE)  # atomic: run.json is never half-written

    def set_status(self, status: str, **info: Any) -> None:
        """Update the run status (``running``/``paused``/``failed``/``complete``) and info."""
        self.meta["status"] = status
        self.meta.setdefault("sessions", [{}])[-1].update(
            {"ended": _now().isoformat(), "status": status}
        )
        self.meta.update(info)
        self._write_meta()

    # --- log ------------------------------------------------------------------------

    def append(self, record: dict[str, Any]) -> None:
        """Append one record to the log, durably (flush + fsync)."""
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        _check_no_secrets(line, LOG_FILE)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def records(self, type_: str | None = None) -> Iterator[dict[str, Any]]:
        """Iterate over the log's records (optionally only one ``type``); skips a torn last line."""
        if not self.log_path.exists():
            return
        with self.log_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    return  # torn final write from a crash
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if type_ is None or rec.get("type") == type_:
                    yield rec

    def completed_keys(self, type_: str | None = None) -> set[str]:
        """Keys of records already in the log (the work a resumed run skips)."""
        return {r["key"] for r in self.records(type_) if "key" in r}

    def _repair_log(self) -> None:
        """Drop a torn final line so new appends start on a clean line."""
        if not self.log_path.exists():
            return
        data = self.log_path.read_bytes()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1  # 0 if there's no complete line at all
            with self.log_path.open("r+b") as f:
                f.truncate(cut)
