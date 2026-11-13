"""Persistent response cache (SQLite) for LLM calls.

Used for the Groq mutator: re-running or resuming an experiment with the same seed
replays identical mutations instead of spending rate-limited quota, and makes the
run reproducible even though the hosted model is not bit-deterministic.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    """Stable SHA-256 key for a JSON-serialisable request description."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Key -> JSON value store backed by a single SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached value for ``key``, or None."""
        row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store (or overwrite) ``value`` under ``key``."""
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, value, created) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()
