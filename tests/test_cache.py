"""Tests for the SQLite response cache."""

from pathlib import Path

from redevolve.cache import ResponseCache, cache_key


def test_key_is_order_independent_and_sensitive_to_values() -> None:
    assert cache_key({"a": 1, "b": [1, 2]}) == cache_key({"b": [1, 2], "a": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})
    assert cache_key({"seed": 1}) != cache_key({"seed": None})


def test_put_get_roundtrip_and_counters(tmp_path: Path) -> None:
    c = ResponseCache(tmp_path / "c.sqlite")
    assert c.get("k") is None
    c.put("k", {"text": "héllo", "n": 3})
    assert c.get("k") == {"text": "héllo", "n": 3}
    assert (c.hits, c.misses, len(c)) == (1, 1, 1)


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "c.sqlite"  # parent dir is created
    c1 = ResponseCache(path)
    c1.put("k", {"v": 1})
    c1.close()
    assert ResponseCache(path).get("k") == {"v": 1}


def test_overwrite(tmp_path: Path) -> None:
    c = ResponseCache(tmp_path / "c.sqlite")
    c.put("k", {"v": 1})
    c.put("k", {"v": 2})
    assert c.get("k") == {"v": 2}
    assert len(c) == 1
