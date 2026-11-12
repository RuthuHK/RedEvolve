"""Tests for config loading and the CLI's handling of config errors."""

from pathlib import Path

import pytest

from redevolve.cli import main
from redevolve.config import ConfigError, load_config

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def test_loads_explicit_path(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("seed: 7\ntarget: {model: m}\n", encoding="utf-8")
    assert load_config(cfg_file) == {"seed": 7, "target": {"model": "m"}}


def test_default_is_cwd_not_package_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_config() == {"seed": 1}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("body", ["- just\n- a list\n", "", "key: [unclosed\n"])
def test_non_mapping_or_invalid_yaml_raises(tmp_path: Path, body: str) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_file)


def test_repo_config_has_required_sections() -> None:
    cfg = load_config(REPO_CONFIG)
    assert isinstance(cfg["seed"], int)
    for section in ("target", "mutator", "classifier", "ga", "defend"):
        assert section in cfg, section
    assert cfg["target"]["provider"] == "ollama"


def test_cli_missing_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", str(tmp_path / "missing.yaml"), "smoke"])
    assert code == 2
    assert "config file not found" in capsys.readouterr().err
