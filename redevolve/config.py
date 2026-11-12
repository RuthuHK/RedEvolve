"""Loading of the experiment configuration (config.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config.yaml")


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config.

    Args:
        path: Path to the config file. Defaults to ``config.yaml`` in the current
            working directory (not the package directory), so the CLI behaves the
            same whether the package is installed or run from a checkout.

    Returns:
        The parsed configuration as a dict.

    Raises:
        ConfigError: If the file does not exist or does not contain a YAML mapping.
    """
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.is_file():
        raise ConfigError(f"config file not found: {p.resolve()} (pass --config PATH)")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level")
    return data
