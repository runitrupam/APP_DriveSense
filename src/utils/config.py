"""Configuration loading with dotted access and stable hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is malformed."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


class Config:
    """Immutable view over a resolved configuration mapping."""

    def __init__(self, data: Mapping[str, Any], source: str | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self._flat: dict[str, Any] = flatten(self._data)
        self.source = source

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "Config":
        """Load YAML from ``path`` (default: repo config) and apply overrides."""
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise ConfigError(f"config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"config root must be a mapping: {config_path}")
        if overrides:
            data = deep_merge(data, overrides)
        return cls(data, source=str(config_path))

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
    ) -> "Config":
        merged = deep_merge(data, overrides) if overrides else data
        return cls(merged)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return the value at ``dotted_key`` or ``default``."""
        return self._flat.get(dotted_key, default)

    def require(self, dotted_key: str) -> Any:
        """Return the value at ``dotted_key`` or raise."""
        if dotted_key not in self._flat:
            raise ConfigError(f"missing required config key: {dotted_key}")
        return self._flat[dotted_key]

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level section as a plain dict."""
        value = self._data.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section is not a mapping: {name}")
        return copy.deepcopy(dict(value))

    def as_dict(self, redact_keys: tuple[str, ...] = ()) -> dict[str, Any]:
        """Return the full config, optionally removing sensitive keys."""
        data = copy.deepcopy(self._data)
        if not redact_keys:
            return data
        for key in list(flatten(data)):
            if any(marker in key.lower() for marker in redact_keys):
                target = data
                parts = key.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target.pop(parts[-1], None)
        return data

    @property
    def flat(self) -> dict[str, Any]:
        """Flattened dotted-key view."""
        return dict(self._flat)

    def hash(self) -> str:
        """Stable SHA-256 of the resolved configuration."""
        payload = json.dumps(self._data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
