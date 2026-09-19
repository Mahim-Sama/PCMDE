"""
Configuration manager.

Loads YAML or JSON configuration files, interpolates environment variables,
and provides dot-notation access to nested values. Also supports merging
of multiple config sources (defaults < file < CLI overrides).

Usage:
    cfg = ConfigManager.load("domains/animal/config.yaml")
    model = cfg.get("llm.model_name", default="deepseek-r1:70b")
    cfg.override({"paths.output_json": "/tmp/out.json"})
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    """Raised for any configuration loading or access error."""


class ConfigManager:
    """
    Domain-agnostic configuration holder.

    Internally stores a nested dict. Access is by dot-notation path
    (e.g., "llm.model_name") with optional defaults.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = data or {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConfigManager":
        """
        Load a YAML or JSON config from disk.

        The file extension (.yaml, .yml, .json) determines the parser.
        Environment variables of the form ${VAR_NAME} are interpolated
        during load.
        """
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")

        raw = p.read_text(encoding="utf-8")
        raw = cls._interpolate_env_vars(raw)

        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ConfigError(
                    "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
                ) from exc
            data = yaml.safe_load(raw) or {}
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            raise ConfigError(
                f"Unsupported config extension '{suffix}' for {p}. Use .yaml, .yml, or .json."
            )

        if not isinstance(data, dict):
            raise ConfigError(f"Top-level config in {p} must be a mapping, got {type(data).__name__}")

        return cls(data)

    @staticmethod
    def _interpolate_env_vars(text: str) -> str:
        """
        Replace ${VAR_NAME} tokens with the corresponding environment variable.
        Missing env vars are left as-is with a warning embedded (so failures
        surface visibly rather than being silently substituted with empty string).
        """
        def replace(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                # Leave the token intact; this surfaces the problem on use.
                return match.group(0)
            return value

        return ENV_VAR_PATTERN.sub(replace, text)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieve a value by dot-notation path.

        Example:
            cfg.get("llm.model_name", default="deepseek-r1:70b")
            cfg.get("paths.output_json")

        Returns default if any segment of the path is missing.
        """
        segments = key_path.split(".")
        node: Any = self._data
        for seg in segments:
            if isinstance(node, dict) and seg in node:
                node = node[seg]
            else:
                return default
        return node

    def require(self, key_path: str) -> Any:
        """
        Like get(), but raises ConfigError if the path is missing or value is None.
        """
        value = self.get(key_path, default=None)
        if value is None:
            raise ConfigError(f"Required config key missing: '{key_path}'")
        return value

    def has(self, key_path: str) -> bool:
        """Return True if the key path exists (even if its value is None)."""
        segments = key_path.split(".")
        node: Any = self._data
        for seg in segments:
            if isinstance(node, dict) and seg in node:
                node = node[seg]
            else:
                return False
        return True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def override(self, overrides: Dict[str, Any]) -> None:
        """
        Apply dot-notation overrides in place. Useful for CLI overrides.

        Example:
            cfg.override({"paths.output_json": "/tmp/out.json", "llm.temperature": 0.2})
        """
        for key_path, value in overrides.items():
            if value is None:
                continue
            self._set(key_path, value)

    def _set(self, key_path: str, value: Any) -> None:
        segments = key_path.split(".")
        node = self._data
        for seg in segments[:-1]:
            if seg not in node or not isinstance(node[seg], dict):
                node[seg] = {}
            node = node[seg]
        node[segments[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the raw config dict for serialization."""
        return dict(self._data)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def require_paths(self, keys):
        """
        Validate that the file paths at the given config keys exist.
        Does not fail for missing *output* paths (their parent must exist though).
        """
        for k in keys:
            p = self.require(k)
            if not Path(p).exists():
                raise ConfigError(f"Path from config key '{k}' does not exist: {p}")
