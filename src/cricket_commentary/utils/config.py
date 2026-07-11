"""YAML config loading with single-level inheritance and dotted overrides.

Every experiment is driven by a YAML file in configs/ (§2 rule 3). Ablation
cells stay one-file-per-cell without duplication via an `extends:` key naming
a base config (path relative to the child file). CLI overrides use dotted
paths: ``--set decoding.temperature=0.7``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    parent = cfg.pop("extends", None)
    if parent is not None:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    cfg.setdefault("_config_path", str(path))
    return cfg


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.sub=value`` strings; values are parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key.sub=value, got: {item}")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = _parse_scalar(raw)
    return cfg


def require(cfg: dict, dotted: str) -> Any:
    """Fetch a required key, failing with the config path in the message."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(
                f"missing required config key '{dotted}' in {cfg.get('_config_path', '<dict>')}"
            )
        node = node[key]
    return node
