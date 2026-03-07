"""轻量级 YAML 配置加载与递归合并工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import copy
import random

import numpy as np
import torch

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required for config loading. Please install pyyaml>=6.0."
    ) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return _repo_root() / "config" / "default_config.yaml"


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML config at '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping/object at root: {path}")
    return data


def recursive_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典: override 覆盖 base。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    加载配置：先读 default_config.yaml，再用用户配置递归覆盖。
    """
    default_path = default_config_path()
    base_cfg = _safe_load_yaml(default_path)

    if config_path is None:
        return base_cfg

    user_path = Path(config_path)
    if not user_path.is_absolute():
        user_path = (_repo_root() / user_path).resolve()

    user_cfg = _safe_load_yaml(user_path)
    return recursive_merge(base_cfg, user_cfg)


def cfg_get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for token in dotted_key.split("."):
        if not isinstance(cur, dict) or token not in cur:
            return default
        cur = cur[token]
    return cur


def set_global_seed(seed: int | None, deterministic: bool = False) -> None:
    """设置随机种子，便于实验可复现。"""
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
