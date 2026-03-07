"""Rollout entrypoint kept for backward compatibility.

The rollout logic lives in evaluate.py.
"""

import argparse

from evaluate import run_evaluation
from utils.config import load_config


def _parse_args():
    parser = argparse.ArgumentParser(description="Rollout entrypoint (config-driven).")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选）")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    run_evaluation(cfg)
