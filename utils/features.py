"""Shared Burgers feature helpers used by data generation and rollout."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


CANONICAL_PHYSICS_FEATURE_NAMES = [
    "u_x",
    "abs_u_x",
    "abs_u_xx",
    "grad_var",
    "dt",
    "sin_t",
    "cos_t",
]
LEGACY_PHYSICS_FEATURE_NAMES = [
    "u_x",
    "abs_u_x",
    "dt",
]
FEATURE_LAYOUT_VERSION = "burgers_1d_phys_v2"


def physics_feature_names(phys_dim: int) -> List[str]:
    """Return the canonical feature names for the requested physics width."""
    phys_dim = int(phys_dim)
    if phys_dim == len(CANONICAL_PHYSICS_FEATURE_NAMES):
        return list(CANONICAL_PHYSICS_FEATURE_NAMES)
    if phys_dim == len(LEGACY_PHYSICS_FEATURE_NAMES):
        return list(LEGACY_PHYSICS_FEATURE_NAMES)
    return [f"phys_{idx}" for idx in range(phys_dim)]


def require_integer_refinement(n_fine: int, n_coarse: int) -> int:
    """Ensure the fine grid can be cleanly downsampled onto the coarse grid."""
    n_fine = int(n_fine)
    n_coarse = int(n_coarse)
    if n_fine % n_coarse != 0:
        raise ValueError(
            f"N_fine ({n_fine}) must be an integer multiple of N_coarse ({n_coarse})."
        )
    return n_fine // n_coarse


def downsample_periodic(u_fine: np.ndarray, *, n_coarse: int | None = None, ratio: int | None = None) -> np.ndarray:
    """Periodic stride downsampling used by the reference/coarse pairing."""
    u_fine = np.asarray(u_fine)
    if ratio is None:
        if n_coarse is None:
            raise ValueError("Either n_coarse or ratio must be provided for downsampling.")
        ratio = require_integer_refinement(u_fine.shape[0], n_coarse)
    ratio = int(ratio)
    return u_fine[::ratio].copy()


def compute_physics_features(
    u: np.ndarray,
    dx: float,
    dt: float,
    t_stamp: float,
    phys_dim: int | None = None,
) -> np.ndarray:
    """Build local physics descriptors for Burgers' equation."""
    u = np.asarray(u)
    u_x = np.gradient(u, dx)
    abs_u_x = np.abs(u_x)
    u_xx = np.gradient(u_x, dx)
    abs_u_xx = np.abs(u_xx)
    grad_var = np.sqrt(((np.roll(u_x, -1) - u_x) ** 2 + (np.roll(u_x, 1) - u_x) ** 2) * 0.5)
    dt_feat = np.full_like(u_x, dt)
    sin_t = np.full_like(u_x, np.sin(t_stamp))
    cos_t = np.full_like(u_x, np.cos(t_stamp))

    features: List[np.ndarray] = [
        u_x,
        abs_u_x,
        abs_u_xx,
        grad_var,
        dt_feat,
        sin_t,
        cos_t,
    ]

    if phys_dim is None:
        phys_dim = len(CANONICAL_PHYSICS_FEATURE_NAMES)
    phys_dim = int(phys_dim)

    if phys_dim == len(CANONICAL_PHYSICS_FEATURE_NAMES):
        return np.stack(features, axis=1)
    if phys_dim == len(LEGACY_PHYSICS_FEATURE_NAMES):
        return np.stack([u_x, abs_u_x, dt_feat], axis=1)

    if phys_dim <= len(features):
        return np.stack(features[:phys_dim], axis=1)

    padded = list(features)
    while len(padded) < phys_dim:
        padded.append(np.zeros_like(u_x))
    return np.stack(padded[:phys_dim], axis=1)


def build_model_inputs(
    u: np.ndarray,
    *,
    stencil_size: int,
    dx: float,
    dt: float,
    t_stamp: float,
    phys_dim: int | None = None,
) -> np.ndarray:
    """Compose periodic stencils and physics features into the model input block."""
    pad_size = int(stencil_size) // 2
    u_padded = np.pad(np.asarray(u), (pad_size, pad_size), mode="wrap")
    stencils = sliding_window_view(u_padded, window_shape=int(stencil_size))
    phys_feats = compute_physics_features(u, dx, dt, t_stamp, phys_dim=phys_dim)
    return np.hstack([stencils, phys_feats])


def ensure_feature_names(value: Iterable[str] | None, phys_dim: int) -> List[str]:
    if value is None:
        return physics_feature_names(phys_dim)
    return [str(item) for item in value]
