"""Metadata helpers for dataset/checkpoint compatibility checks."""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict

import numpy as np

from utils.features import (
    FEATURE_LAYOUT_VERSION,
    LEGACY_FEATURE_LAYOUT_VERSIONS,
    build_feature_layout_metadata,
    ensure_feature_names,
)


ONE_STEP_TARGET_SEMANTICS = "one_step_correction"
ONE_STEP_TARGET_DEFINITION = "u_ref_next - u_weno_next"
TARGET_SPACE_PHYSICAL_CORRECTION = "physical_correction"
TARGET_SCALER_TYPE_AFFINE = "affine"
REFERENCE_MODE_PAIRED_FINE = "paired_fine_rollout_stride_downsample"
DEFAULT_CORRECTION_HEAD_MODE = "linear"
DEFAULT_CORRECTION_HEAD_SCALE = 1.0
LEGACY_CORRECTION_HEAD_MODE = "softsign_scaled"
LEGACY_CORRECTION_HEAD_SCALE = 0.5


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def serialize_metadata(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True)


def deserialize_metadata(raw_value: Any) -> Dict[str, Any] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value

    if isinstance(raw_value, np.ndarray):
        if raw_value.shape == ():
            raw_value = raw_value.item()
        else:
            raw_value = raw_value.tolist()

    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")

    if isinstance(raw_value, str):
        return json.loads(raw_value)

    raise ValueError(f"Unsupported metadata payload type: {type(raw_value)!r}")


def build_dataset_metadata(
    *,
    steps_ahead: int,
    stencil_size: int,
    phys_dim: int,
    physics_feature_names: list[str],
    solver_cfg: Dict[str, Any],
    generation_cfg: Dict[str, Any],
    feature_layout: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "metadata_version": 3,
        "source_kind": "dataset",
        "target_semantics": ONE_STEP_TARGET_SEMANTICS,
        "target_definition": ONE_STEP_TARGET_DEFINITION,
        "target_space": TARGET_SPACE_PHYSICAL_CORRECTION,
        "steps_ahead": int(steps_ahead),
        "stencil_size": int(stencil_size),
        "phys_dim": int(phys_dim),
        "physics_feature_names": list(physics_feature_names),
        "feature_layout_version": FEATURE_LAYOUT_VERSION,
        "feature_layout": feature_layout
        or build_feature_layout_metadata(
            stencil_size=stencil_size,
            phys_dim=phys_dim,
            physics_feature_names=physics_feature_names,
            use_stencil_features=True,
            stencil_radius=int(stencil_size) // 2,
            gate_use_stencil_features=False,
        ),
        "reference_mode": REFERENCE_MODE_PAIRED_FINE,
        "solver": {
            "nu": float(solver_cfg.get("nu", 0.0)),
            "weno_epsilon": float(solver_cfg.get("weno_epsilon", 1e-6)),
        },
        "data_generation": {
            "N_coarse": int(generation_cfg.get("N_coarse")),
            "N_fine": int(generation_cfg.get("N_fine")),
            "cfl": float(generation_cfg.get("cfl")),
            "sample_stride": int(generation_cfg.get("sample_stride", 1)),
            "num_sessions": int(generation_cfg.get("num_sessions", 1)),
        },
    }


def build_checkpoint_metadata(
    *,
    dataset_metadata: Dict[str, Any],
    solver_cfg: Dict[str, Any],
    target_scaler_cfg: Dict[str, Any],
    target_scaler_enabled: bool,
    correction_head_cfg: Dict[str, Any] | None = None,
    feature_layout: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(dataset_metadata)
    metadata.update(
        {
            "metadata_version": 3,
            "source_kind": "checkpoint",
            "solver": {
                "nu": float(solver_cfg.get("nu", 0.0)),
                "weno_epsilon": float(solver_cfg.get("weno_epsilon", 1e-6)),
            },
            "target_scaling": {
                "target_space": TARGET_SPACE_PHYSICAL_CORRECTION,
                "scaler_type": TARGET_SCALER_TYPE_AFFINE,
                "state_required": bool(target_scaler_enabled),
                "clip_z": target_scaler_cfg.get("clip_z", None),
                "min_std": float(target_scaler_cfg.get("min_std", 1e-8)),
            },
            "correction_head": correction_head_cfg
            or {
                "output_mode": DEFAULT_CORRECTION_HEAD_MODE,
                "output_scale": DEFAULT_CORRECTION_HEAD_SCALE,
            },
            "feature_layout": feature_layout or dataset_metadata.get("feature_layout"),
            "default_correction_sign": "plus",
            "dataset_metadata": dataset_metadata,
        }
    )
    return metadata


def extract_dataset_metadata(npz_data: Any, *, source_label: str, warn_on_missing: bool = True) -> Dict[str, Any]:
    raw_metadata = None
    if hasattr(npz_data, "files") and "metadata_json" in npz_data.files:
        raw_metadata = npz_data["metadata_json"]

    metadata = deserialize_metadata(raw_metadata) if raw_metadata is not None else {}
    if raw_metadata is None and warn_on_missing:
        _warn(f"{source_label}: metadata_json missing, falling back to legacy scalar fields.")

    steps_ahead = _extract_scalar(npz_data, "steps_ahead")
    stencil_size = _extract_scalar(npz_data, "stencil_size")
    phys_dim = _extract_scalar(npz_data, "phys_dim")

    if "steps_ahead" not in metadata and steps_ahead is not None:
        metadata["steps_ahead"] = steps_ahead
    if "stencil_size" not in metadata and stencil_size is not None:
        metadata["stencil_size"] = stencil_size
    if "phys_dim" not in metadata and phys_dim is not None:
        metadata["phys_dim"] = phys_dim

    if "physics_feature_names" not in metadata and metadata.get("phys_dim") is not None:
        if warn_on_missing:
            _warn(f"{source_label}: physics_feature_names missing, using canonical fallback by phys_dim.")
        metadata["physics_feature_names"] = ensure_feature_names(None, int(metadata["phys_dim"]))

    if "feature_layout_version" not in metadata and metadata.get("phys_dim") is not None:
        metadata["feature_layout_version"] = FEATURE_LAYOUT_VERSION

    _ensure_feature_layout_metadata(
        metadata,
        source_label=source_label,
        warn_on_missing=warn_on_missing,
    )

    return metadata


def extract_checkpoint_metadata(
    checkpoint: Dict[str, Any],
    *,
    source_label: str,
    warn_on_missing: bool = True,
) -> Dict[str, Any]:
    metadata = deserialize_metadata(checkpoint.get("metadata")) or {}
    if "metadata" not in checkpoint and warn_on_missing:
        _warn(f"{source_label}: checkpoint metadata missing, falling back to legacy scalar fields.")

    for key in ("steps_ahead", "stencil_size", "phys_dim"):
        if key not in metadata and key in checkpoint:
            metadata[key] = _coerce_scalar(checkpoint[key])

    if "physics_feature_names" not in metadata and metadata.get("phys_dim") is not None:
        if warn_on_missing:
            _warn(f"{source_label}: physics_feature_names missing, using canonical fallback by phys_dim.")
        metadata["physics_feature_names"] = ensure_feature_names(None, int(metadata["phys_dim"]))

    if "feature_layout_version" not in metadata and metadata.get("phys_dim") is not None:
        metadata["feature_layout_version"] = FEATURE_LAYOUT_VERSION

    _ensure_feature_layout_metadata(
        metadata,
        source_label=source_label,
        warn_on_missing=warn_on_missing,
    )
    _ensure_correction_head_metadata(
        metadata,
        checkpoint=checkpoint,
        source_label=source_label,
        warn_on_missing=warn_on_missing,
    )

    return metadata


def validate_one_step_metadata(
    metadata: Dict[str, Any],
    *,
    source_label: str,
    expected_steps_ahead: int = 1,
    expected_stencil_size: int | None = None,
    expected_phys_dim: int | None = None,
    expected_feature_names: list[str] | None = None,
    expected_target_scaling: Dict[str, Any] | None = None,
    expected_feature_layout: Dict[str, Any] | None = None,
    expected_correction_head: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    mismatches: list[str] = []

    def _check(key: str, expected: Any) -> None:
        actual = metadata.get(key)
        if actual is None:
            return
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")

    _check("steps_ahead", int(expected_steps_ahead))
    if expected_stencil_size is not None:
        _check("stencil_size", int(expected_stencil_size))
    if expected_phys_dim is not None:
        _check("phys_dim", int(expected_phys_dim))

    target_semantics = metadata.get("target_semantics")
    if target_semantics is not None and target_semantics != ONE_STEP_TARGET_SEMANTICS:
        mismatches.append(
            f"target_semantics: expected {ONE_STEP_TARGET_SEMANTICS!r}, got {target_semantics!r}"
        )

    target_definition = metadata.get("target_definition")
    if target_definition is not None and target_definition != ONE_STEP_TARGET_DEFINITION:
        mismatches.append(
            f"target_definition: expected {ONE_STEP_TARGET_DEFINITION!r}, got {target_definition!r}"
        )

    target_space = metadata.get("target_space")
    if target_space is not None and target_space != TARGET_SPACE_PHYSICAL_CORRECTION:
        mismatches.append(
            f"target_space: expected {TARGET_SPACE_PHYSICAL_CORRECTION!r}, got {target_space!r}"
        )

    feature_names = metadata.get("physics_feature_names")
    if expected_feature_names is not None and feature_names is not None:
        if list(feature_names) != list(expected_feature_names):
            mismatches.append(
                f"physics_feature_names: expected {expected_feature_names!r}, got {feature_names!r}"
            )

    feature_layout_version = metadata.get("feature_layout_version")
    if (
        feature_layout_version is not None
        and feature_layout_version != FEATURE_LAYOUT_VERSION
        and feature_layout_version not in LEGACY_FEATURE_LAYOUT_VERSIONS
    ):
        mismatches.append(
            f"feature_layout_version: expected {FEATURE_LAYOUT_VERSION!r}, got {feature_layout_version!r}"
        )

    if expected_feature_layout is not None:
        feature_layout = metadata.get("feature_layout")
        if feature_layout is None:
            mismatches.append("feature_layout: missing dataset/checkpoint feature layout metadata")
        else:
            for key, expected in expected_feature_layout.items():
                actual = feature_layout.get(key)
                if actual is not None and actual != expected:
                    mismatches.append(f"feature_layout.{key}: expected {expected!r}, got {actual!r}")

    if expected_target_scaling is not None:
        scaling = metadata.get("target_scaling")
        if scaling is None:
            mismatches.append("target_scaling: missing checkpoint scaling metadata")
        else:
            for key, expected in expected_target_scaling.items():
                actual = scaling.get(key)
                if actual != expected:
                    mismatches.append(f"target_scaling.{key}: expected {expected!r}, got {actual!r}")

    if expected_correction_head is not None:
        correction_head = metadata.get("correction_head")
        if correction_head is None:
            mismatches.append("correction_head: missing checkpoint correction_head metadata")
        else:
            for key, expected in expected_correction_head.items():
                actual = correction_head.get(key)
                if actual is not None and actual != expected:
                    mismatches.append(f"correction_head.{key}: expected {expected!r}, got {actual!r}")

    if mismatches:
        raise ValueError(
            f"{source_label} metadata mismatch for strict one-step correction:\n- "
            + "\n- ".join(mismatches)
        )

    return metadata


def default_target_scaling_metadata(*, target_scaler_enabled: bool) -> Dict[str, Any]:
    return {
        "target_space": TARGET_SPACE_PHYSICAL_CORRECTION,
        "scaler_type": TARGET_SCALER_TYPE_AFFINE,
        "state_required": bool(target_scaler_enabled),
    }


def default_correction_head_metadata(
    *,
    output_mode: str = DEFAULT_CORRECTION_HEAD_MODE,
    output_scale: float = DEFAULT_CORRECTION_HEAD_SCALE,
) -> Dict[str, Any]:
    return {
        "output_mode": str(output_mode).lower(),
        "output_scale": float(output_scale),
    }


def infer_default_sign(metadata: Dict[str, Any]) -> str:
    if metadata.get("target_definition") == ONE_STEP_TARGET_DEFINITION:
        return "plus"
    return str(metadata.get("default_correction_sign", "plus")).lower()


def _extract_scalar(npz_data: Any, key: str) -> Any:
    if not hasattr(npz_data, "files") or key not in npz_data.files:
        return None
    return _coerce_scalar(npz_data[key])


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _ensure_feature_layout_metadata(
    metadata: Dict[str, Any],
    *,
    source_label: str,
    warn_on_missing: bool,
) -> None:
    stencil_size = metadata.get("stencil_size")
    phys_dim = metadata.get("phys_dim")
    if stencil_size is None or phys_dim is None:
        return

    if "feature_layout" not in metadata:
        if warn_on_missing:
            _warn(
                f"{source_label}: feature_layout missing, inferring legacy full-stencil correction layout."
            )
        metadata["feature_layout_inferred"] = True
        metadata["feature_layout"] = build_feature_layout_metadata(
            stencil_size=int(stencil_size),
            phys_dim=int(phys_dim),
            physics_feature_names=metadata.get("physics_feature_names"),
            use_stencil_features=True,
            stencil_radius=int(stencil_size) // 2,
            gate_use_stencil_features=False,
        )
    else:
        metadata.setdefault("feature_layout_inferred", False)
    metadata["feature_layout"].setdefault("input_layout", "full_stencil_plus_physics")
    metadata["feature_layout"].setdefault("stencil_size", int(stencil_size))
    metadata["feature_layout"].setdefault("phys_dim", int(phys_dim))
    metadata["feature_layout"].setdefault(
        "physics_feature_names",
        ensure_feature_names(metadata.get("physics_feature_names"), int(phys_dim)),
    )


def _ensure_correction_head_metadata(
    metadata: Dict[str, Any],
    *,
    checkpoint: Dict[str, Any],
    source_label: str,
    warn_on_missing: bool,
) -> None:
    if "correction_head" in metadata:
        metadata["correction_head"].setdefault("output_mode", DEFAULT_CORRECTION_HEAD_MODE)
        metadata["correction_head"].setdefault("output_scale", DEFAULT_CORRECTION_HEAD_SCALE)
        metadata.setdefault("correction_head_inferred", False)
        return

    model_cfg = checkpoint.get("config", {}).get("model", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    if isinstance(model_cfg.get("correction_head"), dict):
        metadata["correction_head"] = {
            "output_mode": str(model_cfg["correction_head"].get("output_mode", DEFAULT_CORRECTION_HEAD_MODE)).lower(),
            "output_scale": float(model_cfg["correction_head"].get("output_scale", DEFAULT_CORRECTION_HEAD_SCALE)),
        }
        metadata["correction_head_inferred"] = True
        return

    if "shape_output_scale" in model_cfg:
        if warn_on_missing:
            _warn(
                f"{source_label}: correction_head missing, inferring legacy softsign head from model.shape_output_scale."
            )
        metadata["correction_head"] = {
            "output_mode": LEGACY_CORRECTION_HEAD_MODE,
            "output_scale": float(model_cfg.get("shape_output_scale", LEGACY_CORRECTION_HEAD_SCALE)),
        }
        metadata["correction_head_inferred"] = True
        return

    if warn_on_missing:
        _warn(f"{source_label}: correction_head missing, using default linear head metadata.")
    metadata["correction_head"] = default_correction_head_metadata()
    metadata["correction_head_inferred"] = True
