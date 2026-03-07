import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

from kan import GatedKAN, HybridScaler, TargetAffineScaler, resolve_model_runtime_config
from solvers.weno import rk3_step
from utils.config import cfg_get, load_config, set_global_seed
from utils.features import build_model_inputs, downsample_periodic, physics_feature_names, require_integer_refinement
from utils.metadata import (
    default_target_scaling_metadata,
    extract_checkpoint_metadata,
    infer_default_sign,
    validate_one_step_metadata,
)


def _normalize_gate_mode(mode: str) -> str:
    mode_l = str(mode).lower()
    if mode_l == "original":
        warnings.warn(
            "gate_mode='original' is deprecated; using 'hard_mask' instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "hard_mask"
    return mode_l


class KANPredictor:
    """
    Predictor with configurable gate policy.

    gate_mode options:
      - train_equivalent: use the exact gated output path seen during training
      - raw_gate_only:    recompute raw_corr * raw_gate
      - soft_mask:        effective_gate = raw_gate * soft_mask
      - hard_mask:        effective_gate = raw_gate * hard_mask
      - gate_open:        effective_gate = 1
    """

    def __init__(
        self,
        model_path,
        device="cpu",
        model_cfg=None,
        scaler_cfg=None,
        gate_mode="train_equivalent",
        hard_gate_sensor_threshold=0.1,
        soft_mask_center=0.1,
        soft_mask_width=0.08,
        soft_mask_floor=0.02,
        correction_clip_abs=None,
        debug_rollout=False,
        debug_t_start=1.0,
        metadata_cfg=None,
    ):
        self.device = device
        self.model_cfg = model_cfg or {}
        self.scaler_cfg = scaler_cfg or {}
        self.metadata_cfg = metadata_cfg or {}

        self.gate_mode = _normalize_gate_mode(gate_mode)
        self.hard_gate_sensor_threshold = float(hard_gate_sensor_threshold)
        self.soft_mask_center = float(soft_mask_center)
        self.soft_mask_width = float(max(soft_mask_width, 1e-6))
        self.soft_mask_floor = float(np.clip(soft_mask_floor, 0.0, 1.0))
        self.correction_clip_abs = correction_clip_abs
        self.debug_rollout = bool(debug_rollout)
        self.debug_t_start = float(debug_t_start)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        warn_on_missing = bool(self.metadata_cfg.get("warn_on_missing", True))
        strict_metadata = bool(self.metadata_cfg.get("strict", True))

        self.metadata = extract_checkpoint_metadata(
            checkpoint,
            source_label=str(model_path),
            warn_on_missing=warn_on_missing,
        )
        self.metadata.setdefault("steps_ahead", int(checkpoint.get("steps_ahead", 1)))
        self.metadata.setdefault("stencil_size", int(checkpoint.get("stencil_size", 9)))
        self.metadata.setdefault("phys_dim", int(checkpoint.get("phys_dim", 7)))
        self.metadata.setdefault("physics_feature_names", physics_feature_names(int(self.metadata["phys_dim"])))

        has_target_scaler_state = "target_scaler_state" in checkpoint
        if "target_scaling" not in self.metadata:
            if warn_on_missing:
                warnings.warn(
                    f"{model_path}: target_scaling metadata missing, inferring from checkpoint fields.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.metadata["target_scaling"] = default_target_scaling_metadata(
                target_scaler_enabled=has_target_scaler_state
            )

        try:
            validate_one_step_metadata(
                self.metadata,
                source_label=str(model_path),
                expected_steps_ahead=1,
                expected_stencil_size=int(self.metadata["stencil_size"]),
                expected_phys_dim=int(self.metadata["phys_dim"]),
                expected_feature_names=physics_feature_names(int(self.metadata["phys_dim"])),
                expected_target_scaling=default_target_scaling_metadata(
                    target_scaler_enabled=bool(self.metadata["target_scaling"].get("state_required", has_target_scaler_state))
                ),
            )
        except ValueError:
            if strict_metadata:
                raise
            warnings.warn(
                f"{model_path}: checkpoint metadata validation failed, continuing because metadata.strict=false.",
                RuntimeWarning,
                stacklevel=2,
            )

        if bool(self.metadata["target_scaling"].get("state_required", False)) and not has_target_scaler_state:
            raise ValueError(
                "Checkpoint metadata requires target_scaler_state, but the state is missing. "
                "This artifact is incompatible with strict one-step evaluation."
            )

        self.stencil_size = int(self.metadata["stencil_size"])
        self.phys_dim = int(self.metadata["phys_dim"])
        self.steps_ahead = int(self.metadata.get("steps_ahead", 1))
        self.default_correction_sign = infer_default_sign(self.metadata)
        if self.steps_ahead != 1:
            raise ValueError(
                f"Checkpoint uses steps_ahead={self.steps_ahead}; strict one-step correction requires 1."
            )

        ckpt_model_cfg = checkpoint.get("config", {}).get("model", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
        self.model_runtime_cfg = resolve_model_runtime_config(
            stencil_size=self.stencil_size,
            artifact_model_cfg=ckpt_model_cfg,
            fallback_model_cfg=self.model_cfg,
            metadata=self.metadata,
            prefer_legacy_when_missing=True,
            warn_on_legacy=warn_on_missing,
            source_label=str(model_path),
        )
        self.feature_layout = dict(self.metadata.get("feature_layout", {}))
        self.correction_head = dict(self.metadata.get("correction_head", {}))

        self.scaler = HybridScaler(
            stencil_size=self.stencil_size,
            eps=float(self.scaler_cfg.get("eps", 1e-8)),
            clip_percentile_abs_features=float(self.scaler_cfg.get("clip_percentile_abs_features", 99.5)),
        )
        self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.target_scaler = TargetAffineScaler()
        if has_target_scaler_state:
            self.target_scaler.load_state_dict(checkpoint["target_scaler_state"])
            print(
                "Target scaler loaded: "
                f"y_mean={self.target_scaler.mean:.3e}, "
                f"y_std={self.target_scaler.std:.3e}"
            )
        else:
            warnings.warn(
                "Target scaler missing in checkpoint. Falling back to identity inverse transform.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.model = GatedKAN(
            stencil_size=self.stencil_size,
            phys_dim=self.phys_dim,
            hidden_dim=int(self.model_runtime_cfg["hidden_dim"]),
            shape_grid_size=int(self.model_runtime_cfg["shape_grid_size"]),
            shape_spline_order=int(self.model_runtime_cfg["shape_spline_order"]),
            correction_head_output_mode=str(self.model_runtime_cfg["correction_head_output_mode"]),
            correction_head_output_scale=float(self.model_runtime_cfg["correction_head_output_scale"]),
            use_stencil_features=bool(self.model_runtime_cfg["use_stencil_features"]),
            stencil_radius=int(self.model_runtime_cfg["stencil_radius"]),
            gate_use_stencil_features=bool(self.model_runtime_cfg["gate_use_stencil_features"]),
            gate_hidden_dims=tuple(self.model_runtime_cfg["gate_hidden_dims"]),
            gate_temperature=float(self.model_runtime_cfg["gate_temperature"]),
            gate_bias_init=float(self.model_runtime_cfg["gate_bias_init"]),
            shock_indicator_threshold=float(self.model_runtime_cfg["shock_indicator_threshold"]),
            curvature_eps=float(self.model_runtime_cfg["curvature_eps"]),
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(
            "Gate config: "
            f"mode={self.gate_mode}, hard_thr={self.hard_gate_sensor_threshold:.3f}, "
            f"soft_center={self.soft_mask_center:.3f}, soft_width={self.soft_mask_width:.3f}, "
            f"soft_floor={self.soft_mask_floor:.3f}, corr_clip_abs={self.correction_clip_abs}"
        )
        print(
            "Correction head config: "
            f"mode={self.model_runtime_cfg['correction_head_output_mode']}, "
            f"scale={self.model_runtime_cfg['correction_head_output_scale']:.3f}, "
            f"use_stencil={self.model_runtime_cfg['use_stencil_features']}, "
            f"stencil_radius={self.model_runtime_cfg['stencil_radius']}, "
            f"gate_use_stencil={self.model_runtime_cfg['gate_use_stencil_features']}"
        )

    def _shock_sensor_from_inputs(self, inputs_norm):
        shock_idx = self.stencil_size + (1 if self.phys_dim > 1 else 0)
        sensor = inputs_norm[:, shock_idx]
        return np.clip(sensor, 0.0, 1.0)

    def _soft_mask(self, shock_sensor):
        z = (shock_sensor - self.soft_mask_center) / self.soft_mask_width
        mask = 1.0 / (1.0 + np.exp(-z))
        mask = self.soft_mask_floor + (1.0 - self.soft_mask_floor) * mask
        return np.clip(mask, 0.0, 1.0)

    def _effective_gate(self, gate_raw, shock_sensor):
        hard_mask = (shock_sensor >= self.hard_gate_sensor_threshold).astype(np.float32)
        soft_mask = self._soft_mask(shock_sensor).astype(np.float32)

        if self.gate_mode == "train_equivalent":
            eff = gate_raw
            mode = "TRAIN_EQUIVALENT"
            mask_for_stats = np.ones_like(hard_mask, dtype=np.float32)
        elif self.gate_mode == "raw_gate_only":
            eff = gate_raw
            mode = "RAW_GATE_ONLY"
            mask_for_stats = np.ones_like(hard_mask, dtype=np.float32)
        elif self.gate_mode == "soft_mask":
            eff = gate_raw * soft_mask
            mode = "SOFT_MASK"
            mask_for_stats = (soft_mask > 0.5).astype(np.float32)
        elif self.gate_mode == "hard_mask":
            eff = gate_raw * hard_mask
            mode = "HARD_MASK"
            mask_for_stats = hard_mask
        elif self.gate_mode == "gate_open":
            eff = np.ones_like(gate_raw, dtype=np.float32)
            mode = "GATE_OPEN"
            mask_for_stats = np.ones_like(hard_mask, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

        return np.clip(eff, 0.0, 1.0), hard_mask, soft_mask, mask_for_stats, mode

    def predict_from_inputs_raw(self, inputs_raw, return_debug=False, return_components=False):
        inputs_norm = self.scaler.transform(np.asarray(inputs_raw))
        inputs_tensor = torch.as_tensor(inputs_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            raw_output_z_t, gate_raw_t, gated_output_z_t = self.model.forward_components(inputs_tensor)

            raw_output_z = raw_output_z_t.cpu().numpy()
            gate_raw = gate_raw_t.cpu().numpy().flatten()
            gated_output_z = gated_output_z_t.cpu().numpy()
            shock_sensor = self._shock_sensor_from_inputs(inputs_norm)

            effective_gate, hard_mask, soft_mask, mask_for_stats, mode = self._effective_gate(
                gate_raw, shock_sensor
            )

            if self.gate_mode == "train_equivalent":
                corr_z = gated_output_z
            else:
                corr_z = raw_output_z * effective_gate[:, None]

            corr_phys = self.target_scaler.inverse_transform(corr_z).reshape(-1)
            corr_raw_phys = self.target_scaler.inverse_transform(raw_output_z).reshape(-1)

            if self.correction_clip_abs is not None:
                corr_phys = np.clip(corr_phys, -self.correction_clip_abs, self.correction_clip_abs)
                corr_raw_phys = np.clip(corr_raw_phys, -self.correction_clip_abs, self.correction_clip_abs)

            debug_metrics = {
                "mode": mode,
                "head_output_mode": self.model.correction_head_output_mode,
                "head_output_scale": float(self.model.correction_head_output_scale),
                "head_is_bounded": bool(self.model.correction_head_is_bounded),
                "shock_indicator_min": float(np.min(shock_sensor)),
                "shock_indicator_mean": float(np.mean(shock_sensor)),
                "shock_indicator_max": float(np.max(shock_sensor)),
                "raw_gate_min": float(np.min(gate_raw)),
                "raw_gate_mean": float(np.mean(gate_raw)),
                "raw_gate_max": float(np.max(gate_raw)),
                "effective_gate_min": float(np.min(effective_gate)),
                "effective_gate_mean": float(np.mean(effective_gate)),
                "effective_gate_max": float(np.max(effective_gate)),
                "mask_active_ratio": float(np.mean(mask_for_stats)),
                "hard_mask_active_ratio": float(np.mean(hard_mask)),
                "soft_mask_mean": float(np.mean(soft_mask)),
                "kan_raw_abs_mean": float(np.mean(np.abs(raw_output_z))),
                "kan_raw_abs_max": float(np.max(np.abs(raw_output_z))),
                "corr_before_gate_abs_mean": float(np.mean(np.abs(corr_raw_phys))),
                "corr_before_gate_abs_max": float(np.max(np.abs(corr_raw_phys))),
                "corr_after_gate_abs_mean": float(np.mean(np.abs(corr_phys))),
                "corr_after_gate_abs_max": float(np.max(np.abs(corr_phys))),
                "corr_abs_mean": float(np.mean(np.abs(corr_phys))),
                "corr_abs_max": float(np.max(np.abs(corr_phys))),
            }

            if return_debug and return_components:
                components = {
                    "shock_sensor": shock_sensor,
                    "raw_gate": gate_raw,
                    "effective_gate": effective_gate,
                    "mask_for_stats": mask_for_stats,
                    "corr_before_gate": corr_raw_phys,
                    "corr_after_gate": corr_phys,
                }
                return corr_phys, debug_metrics, components
            if return_debug:
                return corr_phys, debug_metrics
            return corr_phys, None

    def predict(self, u_current, dx, dt, t_stamp=0.0, return_debug=False, return_components=False):
        inputs_raw = build_model_inputs(
            u_current,
            stencil_size=self.stencil_size,
            dx=dx,
            dt=dt,
            t_stamp=t_stamp,
            phys_dim=self.phys_dim,
        )
        return self.predict_from_inputs_raw(
            inputs_raw,
            return_debug=return_debug,
            return_components=return_components,
        )


def _resolve_device(runtime_cfg):
    dev = str(runtime_cfg.get("device", "auto")).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _apply_legacy_overrides(cfg, legacy_kwargs):
    """兼容旧接口: run_evaluation(model_path=..., ...)."""
    if not legacy_kwargs:
        return cfg

    cfg = dict(cfg)
    cfg['paths'] = dict(cfg.get('paths', {}))
    cfg['evaluation'] = dict(cfg.get('evaluation', {}))
    cfg['ablation'] = dict(cfg.get('ablation', {}))
    cfg['logging'] = dict(cfg.get('logging', {}))

    if 'model_path' in legacy_kwargs:
        cfg['paths']['model_save_path'] = legacy_kwargs['model_path']

    eval_map = {
        'N_coarse': 'N_coarse',
        'N_ref': 'N_ref',
        'T_final': 'T_final',
        'cfl': 'cfl',
        'correction_scale': 'correction_scale',
    }
    for k_old, k_new in eval_map.items():
        if k_old in legacy_kwargs and legacy_kwargs[k_old] is not None:
            cfg['evaluation'][k_new] = legacy_kwargs[k_old]

    ab_map = {
        'gate_mode': 'gate_mode',
        'hard_gate_sensor_threshold': 'hard_gate_sensor_threshold',
        'soft_mask_center': 'soft_mask_center',
        'soft_mask_width': 'soft_mask_width',
        'soft_mask_floor': 'soft_mask_floor',
        'correction_clip_abs': 'correction_clip_abs',
        'correction_sign_mode': 'correction_sign_mode',
    }
    for k_old, k_new in ab_map.items():
        if k_old in legacy_kwargs and legacy_kwargs[k_old] is not None:
            cfg['ablation'][k_new] = legacy_kwargs[k_old]

    log_map = {
        'debug_rollout': 'debug_rollout',
        'debug_t_start': 'debug_t_start',
        'debug_sign_metrics': 'debug_sign_metrics',
        'debug_gate_metrics': 'debug_gate_metrics',
    }
    for k_old, k_new in log_map.items():
        if k_old in legacy_kwargs and legacy_kwargs[k_old] is not None:
            cfg['logging'][k_new] = legacy_kwargs[k_old]

    return cfg


def _resolve_correction_sign_mode(mode, metadata):
    mode_l = str(mode).lower()
    if mode_l == "auto":
        resolved = infer_default_sign(metadata)
        if resolved not in ("plus", "minus"):
            raise ValueError(f"Unsupported inferred correction sign: {resolved}")
        return resolved, (1.0 if resolved == "plus" else -1.0)
    if mode_l == "plus":
        return "plus", 1.0
    if mode_l == "minus":
        return "minus", -1.0
    raise ValueError(f"Unsupported correction_sign_mode: {mode}")


def _compute_pair_metrics(pred, true):
    if pred.size == 0:
        return {
            'mean_prod': float('nan'),
            'cosine_similarity': float('nan'),
            'sign_agreement_ratio': float('nan'),
            'correlation_coeff': float('nan'),
            'pred_abs_mean': float('nan'),
            'pred_abs_max': float('nan'),
            'true_abs_mean': float('nan'),
            'true_abs_max': float('nan'),
            'count': 0,
        }

    mean_prod = float(np.mean(pred * true))
    denom = float(np.linalg.norm(pred) * np.linalg.norm(true)) + 1e-12
    cosine = float(np.dot(pred, true) / denom)
    sign_agree = float(np.mean(np.sign(pred) == np.sign(true)))
    if pred.size < 2 or float(np.std(pred)) < 1e-12 or float(np.std(true)) < 1e-12:
        corr = float('nan')
    else:
        corr = float(np.corrcoef(pred, true)[0, 1])

    return {
        'mean_prod': mean_prod,
        'cosine_similarity': cosine,
        'sign_agreement_ratio': sign_agree,
        'correlation_coeff': corr,
        'pred_abs_mean': float(np.mean(np.abs(pred))),
        'pred_abs_max': float(np.max(np.abs(pred))),
        'true_abs_mean': float(np.mean(np.abs(true))),
        'true_abs_max': float(np.max(np.abs(true))),
        'count': int(pred.size),
    }


def _summarize_sign_buffers(pred_list, true_list, shock_mask_list):
    if not pred_list:
        return None

    pred = np.concatenate(pred_list).astype(np.float64)
    true = np.concatenate(true_list).astype(np.float64)
    shock_mask = np.concatenate(shock_mask_list).astype(bool)

    return {
        'global': _compute_pair_metrics(pred, true),
        'shock': _compute_pair_metrics(pred[shock_mask], true[shock_mask]),
    }


def _run_rollout_once(
    predictor,
    *,
    N_coarse,
    N_ref,
    T_final,
    cfl,
    remove_correction_mean,
    ic_cfg,
    nu,
    weno_epsilon,
    correction_sign_mode,
    correction_scale,
    alpha=1.0,
    debug_rollout=False,
    debug_t_start=1.0,
    debug_sign_metrics=False,
    debug_gate_metrics=False,
):
    resolved_sign_mode, sign_factor = _resolve_correction_sign_mode(
        correction_sign_mode, predictor.metadata
    )

    sin1_amp = float(ic_cfg.get("sin1_amp", 1.0))
    sin2_amp = float(ic_cfg.get("sin2_amp", 0.5))
    sin2_phase = float(ic_cfg.get("sin2_phase", 0.5))
    cos5_amp = float(ic_cfg.get("cos5_amp", -0.2))

    ref_substeps = require_integer_refinement(N_ref, N_coarse)
    x = np.linspace(0, 2 * np.pi, N_ref, endpoint=False)
    u_ref_init = sin1_amp * np.sin(x) + sin2_amp * np.sin(2 * x + sin2_phase) + cos5_amp * np.cos(5 * x)
    u_ref_init /= np.max(np.abs(u_ref_init))

    u_coarse = downsample_periodic(u_ref_init, ratio=ref_substeps)
    u_hybrid = u_coarse.copy()
    u_truth = u_ref_init.copy()

    t, dx_coarse, dx_ref = 0.0, 2 * np.pi / N_coarse, 2 * np.pi / N_ref
    effective_applied_scale = float(correction_scale) * float(alpha)

    history = {
        "time": [],
        "l2_base": [],
        "l2_hybrid": [],
        "raw_gate_mean": [],
        "effective_gate_mean": [],
        "mask_active_ratio": [],
        "pred_corr_abs_mean": [],
        "pred_corr_abs_max": [],
        "corr_abs_mean": [],
        "corr_abs_max": [],
        "baseline_update_abs_mean": [],
        "baseline_update_abs_max": [],
        "corr_to_baseline_ratio": [],
        "correction_scale": [],
        "effective_correction_scale": [],
    }

    one_pred_buf, one_true_buf, one_shock_buf = [], [], []
    gate_raw_buf, gate_eff_buf, gate_mask_buf = [], [], []
    shock_sensor_buf = []
    corr_before_gate_buf, corr_after_gate_buf = [], []
    applied_corr_buf = []

    while t < T_final:
        dt = cfl * dx_coarse / (max(np.max(np.abs(u_coarse)), np.max(np.abs(u_hybrid))) + 1e-6)
        if t + dt > T_final:
            dt = T_final - t

        for _ in range(ref_substeps):
            u_truth = rk3_step(u_truth, dx_ref, dt / ref_substeps, nu=nu, weno_epsilon=weno_epsilon)

        u_coarse = rk3_step(u_coarse, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)

        u_hybrid_prev = u_hybrid.copy()
        u_weno_next = rk3_step(u_hybrid, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)

        corr, dbg, components = predictor.predict(
            u_hybrid,
            dx_coarse,
            dt,
            t_stamp=t,
            return_debug=True,
            return_components=True,
        )
        pred_corr = corr.copy()
        corr_centered = corr - np.mean(corr) if remove_correction_mean else corr
        applied_corr = effective_applied_scale * sign_factor * corr_centered

        baseline_update = u_weno_next - u_hybrid_prev
        base_abs_mean = float(np.mean(np.abs(baseline_update)))
        base_abs_max = float(np.max(np.abs(baseline_update)))
        pred_corr_abs_mean = float(np.mean(np.abs(pred_corr)))
        pred_corr_abs_max = float(np.max(np.abs(pred_corr)))
        corr_abs_mean = float(np.mean(np.abs(applied_corr)))
        corr_abs_max = float(np.max(np.abs(applied_corr)))
        corr_to_base = corr_abs_mean / (base_abs_mean + 1e-12)

        u_truth_down = downsample_periodic(u_truth, ratio=ref_substeps)
        if debug_sign_metrics:
            shock_sensor = components["shock_sensor"]
            shock_mask = shock_sensor >= predictor.hard_gate_sensor_threshold
            one_true_corr = u_truth_down - u_weno_next

            one_pred_buf.append(pred_corr.copy())
            one_true_buf.append(one_true_corr.copy())
            one_shock_buf.append(shock_mask.copy())

        if debug_gate_metrics:
            gate_raw_buf.append(components["raw_gate"].copy())
            gate_eff_buf.append(components["effective_gate"].copy())
            gate_mask_buf.append(components["mask_for_stats"].copy())
            shock_sensor_buf.append(components["shock_sensor"].copy())
            corr_before_gate_buf.append(components["corr_before_gate"].copy())
            corr_after_gate_buf.append(components["corr_after_gate"].copy())
            applied_corr_buf.append(applied_corr.copy())

        if debug_rollout and t > debug_t_start:
            print(
                "DEBUG "
                f"t={t:.3f} sign={resolved_sign_mode} mode={dbg['mode']} "
                f"corr_scale={correction_scale:.3f} eff_scale={effective_applied_scale:.3f} "
                f"raw_gate(mean/min/max)=({dbg['raw_gate_mean']:.3f}/{dbg['raw_gate_min']:.3f}/{dbg['raw_gate_max']:.3f}) "
                f"eff_gate(mean/min/max)=({dbg['effective_gate_mean']:.3f}/{dbg['effective_gate_min']:.3f}/{dbg['effective_gate_max']:.3f}) "
                f"mask_ratio={dbg['mask_active_ratio']:.3f} "
                f"shock(mean/min/max)=({dbg['shock_indicator_mean']:.3f}/{dbg['shock_indicator_min']:.3f}/{dbg['shock_indicator_max']:.3f}) "
                f"corr_before_gate_abs(mean/max)=({dbg['corr_before_gate_abs_mean']:.3e}/{dbg['corr_before_gate_abs_max']:.3e}) "
                f"corr_after_gate_abs(mean/max)=({dbg['corr_after_gate_abs_mean']:.3e}/{dbg['corr_after_gate_abs_max']:.3e}) "
                f"pred_corr_abs(mean/max)=({pred_corr_abs_mean:.3e}/{pred_corr_abs_max:.3e}) "
                f"applied_corr_abs(mean/max)=({corr_abs_mean:.3e}/{corr_abs_max:.3e}) "
                f"base_abs(mean/max)=({base_abs_mean:.3e}/{base_abs_max:.3e}) "
                f"corr/base={corr_to_base:.3e}"
            )

        # 默认路径保持与训练一致的 one-step 加法语义。
        u_hybrid = u_weno_next + applied_corr
        t += dt

        history["time"].append(t)
        history["l2_base"].append(np.sqrt(np.mean((u_coarse - u_truth_down) ** 2)))
        history["l2_hybrid"].append(np.sqrt(np.mean((u_hybrid - u_truth_down) ** 2)))
        history["raw_gate_mean"].append(dbg["raw_gate_mean"])
        history["effective_gate_mean"].append(dbg["effective_gate_mean"])
        history["mask_active_ratio"].append(dbg["mask_active_ratio"])
        history["pred_corr_abs_mean"].append(pred_corr_abs_mean)
        history["pred_corr_abs_max"].append(pred_corr_abs_max)
        history["corr_abs_mean"].append(corr_abs_mean)
        history["corr_abs_max"].append(corr_abs_max)
        history["baseline_update_abs_mean"].append(base_abs_mean)
        history["baseline_update_abs_max"].append(base_abs_max)
        history["corr_to_baseline_ratio"].append(corr_to_base)
        history["correction_scale"].append(float(correction_scale))
        history["effective_correction_scale"].append(float(effective_applied_scale))

    result = {
        "history": history,
        "resolved_sign_mode": resolved_sign_mode,
        "sign_factor": sign_factor,
        "correction_scale": float(correction_scale),
        "alpha": float(alpha),
        "effective_applied_scale": float(effective_applied_scale),
        "final_l2_base": float(history["l2_base"][-1]),
        "final_l2_hybrid": float(history["l2_hybrid"][-1]),
    }

    if debug_sign_metrics:
        result["sign_metrics"] = {
            "one_step": _summarize_sign_buffers(one_pred_buf, one_true_buf, one_shock_buf),
        }

    if debug_gate_metrics:
        gate_raw = np.concatenate(gate_raw_buf) if gate_raw_buf else np.array([], dtype=np.float64)
        gate_eff = np.concatenate(gate_eff_buf) if gate_eff_buf else np.array([], dtype=np.float64)
        gate_mask = np.concatenate(gate_mask_buf) if gate_mask_buf else np.array([], dtype=np.float64)
        shock_sensor = np.concatenate(shock_sensor_buf) if shock_sensor_buf else np.array([], dtype=np.float64)
        corr_before_gate = np.concatenate(corr_before_gate_buf) if corr_before_gate_buf else np.array([], dtype=np.float64)
        corr_after_gate = np.concatenate(corr_after_gate_buf) if corr_after_gate_buf else np.array([], dtype=np.float64)
        applied_corr = np.concatenate(applied_corr_buf) if applied_corr_buf else np.array([], dtype=np.float64)

        result["gate_metrics"] = {
            "raw_gate_min": float(np.min(gate_raw)),
            "raw_gate_mean": float(np.mean(gate_raw)),
            "raw_gate_max": float(np.max(gate_raw)),
            "effective_gate_min": float(np.min(gate_eff)),
            "effective_gate_mean": float(np.mean(gate_eff)),
            "effective_gate_max": float(np.max(gate_eff)),
            "mask_active_ratio": float(np.mean(gate_mask)),
            "shock_indicator_min": float(np.min(shock_sensor)),
            "shock_indicator_mean": float(np.mean(shock_sensor)),
            "shock_indicator_max": float(np.max(shock_sensor)),
            "corr_before_gate_abs_mean": float(np.mean(np.abs(corr_before_gate))),
            "corr_before_gate_abs_max": float(np.max(np.abs(corr_before_gate))),
            "corr_after_gate_abs_mean": float(np.mean(np.abs(corr_after_gate))),
            "corr_after_gate_abs_max": float(np.max(np.abs(corr_after_gate))),
            "applied_corr_abs_mean": float(np.mean(np.abs(applied_corr))),
            "applied_corr_abs_max": float(np.max(np.abs(applied_corr))),
            "corr_to_baseline_ratio_mean": float(np.mean(history["corr_to_baseline_ratio"])),
            "corr_to_baseline_ratio_max": float(np.max(history["corr_to_baseline_ratio"])),
            "correction_scale": float(correction_scale),
            "effective_applied_scale": float(effective_applied_scale),
        }

    return result


def _print_sign_metrics(label, metrics):
    if metrics is None:
        print(f"{label}: unavailable")
        return
    g = metrics['global']
    s = metrics['shock']
    print(
        f"{label} global: count={g['count']} "
        f"mean(pred*true)={g['mean_prod']:.3e}, cosine={g['cosine_similarity']:.3f}, "
        f"sign_ratio={g['sign_agreement_ratio']:.3f}, corr={g['correlation_coeff']:.3f}, "
        f"|pred|(mean/max)=({g['pred_abs_mean']:.3e}/{g['pred_abs_max']:.3e}), "
        f"|true|(mean/max)=({g['true_abs_mean']:.3e}/{g['true_abs_max']:.3e})"
    )
    print(
        f"{label} shock : count={s['count']} "
        f"mean(pred*true)={s['mean_prod']:.3e}, cosine={s['cosine_similarity']:.3f}, "
        f"sign_ratio={s['sign_agreement_ratio']:.3f}, corr={s['correlation_coeff']:.3f}, "
        f"|pred|(mean/max)=({s['pred_abs_mean']:.3e}/{s['pred_abs_max']:.3e}), "
        f"|true|(mean/max)=({s['true_abs_mean']:.3e}/{s['true_abs_max']:.3e})"
    )


def run_evaluation(cfg=None, **legacy_kwargs):
    if isinstance(cfg, dict):
        run_cfg = cfg
    else:
        run_cfg = load_config(None)
        if cfg is not None:
            legacy_kwargs.setdefault("model_path", cfg)

    run_cfg = _apply_legacy_overrides(run_cfg, legacy_kwargs)
    runtime_cfg = run_cfg.get("runtime", {})
    eval_cfg = run_cfg.get("evaluation", {})
    path_cfg = run_cfg.get("paths", {})
    ab_cfg = run_cfg.get("ablation", {})
    log_cfg = run_cfg.get("logging", {})
    plot_cfg = run_cfg.get("plotting", {})
    solver_cfg = run_cfg.get("solver", {})
    metadata_cfg = run_cfg.get("metadata", {})

    cfg_steps_ahead = int(cfg_get(run_cfg, "data_generation.steps_ahead", 1))
    if cfg_steps_ahead != 1:
        raise ValueError(
            "Evaluation config must use strict one-step correction: "
            f"data_generation.steps_ahead={cfg_steps_ahead}."
        )

    set_global_seed(runtime_cfg.get("seed", None), bool(runtime_cfg.get("deterministic", False)))

    device = _resolve_device(runtime_cfg)
    model_path = Path(path_cfg.get("model_save_path", "kan_model.pth"))

    predictor = KANPredictor(
        model_path=str(model_path),
        device=device,
        model_cfg=run_cfg.get("model", {}),
        scaler_cfg=run_cfg.get("scaler", {}),
        gate_mode=str(ab_cfg.get("gate_mode", "train_equivalent")),
        hard_gate_sensor_threshold=float(ab_cfg.get("hard_gate_sensor_threshold", 0.1)),
        soft_mask_center=float(ab_cfg.get("soft_mask_center", 0.1)),
        soft_mask_width=float(ab_cfg.get("soft_mask_width", 0.08)),
        soft_mask_floor=float(ab_cfg.get("soft_mask_floor", 0.02)),
        correction_clip_abs=ab_cfg.get("correction_clip_abs", None),
        debug_rollout=bool(log_cfg.get("debug_rollout", False)),
        debug_t_start=float(log_cfg.get("debug_t_start", 1.0)),
        metadata_cfg=metadata_cfg,
    )

    N_coarse = int(eval_cfg.get("N_coarse", 128))
    N_ref = int(eval_cfg.get("N_ref", 2048))
    T_final = float(eval_cfg.get("T_final", 1.5))
    cfl = float(eval_cfg.get("cfl", 0.4))
    correction_scale = float(eval_cfg.get("correction_scale", 0.25))
    remove_correction_mean = bool(eval_cfg.get("remove_correction_mean", False))
    ic_cfg = eval_cfg.get("initial_condition", {})
    nu = float(solver_cfg.get("nu", 0.0))
    weno_epsilon = float(solver_cfg.get("weno_epsilon", 1e-6))

    correction_sign_mode = str(ab_cfg.get("correction_sign_mode", "auto")).lower()
    debug_rollout = bool(log_cfg.get("debug_rollout", False))
    debug_t_start = float(log_cfg.get("debug_t_start", 1.0))
    debug_sign_metrics = bool(log_cfg.get("debug_sign_metrics", False))
    debug_gate_metrics = bool(log_cfg.get("debug_gate_metrics", False))

    print(
        "Starting Rollout Evaluation... "
        f"correction_sign_mode={correction_sign_mode}, "
        f"correction_scale={correction_scale:.3f}, "
        f"default_sign={predictor.default_correction_sign}, "
        f"gate_mode={predictor.gate_mode}, "
        f"remove_correction_mean={remove_correction_mean}, "
        f"debug_sign_metrics={debug_sign_metrics}, "
        f"debug_gate_metrics={debug_gate_metrics}"
    )

    main_result = _run_rollout_once(
        predictor,
        N_coarse=N_coarse,
        N_ref=N_ref,
        T_final=T_final,
        cfl=cfl,
        remove_correction_mean=remove_correction_mean,
        ic_cfg=ic_cfg,
        nu=nu,
        weno_epsilon=weno_epsilon,
        correction_sign_mode=correction_sign_mode,
        correction_scale=correction_scale,
        alpha=1.0,
        debug_rollout=debug_rollout,
        debug_t_start=debug_t_start,
        debug_sign_metrics=debug_sign_metrics,
        debug_gate_metrics=debug_gate_metrics,
    )
    history = main_result["history"]

    print(
        "Rollout summary: "
        f"final L2 baseline={main_result['final_l2_base']:.6e}, "
        f"final L2 hybrid={main_result['final_l2_hybrid']:.6e}, "
        f"ratio={main_result['final_l2_hybrid'] / (main_result['final_l2_base'] + 1e-12):.3f}, "
        f"resolved_sign_mode={main_result['resolved_sign_mode']}, "
        f"effective_applied_scale={main_result['effective_applied_scale']:.3f}"
    )

    if debug_sign_metrics:
        print("Sign diagnostics (pred_corr vs one-step true_corr):")
        _print_sign_metrics("one-step", main_result["sign_metrics"]["one_step"])

        plus_minus_results = {}
        for mode in ("plus", "minus"):
            if mode == main_result["resolved_sign_mode"]:
                plus_minus_results[mode] = main_result
            else:
                plus_minus_results[mode] = _run_rollout_once(
                    predictor,
                    N_coarse=N_coarse,
                    N_ref=N_ref,
                    T_final=T_final,
                    cfl=cfl,
                    remove_correction_mean=remove_correction_mean,
                    ic_cfg=ic_cfg,
                    nu=nu,
                    weno_epsilon=weno_epsilon,
                    correction_sign_mode=mode,
                    correction_scale=correction_scale,
                    alpha=1.0,
                    debug_rollout=False,
                    debug_sign_metrics=False,
                    debug_gate_metrics=False,
                )

        print("Correction sign ablation (plus vs minus):")
        for mode in ("plus", "minus"):
            res = plus_minus_results[mode]
            ratio = res['final_l2_hybrid'] / (res['final_l2_base'] + 1e-12)
            print(
                f"  {mode}: final L2 baseline={res['final_l2_base']:.6e}, "
                f"hybrid={res['final_l2_hybrid']:.6e}, ratio={ratio:.3f}"
            )

    if debug_gate_metrics and 'gate_metrics' in main_result:
        g = main_result["gate_metrics"]
        print(
            "Gate diagnostics summary: "
            f"raw_gate(mean/min/max)=({g['raw_gate_mean']:.3f}/{g['raw_gate_min']:.3f}/{g['raw_gate_max']:.3f}), "
            f"effective_gate(mean/min/max)=({g['effective_gate_mean']:.3f}/{g['effective_gate_min']:.3f}/{g['effective_gate_max']:.3f}), "
            f"mask_active_ratio={g['mask_active_ratio']:.3f}, "
            f"shock(mean/min/max)=({g['shock_indicator_mean']:.3f}/{g['shock_indicator_min']:.3f}/{g['shock_indicator_max']:.3f}), "
            f"corr_before_gate_abs(mean/max)=({g['corr_before_gate_abs_mean']:.3e}/{g['corr_before_gate_abs_max']:.3e}), "
            f"corr_after_gate_abs(mean/max)=({g['corr_after_gate_abs_mean']:.3e}/{g['corr_after_gate_abs_max']:.3e}), "
            f"applied_corr_abs(mean/max)=({g['applied_corr_abs_mean']:.3e}/{g['applied_corr_abs_max']:.3e}), "
            f"corr/base(mean/max)=({g['corr_to_baseline_ratio_mean']:.3e}/{g['corr_to_baseline_ratio_max']:.3e})"
        )

    if not bool(plot_cfg.get("enabled", True)):
        print("Evaluation complete. Plotting disabled by config.")
        return main_result

    try:
        fig_size = plot_cfg.get("figsize", [10, 8])
        fig, axes = plt.subplots(2, 1, figsize=(fig_size[0], fig_size[1]), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

        axes[0].plot(history['time'], history['l2_base'], 'k--', label='Baseline (WENO5)')
        axes[0].plot(history['time'], history['l2_hybrid'], 'r-', label='Hybrid (WENO5+KAN)')
        axes[0].set_yscale(str(plot_cfg.get("l2_yscale", "log")))
        axes[0].set_title('L2 Error Comparison')
        axes[0].grid(True)
        axes[0].legend()

        ax_g = axes[1]
        ax_g.plot(history['time'], history['raw_gate_mean'], 'b-', label='Mean Raw Gate')
        ax_g.plot(history['time'], history['effective_gate_mean'], 'g-', label='Mean Effective Gate')
        ax_g.plot(history['time'], history['mask_active_ratio'], 'm--', label='Mask Active Ratio')

        gate_ylim = plot_cfg.get("gate_ylim", [-0.02, 1.02])
        ax_g.set_ylim(gate_ylim[0], gate_ylim[1])
        ax_g.set_ylabel('Gate / Ratio')
        ax_g.grid(True)

        ax_c = ax_g.twinx()
        corr_mag = np.maximum(np.asarray(history['corr_abs_mean']), 1e-16)
        ax_c.plot(history['time'], corr_mag, 'c-', label='Applied Correction |.| mean')
        ax_c.set_yscale('log')
        ax_c.set_ylabel('Correction Magnitude')

        lines_g, labels_g = ax_g.get_legend_handles_labels()
        lines_c, labels_c = ax_c.get_legend_handles_labels()
        ax_g.legend(lines_g + lines_c, labels_g + labels_c, loc='upper left')

        axes[1].set_xlabel('Time')
        plt.tight_layout()

        out_path = Path(path_cfg.get("evaluation_plot_path", "evaluation_result.png"))
        dpi = int(plot_cfg.get("dpi", 120))
        plt.savefig(str(out_path), dpi=dpi)
        print(f"Evaluation complete. Results saved to '{out_path}'.")
    except Exception as e:
        print(f"Plotting failed (expected in some CI environments): {e}")
        print("Evaluation numeric data collected successfully.")

    return main_result


def _parse_args():
    parser = argparse.ArgumentParser(description='Evaluate hybrid WENO5+KAN rollout with config.')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（可选）')
    parser.add_argument(
        '--correction-sign-mode',
        type=str,
        choices=['auto', 'plus', 'minus'],
        default=None,
        help='Correction sign mode override: auto/plus/minus',
    )
    parser.add_argument(
        '--debug-sign-metrics',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable/disable sign diagnostics and plus/minus rollout comparison',
    )
    parser.add_argument(
        '--debug-gate-metrics',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable/disable gate/correction magnitude diagnostics summary',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    overrides = {}
    if args.correction_sign_mode is not None:
        overrides['correction_sign_mode'] = args.correction_sign_mode
    if args.debug_sign_metrics is not None:
        overrides['debug_sign_metrics'] = bool(args.debug_sign_metrics)
    if args.debug_gate_metrics is not None:
        overrides['debug_gate_metrics'] = bool(args.debug_gate_metrics)
    run_evaluation(cfg, **overrides)
