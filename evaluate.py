import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solvers.weno import rk3_step
from numpy.lib.stride_tricks import sliding_window_view
from data.generate import get_multistep_error
from kan import GatedKAN, HybridScaler
from utils.config import load_config, set_global_seed


class TargetAffineScaler:
    """Affine target scaler used during train/eval checkpoint exchange."""

    def __init__(self, eps=1e-12):
        self.eps = eps
        self.mean = 0.0
        self.std = 1.0

    def load_state_dict(self, state):
        self.mean = float(state.get("mean", 0.0))
        std = float(state.get("std", 1.0))
        self.std = std if std >= self.eps else 1.0

    def inverse_transform(self, y):
        return y * self.std + self.mean


class KANPredictor:
    """
    Predictor with configurable gate policy.

    gate_mode options:
      - original:      effective_gate = raw_gate * hard_mask
      - raw_gate_only: effective_gate = raw_gate
      - gate_open:     effective_gate = 1
      - soft_mask:     effective_gate = raw_gate * soft_mask (recommended)
    """

    def __init__(
        self,
        model_path,
        device='cpu',
        model_cfg=None,
        scaler_cfg=None,
        gate_mode='soft_mask',
        hard_gate_sensor_threshold=0.1,
        soft_mask_center=0.1,
        soft_mask_width=0.08,
        soft_mask_floor=0.02,
        correction_clip_abs=None,
        debug_rollout=False,
        debug_t_start=1.0,
    ):
        self.device = device
        self.model_cfg = model_cfg or {}
        self.scaler_cfg = scaler_cfg or {}

        self.gate_mode = gate_mode
        self.hard_gate_sensor_threshold = float(hard_gate_sensor_threshold)
        self.soft_mask_center = float(soft_mask_center)
        self.soft_mask_width = float(max(soft_mask_width, 1e-6))
        self.soft_mask_floor = float(np.clip(soft_mask_floor, 0.0, 1.0))
        self.correction_clip_abs = correction_clip_abs
        self.debug_rollout = bool(debug_rollout)
        self.debug_t_start = float(debug_t_start)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.stencil_size = checkpoint.get('stencil_size', 9)
        self.phys_dim = checkpoint.get('phys_dim', 3)
        self.steps_ahead = checkpoint.get('steps_ahead', 10)

        # 优先使用 checkpoint 中保存的模型配置，避免结构不匹配。
        ckpt_model_cfg = checkpoint.get('config', {}).get('model', {}) if isinstance(checkpoint.get('config', {}), dict) else {}
        model_cfg = dict(self.model_cfg)
        model_cfg.update(ckpt_model_cfg)

        self.scaler = HybridScaler(
            stencil_size=self.stencil_size,
            eps=float(self.scaler_cfg.get("eps", 1e-8)),
            clip_percentile_abs_features=float(self.scaler_cfg.get("clip_percentile_abs_features", 99.5)),
        )
        self.scaler.load_state_dict(checkpoint['scaler_state'])

        self.target_scaler = TargetAffineScaler()
        if 'target_scaler_state' in checkpoint:
            self.target_scaler.load_state_dict(checkpoint['target_scaler_state'])
            print(
                "Target scaler loaded: "
                f"y_mean={self.target_scaler.mean:.3e}, "
                f"y_std={self.target_scaler.std:.3e}"
            )
        else:
            print(
                "Target scaler missing in checkpoint. "
                "Using identity fallback: y_mean=0.000e+00, y_std=1.000e+00"
            )

        self.model = GatedKAN(
            stencil_size=self.stencil_size,
            phys_dim=self.phys_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            shape_grid_size=int(model_cfg.get("shape_grid_size", 10)),
            shape_spline_order=int(model_cfg.get("shape_spline_order", 3)),
            shape_output_scale=float(model_cfg.get("shape_output_scale", 0.5)),
            gate_hidden_dims=tuple(model_cfg.get("gate_hidden_dims", [32, 16])),
            gate_temperature=float(model_cfg.get("gate_temperature", 2.0)),
            gate_bias_init=float(model_cfg.get("gate_bias_init", -1.0)),
            shock_indicator_threshold=float(model_cfg.get("shock_indicator_threshold", 0.15)),
            curvature_eps=float(model_cfg.get("curvature_eps", 1e-4)),
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(
            "Gate config: "
            f"mode={self.gate_mode}, hard_thr={self.hard_gate_sensor_threshold:.3f}, "
            f"soft_center={self.soft_mask_center:.3f}, soft_width={self.soft_mask_width:.3f}, "
            f"soft_floor={self.soft_mask_floor:.3f}, corr_clip_abs={self.correction_clip_abs}"
        )

    def _physics_features(self, u_current, dx, dt, t_stamp):
        u_x = np.gradient(u_current, dx)
        abs_ux = np.abs(u_x)

        if self.phys_dim >= 7:
            u_xx = np.gradient(u_x, dx)
            abs_uxx = np.abs(u_xx)
            grad_var = np.sqrt(
                (
                    (np.roll(u_x, -1) - u_x) ** 2
                    + (np.roll(u_x, 1) - u_x) ** 2
                ) * 0.5
            )
            dt_feat = np.full_like(u_x, dt)
            t_sin = np.full_like(u_x, np.sin(t_stamp))
            t_cos = np.full_like(u_x, np.cos(t_stamp))
            return np.stack([u_x, abs_ux, abs_uxx, grad_var, dt_feat, t_sin, t_cos], axis=1)

        if self.phys_dim == 3:
            dt_feat = np.full_like(u_x, dt)
            return np.stack([u_x, abs_ux, dt_feat], axis=1)

        feats = [u_x, abs_ux]
        while len(feats) < self.phys_dim:
            feats.append(np.zeros_like(u_x))
        return np.stack(feats[:self.phys_dim], axis=1)

    def _shock_sensor_from_inputs(self, inputs_norm):
        # 与训练一致：使用归一化后的 |u_x| 通道。
        shock_idx = self.stencil_size + (1 if self.phys_dim > 1 else 0)
        sensor = inputs_norm[:, shock_idx]
        return np.clip(sensor, 0.0, 1.0)

    def _forward_components(self, inputs_tensor):
        x_physics = inputs_tensor[:, self.stencil_size:]
        raw_output = self.model.shape_net(inputs_tensor)
        raw_output = F.softsign(raw_output) * self.model.shape_output_scale

        if hasattr(self.model, '_build_gate_input'):
            gate_input = self.model._build_gate_input(x_physics)
        else:
            gate_input = x_physics

        gate_logits = self.model.gate_net(gate_input)
        gate_temperature = getattr(self.model, 'gate_temperature', 1.0)
        gate_raw = torch.sigmoid(gate_logits / gate_temperature)

        gated_output = raw_output * gate_raw
        return raw_output, gate_raw, gated_output

    def _soft_mask(self, shock_sensor):
        z = (shock_sensor - self.soft_mask_center) / self.soft_mask_width
        mask = 1.0 / (1.0 + np.exp(-z))
        mask = self.soft_mask_floor + (1.0 - self.soft_mask_floor) * mask
        return np.clip(mask, 0.0, 1.0)

    def _effective_gate(self, gate_raw, shock_sensor, t_stamp):
        hard_mask = (shock_sensor >= self.hard_gate_sensor_threshold).astype(np.float32)
        soft_mask = self._soft_mask(shock_sensor).astype(np.float32)

        if self.gate_mode == 'original':
            eff = gate_raw * hard_mask
            mode = 'ORIGINAL'
            mask_for_stats = hard_mask
        elif self.gate_mode == 'raw_gate_only':
            eff = gate_raw
            mode = 'RAW_GATE_ONLY'
            mask_for_stats = np.ones_like(hard_mask, dtype=np.float32)
        elif self.gate_mode == 'gate_open':
            eff = np.ones_like(gate_raw, dtype=np.float32)
            mode = 'GATE_OPEN'
            mask_for_stats = np.ones_like(hard_mask, dtype=np.float32)
        elif self.gate_mode == 'soft_mask':
            eff = gate_raw * soft_mask
            mode = 'SOFT_MASK'
            mask_for_stats = (soft_mask > 0.5).astype(np.float32)
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

        if self.debug_rollout and t_stamp > self.debug_t_start:
            eff = np.ones_like(gate_raw, dtype=np.float32)
            mode = 'FORCE_OPEN_DEBUG'

        return np.clip(eff, 0.0, 1.0), hard_mask, soft_mask, mask_for_stats, mode

    def predict(self, u_current, dx, dt, t_stamp=0.0, return_debug=False):
        pad = self.stencil_size // 2
        u_padded = np.pad(u_current, (pad, pad), mode='wrap')
        stencils = sliding_window_view(u_padded, window_shape=self.stencil_size)

        phys_feats = self._physics_features(u_current, dx, dt, t_stamp)
        inputs_raw = np.hstack([stencils, phys_feats])
        inputs_norm = self.scaler.transform(inputs_raw)
        inputs_tensor = torch.FloatTensor(inputs_norm).to(self.device)

        with torch.no_grad():
            raw_output_z, gate_raw_t, _ = self._forward_components(inputs_tensor)

            raw_output_z_np = raw_output_z.cpu().numpy()
            gate_raw_np = gate_raw_t.cpu().numpy().flatten()
            shock_sensor = self._shock_sensor_from_inputs(inputs_norm)

            effective_gate, hard_mask, soft_mask, mask_for_stats, mode = self._effective_gate(
                gate_raw_np, shock_sensor, t_stamp
            )

            corr_z = raw_output_z_np * effective_gate[:, None]
            corr_phys = self.target_scaler.inverse_transform(corr_z)
            corr_raw_phys = self.target_scaler.inverse_transform(raw_output_z_np)

            if self.correction_clip_abs is not None:
                corr_phys = np.clip(corr_phys, -self.correction_clip_abs, self.correction_clip_abs)
                corr_raw_phys = np.clip(corr_raw_phys, -self.correction_clip_abs, self.correction_clip_abs)

            corr_final = corr_phys.flatten() / self.steps_ahead
            corr_raw_final = corr_raw_phys.flatten() / self.steps_ahead

            debug_metrics = {
                'mode': mode,
                'shock_indicator_min': float(np.min(shock_sensor)),
                'shock_indicator_mean': float(np.mean(shock_sensor)),
                'shock_indicator_max': float(np.max(shock_sensor)),
                'raw_gate_min': float(np.min(gate_raw_np)),
                'raw_gate_mean': float(np.mean(gate_raw_np)),
                'raw_gate_max': float(np.max(gate_raw_np)),
                'effective_gate_min': float(np.min(effective_gate)),
                'effective_gate_mean': float(np.mean(effective_gate)),
                'effective_gate_max': float(np.max(effective_gate)),
                'mask_active_ratio': float(np.mean(mask_for_stats)),
                'hard_mask_active_ratio': float(np.mean(hard_mask)),
                'soft_mask_mean': float(np.mean(soft_mask)),
                'kan_raw_abs_mean': float(np.mean(np.abs(raw_output_z_np))),
                'kan_raw_abs_max': float(np.max(np.abs(raw_output_z_np))),
                'corr_before_gate_abs_mean': float(np.mean(np.abs(corr_raw_final))),
                'corr_before_gate_abs_max': float(np.max(np.abs(corr_raw_final))),
                'corr_after_gate_abs_mean': float(np.mean(np.abs(corr_final))),
                'corr_after_gate_abs_max': float(np.max(np.abs(corr_final))),
                'corr_abs_mean': float(np.mean(np.abs(corr_final))),
                'corr_abs_max': float(np.max(np.abs(corr_final))),
            }

            if return_debug:
                components = {
                    'shock_sensor': shock_sensor,
                    'raw_gate': gate_raw_np,
                    'effective_gate': effective_gate,
                    'mask_for_stats': mask_for_stats,
                    'corr_before_gate': corr_raw_final,
                    'corr_after_gate': corr_final,
                }
                return corr_final, debug_metrics, components
            return corr_final, None, None


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


def _resolve_correction_sign_mode(mode):
    mode_l = str(mode).lower()
    if mode_l in ("auto", "plus"):
        return mode_l, 1.0
    if mode_l == "minus":
        return mode_l, -1.0
    raise ValueError(f"Unsupported correction_sign_mode: {mode}")


def _compute_pair_metrics(pred, true):
    if pred.size == 0:
        return {
            'mean_prod': float('nan'),
            'cosine_similarity': float('nan'),
            'sign_agreement_ratio': float('nan'),
            'correlation_coeff': float('nan'),
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
    debug_rollout=False,
    debug_t_start=1.0,
    debug_sign_metrics=False,
    debug_gate_metrics=False,
):
    resolved_sign_mode, sign_factor = _resolve_correction_sign_mode(correction_sign_mode)

    sin1_amp = float(ic_cfg.get("sin1_amp", 1.0))
    sin2_amp = float(ic_cfg.get("sin2_amp", 0.5))
    sin2_phase = float(ic_cfg.get("sin2_phase", 0.5))
    cos5_amp = float(ic_cfg.get("cos5_amp", -0.2))

    x = np.linspace(0, 2 * np.pi, N_ref, endpoint=False)
    u_ref_init = sin1_amp * np.sin(x) + sin2_amp * np.sin(2 * x + sin2_phase) + cos5_amp * np.cos(5 * x)
    u_ref_init /= np.max(np.abs(u_ref_init))

    u_coarse = u_ref_init[::(N_ref // N_coarse)].copy()
    u_hybrid = u_coarse.copy()
    u_truth = u_ref_init.copy()

    t, dx_coarse, dx_ref = 0.0, 2 * np.pi / N_coarse, 2 * np.pi / N_ref
    ref_substeps = N_ref // N_coarse

    history = {
        'time': [],
        'l2_base': [],
        'l2_hybrid': [],
        'raw_gate_mean': [],
        'effective_gate_mean': [],
        'mask_active_ratio': [],
        'corr_abs_mean': [],
        'baseline_update_abs_mean': [],
        'corr_to_baseline_ratio': [],
    }

    one_pred_buf, one_true_buf, one_shock_buf = [], [], []
    train_pred_buf, train_true_buf, train_shock_buf = [], [], []
    gate_raw_buf, gate_eff_buf, gate_mask_buf = [], [], []
    shock_sensor_buf = []
    corr_before_gate_buf, corr_after_gate_buf = [], []

    while t < T_final:
        dt = cfl * dx_coarse / (max(np.max(np.abs(u_coarse)), np.max(np.abs(u_hybrid))) + 1e-6)
        if t + dt > T_final:
            dt = T_final - t

        for _ in range(ref_substeps):
            u_truth = rk3_step(u_truth, dx_ref, dt / ref_substeps, nu=nu, weno_epsilon=weno_epsilon)

        u_coarse = rk3_step(u_coarse, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)

        u_hybrid_prev = u_hybrid.copy()
        u_phys = rk3_step(u_hybrid, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)

        corr, dbg, components = predictor.predict(u_hybrid, dx_coarse, dt, t_stamp=t, return_debug=True)
        pred_corr = corr.copy()
        corr_centered = corr - np.mean(corr) if remove_correction_mean else corr
        applied_corr = sign_factor * corr_centered

        baseline_update = u_phys - u_hybrid_prev
        base_abs_mean = float(np.mean(np.abs(baseline_update)))
        base_abs_max = float(np.max(np.abs(baseline_update)))
        corr_abs_mean = float(np.mean(np.abs(applied_corr)))
        corr_abs_max = float(np.max(np.abs(applied_corr)))
        corr_to_base = corr_abs_mean / (base_abs_mean + 1e-12)

        u_truth_down = u_truth[::ref_substeps]
        if debug_sign_metrics:
            shock_sensor = components['shock_sensor']
            shock_mask = shock_sensor >= predictor.hard_gate_sensor_threshold
            one_true_corr = u_truth_down - u_phys
            train_true_corr = (
                get_multistep_error(
                    u_hybrid_prev,
                    N_coarse,
                    N_ref,
                    dt,
                    predictor.steps_ahead,
                    nu=nu,
                    weno_epsilon=weno_epsilon,
                ) / predictor.steps_ahead
            )

            one_pred_buf.append(pred_corr.copy())
            one_true_buf.append(one_true_corr.copy())
            one_shock_buf.append(shock_mask.copy())
            train_pred_buf.append(pred_corr.copy())
            train_true_buf.append(train_true_corr.copy())
            train_shock_buf.append(shock_mask.copy())

        if debug_gate_metrics:
            gate_raw_buf.append(components['raw_gate'].copy())
            gate_eff_buf.append(components['effective_gate'].copy())
            gate_mask_buf.append(components['mask_for_stats'].copy())
            shock_sensor_buf.append(components['shock_sensor'].copy())
            corr_before_gate_buf.append(components['corr_before_gate'].copy())
            corr_after_gate_buf.append(components['corr_after_gate'].copy())

        if debug_rollout and t > debug_t_start:
            print(
                "DEBUG "
                f"t={t:.3f} sign={resolved_sign_mode} mode={dbg['mode']} "
                f"raw_gate(mean/min/max)=({dbg['raw_gate_mean']:.3f}/{dbg['raw_gate_min']:.3f}/{dbg['raw_gate_max']:.3f}) "
                f"eff_gate(mean/min/max)=({dbg['effective_gate_mean']:.3f}/{dbg['effective_gate_min']:.3f}/{dbg['effective_gate_max']:.3f}) "
                f"mask_ratio={dbg['mask_active_ratio']:.3f} "
                f"shock(mean/min/max)=({dbg['shock_indicator_mean']:.3f}/{dbg['shock_indicator_min']:.3f}/{dbg['shock_indicator_max']:.3f}) "
                f"corr_before_gate_abs(mean/max)=({dbg['corr_before_gate_abs_mean']:.3e}/{dbg['corr_before_gate_abs_max']:.3e}) "
                f"corr_after_gate_abs(mean/max)=({dbg['corr_after_gate_abs_mean']:.3e}/{dbg['corr_after_gate_abs_max']:.3e}) "
                f"applied_corr_abs(mean/max)=({corr_abs_mean:.3e}/{corr_abs_max:.3e}) "
                f"base_abs(mean/max)=({base_abs_mean:.3e}/{base_abs_max:.3e}) "
                f"corr/base={corr_to_base:.3e}"
            )

        u_hybrid = u_phys + applied_corr
        t += dt

        history['time'].append(t)
        history['l2_base'].append(np.sqrt(np.mean((u_coarse - u_truth_down) ** 2)))
        history['l2_hybrid'].append(np.sqrt(np.mean((u_hybrid - u_truth_down) ** 2)))
        history['raw_gate_mean'].append(dbg['raw_gate_mean'])
        history['effective_gate_mean'].append(dbg['effective_gate_mean'])
        history['mask_active_ratio'].append(dbg['mask_active_ratio'])
        history['corr_abs_mean'].append(corr_abs_mean)
        history['baseline_update_abs_mean'].append(base_abs_mean)
        history['corr_to_baseline_ratio'].append(corr_to_base)

    result = {
        'history': history,
        'resolved_sign_mode': resolved_sign_mode,
        'sign_factor': sign_factor,
        'final_l2_base': float(history['l2_base'][-1]),
        'final_l2_hybrid': float(history['l2_hybrid'][-1]),
    }

    if debug_sign_metrics:
        result['sign_metrics'] = {
            'one_step': _summarize_sign_buffers(one_pred_buf, one_true_buf, one_shock_buf),
            'train_semantic': _summarize_sign_buffers(train_pred_buf, train_true_buf, train_shock_buf),
        }

    if debug_gate_metrics:
        gate_raw = np.concatenate(gate_raw_buf) if gate_raw_buf else np.array([], dtype=np.float64)
        gate_eff = np.concatenate(gate_eff_buf) if gate_eff_buf else np.array([], dtype=np.float64)
        gate_mask = np.concatenate(gate_mask_buf) if gate_mask_buf else np.array([], dtype=np.float64)
        shock_sensor = np.concatenate(shock_sensor_buf) if shock_sensor_buf else np.array([], dtype=np.float64)
        corr_before_gate = np.concatenate(corr_before_gate_buf) if corr_before_gate_buf else np.array([], dtype=np.float64)
        corr_after_gate = np.concatenate(corr_after_gate_buf) if corr_after_gate_buf else np.array([], dtype=np.float64)

        result['gate_metrics'] = {
            'raw_gate_min': float(np.min(gate_raw)),
            'raw_gate_mean': float(np.mean(gate_raw)),
            'raw_gate_max': float(np.max(gate_raw)),
            'effective_gate_min': float(np.min(gate_eff)),
            'effective_gate_mean': float(np.mean(gate_eff)),
            'effective_gate_max': float(np.max(gate_eff)),
            'mask_active_ratio': float(np.mean(gate_mask)),
            'shock_indicator_min': float(np.min(shock_sensor)),
            'shock_indicator_mean': float(np.mean(shock_sensor)),
            'shock_indicator_max': float(np.max(shock_sensor)),
            'corr_before_gate_abs_mean': float(np.mean(np.abs(corr_before_gate))),
            'corr_before_gate_abs_max': float(np.max(np.abs(corr_before_gate))),
            'corr_after_gate_abs_mean': float(np.mean(np.abs(corr_after_gate))),
            'corr_after_gate_abs_max': float(np.max(np.abs(corr_after_gate))),
            'applied_corr_abs_mean': float(np.mean(np.abs(history['corr_abs_mean']))),
            'applied_corr_abs_max': float(np.max(np.abs(history['corr_abs_mean']))),
            'corr_to_baseline_ratio_mean': float(np.mean(history['corr_to_baseline_ratio'])),
            'corr_to_baseline_ratio_max': float(np.max(history['corr_to_baseline_ratio'])),
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
        f"sign_ratio={g['sign_agreement_ratio']:.3f}, corr={g['correlation_coeff']:.3f}"
    )
    print(
        f"{label} shock : count={s['count']} "
        f"mean(pred*true)={s['mean_prod']:.3e}, cosine={s['cosine_similarity']:.3f}, "
        f"sign_ratio={s['sign_agreement_ratio']:.3f}, corr={s['correlation_coeff']:.3f}"
    )


def run_evaluation(cfg=None, **legacy_kwargs):
    # 新接口：run_evaluation(cfg)；旧接口：run_evaluation(model_path='...')
    if isinstance(cfg, dict):
        run_cfg = cfg
    else:
        run_cfg = load_config(None)
        if cfg is not None:
            legacy_kwargs.setdefault('model_path', cfg)

    run_cfg = _apply_legacy_overrides(run_cfg, legacy_kwargs)
    runtime_cfg = run_cfg.get("runtime", {})
    eval_cfg = run_cfg.get("evaluation", {})
    path_cfg = run_cfg.get("paths", {})
    ab_cfg = run_cfg.get("ablation", {})
    log_cfg = run_cfg.get("logging", {})
    plot_cfg = run_cfg.get("plotting", {})
    solver_cfg = run_cfg.get("solver", {})

    set_global_seed(runtime_cfg.get("seed", None), bool(runtime_cfg.get("deterministic", False)))

    device = _resolve_device(runtime_cfg)
    model_path = Path(path_cfg.get("model_save_path", "kan_model.pth"))

    try:
        predictor = KANPredictor(
            model_path=str(model_path),
            device=device,
            model_cfg=run_cfg.get("model", {}),
            scaler_cfg=run_cfg.get("scaler", {}),
            gate_mode=str(ab_cfg.get("gate_mode", "soft_mask")),
            hard_gate_sensor_threshold=float(ab_cfg.get("hard_gate_sensor_threshold", 0.1)),
            soft_mask_center=float(ab_cfg.get("soft_mask_center", 0.1)),
            soft_mask_width=float(ab_cfg.get("soft_mask_width", 0.08)),
            soft_mask_floor=float(ab_cfg.get("soft_mask_floor", 0.02)),
            correction_clip_abs=ab_cfg.get("correction_clip_abs", None),
            debug_rollout=bool(log_cfg.get("debug_rollout", False)),
            debug_t_start=float(log_cfg.get("debug_t_start", 1.0)),
        )
    except Exception:
        print("Model not found. Run training first.")
        return

    N_coarse = int(eval_cfg.get("N_coarse", 128))
    N_ref = int(eval_cfg.get("N_ref", 2048))
    T_final = float(eval_cfg.get("T_final", 1.5))
    cfl = float(eval_cfg.get("cfl", 0.4))
    remove_correction_mean = bool(eval_cfg.get("remove_correction_mean", True))
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
        debug_rollout=debug_rollout,
        debug_t_start=debug_t_start,
        debug_sign_metrics=debug_sign_metrics,
        debug_gate_metrics=debug_gate_metrics,
    )
    history = main_result['history']

    print(
        "Rollout summary: "
        f"final L2 baseline={main_result['final_l2_base']:.6e}, "
        f"final L2 hybrid={main_result['final_l2_hybrid']:.6e}, "
        f"ratio={main_result['final_l2_hybrid'] / (main_result['final_l2_base'] + 1e-12):.3f}, "
        f"resolved_sign_mode={main_result['resolved_sign_mode']}"
    )

    if debug_sign_metrics:
        print("Sign diagnostics (pred_corr vs true_corr):")
        _print_sign_metrics("one-step", main_result['sign_metrics']['one_step'])
        _print_sign_metrics("train-semantic", main_result['sign_metrics']['train_semantic'])

        plus_minus_results = {}
        for mode in ("plus", "minus"):
            if mode == main_result['resolved_sign_mode'] or (
                mode == "plus" and main_result['resolved_sign_mode'] == "auto"
            ):
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
        g = main_result['gate_metrics']
        print(
            "Gate diagnostics summary: "
            f"raw_gate(mean/min/max)=({g['raw_gate_mean']:.3f}/{g['raw_gate_min']:.3f}/{g['raw_gate_max']:.3f}), "
            f"effective_gate(mean/min/max)=({g['effective_gate_mean']:.3f}/{g['effective_gate_min']:.3f}/{g['effective_gate_max']:.3f}), "
            f"mask_active_ratio={g['mask_active_ratio']:.3f}, "
            f"shock(mean/min/max)=({g['shock_indicator_mean']:.3f}/{g['shock_indicator_min']:.3f}/{g['shock_indicator_max']:.3f}), "
            f"corr_before_gate_abs(mean/max)=({g['corr_before_gate_abs_mean']:.3e}/{g['corr_before_gate_abs_max']:.3e}), "
            f"corr_after_gate_abs(mean/max)=({g['corr_after_gate_abs_mean']:.3e}/{g['corr_after_gate_abs_max']:.3e}), "
            f"corr/base(mean/max)=({g['corr_to_baseline_ratio_mean']:.3e}/{g['corr_to_baseline_ratio_max']:.3e})"
        )

    if not bool(plot_cfg.get("enabled", True)):
        print("Evaluation complete. Plotting disabled by config.")
        return

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
