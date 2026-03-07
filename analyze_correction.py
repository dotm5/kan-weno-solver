import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

from evaluate import KANPredictor
from solvers.weno import rk3_step
from utils.config import cfg_get, load_config, set_global_seed
from utils.features import build_model_inputs, downsample_periodic, physics_feature_names, require_integer_refinement
from utils.metadata import extract_dataset_metadata, infer_default_sign


def _resolve_device(runtime_cfg):
    dev = str(runtime_cfg.get("device", "auto")).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _resolve_correction_sign_mode(mode, metadata):
    mode_l = str(mode).lower()
    if mode_l == "auto":
        resolved = infer_default_sign(metadata)
        return resolved, (1.0 if resolved == "plus" else -1.0)
    if mode_l == "plus":
        return "plus", 1.0
    if mode_l == "minus":
        return "minus", -1.0
    raise ValueError(f"Unsupported correction_sign_mode: {mode}")


def _ensure_parent(path_str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_mean(values):
    values = np.asarray(values)
    return float(np.mean(values)) if values.size else float("nan")


def _safe_std(values):
    values = np.asarray(values)
    return float(np.std(values)) if values.size else float("nan")


def _safe_min(values):
    values = np.asarray(values)
    return float(np.min(values)) if values.size else float("nan")


def _safe_max(values):
    values = np.asarray(values)
    return float(np.max(values)) if values.size else float("nan")


def _safe_quantile(values, q):
    values = np.asarray(values)
    return float(np.quantile(values, q)) if values.size else float("nan")


def _summary_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "mean": _safe_mean(values),
        "std": _safe_std(values),
        "abs_mean": _safe_mean(abs_values),
        "abs_max": _safe_max(abs_values),
        "abs_p50": _safe_quantile(abs_values, 0.50),
        "abs_p90": _safe_quantile(abs_values, 0.90),
        "abs_p95": _safe_quantile(abs_values, 0.95),
        "abs_p99": _safe_quantile(abs_values, 0.99),
    }


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return float("nan")
    return _pearson_corr(_rankdata(x), _rankdata(y))


def _pair_metrics(pred, true, *, near_zero_eps):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if pred.size == 0:
        return {
            "count": 0,
            "mse": float("nan"),
            "mae": float("nan"),
            "cosine_similarity": float("nan"),
            "correlation_coeff": float("nan"),
            "sign_agreement_ratio": float("nan"),
            "pred_abs_mean": float("nan"),
            "pred_abs_p50": float("nan"),
            "pred_abs_p90": float("nan"),
            "pred_abs_p95": float("nan"),
            "pred_abs_p99": float("nan"),
            "true_abs_mean": float("nan"),
            "true_abs_p50": float("nan"),
            "true_abs_p90": float("nan"),
            "true_abs_p95": float("nan"),
            "true_abs_p99": float("nan"),
            "pred_to_true_ratio_mean": float("nan"),
            "pred_to_true_ratio_p50": float("nan"),
            "pred_to_true_ratio_p90": float("nan"),
            "pred_to_true_ratio_p95": float("nan"),
            "pred_to_true_ratio_p99": float("nan"),
            "error_abs_mean": float("nan"),
            "error_abs_p90": float("nan"),
            "error_abs_p99": float("nan"),
        }

    err = pred - true
    denom = float(np.linalg.norm(pred) * np.linalg.norm(true)) + 1e-12
    cosine = float(np.dot(pred, true) / denom)
    significant = np.abs(true) > float(near_zero_eps)
    ratio = np.abs(pred[significant]) / np.maximum(np.abs(true[significant]), near_zero_eps)
    sign_ratio = (
        float(np.mean(np.sign(pred[significant]) == np.sign(true[significant])))
        if np.any(significant)
        else float("nan")
    )

    return {
        "count": int(pred.size),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "cosine_similarity": cosine,
        "correlation_coeff": _pearson_corr(pred, true),
        "sign_agreement_ratio": sign_ratio,
        "pred_abs_mean": float(np.mean(np.abs(pred))),
        "pred_abs_p50": _safe_quantile(np.abs(pred), 0.50),
        "pred_abs_p90": _safe_quantile(np.abs(pred), 0.90),
        "pred_abs_p95": _safe_quantile(np.abs(pred), 0.95),
        "pred_abs_p99": _safe_quantile(np.abs(pred), 0.99),
        "true_abs_mean": float(np.mean(np.abs(true))),
        "true_abs_p50": _safe_quantile(np.abs(true), 0.50),
        "true_abs_p90": _safe_quantile(np.abs(true), 0.90),
        "true_abs_p95": _safe_quantile(np.abs(true), 0.95),
        "true_abs_p99": _safe_quantile(np.abs(true), 0.99),
        "pred_to_true_ratio_mean": _safe_mean(ratio),
        "pred_to_true_ratio_p50": _safe_quantile(ratio, 0.50),
        "pred_to_true_ratio_p90": _safe_quantile(ratio, 0.90),
        "pred_to_true_ratio_p95": _safe_quantile(ratio, 0.95),
        "pred_to_true_ratio_p99": _safe_quantile(ratio, 0.99),
        "error_abs_mean": float(np.mean(np.abs(err))),
        "error_abs_p90": _safe_quantile(np.abs(err), 0.90),
        "error_abs_p99": _safe_quantile(np.abs(err), 0.99),
    }


def _region_metrics(pred, true, shock_mask, smooth_mask, *, near_zero_eps):
    transition_mask = ~(shock_mask | smooth_mask)
    return {
        "global": _pair_metrics(pred, true, near_zero_eps=near_zero_eps),
        "shock": _pair_metrics(pred[shock_mask], true[shock_mask], near_zero_eps=near_zero_eps),
        "smooth": _pair_metrics(pred[smooth_mask], true[smooth_mask], near_zero_eps=near_zero_eps),
        "transition": _pair_metrics(pred[transition_mask], true[transition_mask], near_zero_eps=near_zero_eps),
    }


def _mass_by_region(values, shock_mask, smooth_mask):
    abs_values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(np.sum(abs_values)) + 1e-12
    transition_mask = ~(shock_mask | smooth_mask)
    return {
        "shock_mass_frac": float(np.sum(abs_values[shock_mask]) / total),
        "transition_mass_frac": float(np.sum(abs_values[transition_mask]) / total),
        "smooth_mass_frac": float(np.sum(abs_values[smooth_mask]) / total),
        "shock_abs_mean": _safe_mean(abs_values[shock_mask]),
        "transition_abs_mean": _safe_mean(abs_values[transition_mask]),
        "smooth_abs_mean": _safe_mean(abs_values[smooth_mask]),
    }


def _binned_feature_stats(feature, target, *, num_bins=8):
    feature = np.asarray(feature, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(feature, quantiles)
    edges[0] = np.min(feature)
    edges[-1] = np.max(feature)

    rows = []
    for idx in range(num_bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == num_bins - 1:
            mask = (feature >= left) & (feature <= right)
        else:
            mask = (feature >= left) & (feature < right)
        rows.append(
            {
                "bin": idx,
                "left": float(left),
                "right": float(right),
                "count": int(np.sum(mask)),
                "feature_mean": _safe_mean(feature[mask]),
                "target_mean": _safe_mean(target[mask]),
                "target_abs_mean": _safe_mean(np.abs(target[mask])),
                "target_pos_frac": float(np.mean(target[mask] > 0.0)) if np.any(mask) else float("nan"),
            }
        )
    return rows


def _adjacent_sign_agreement(frames, shock_mask_frames, *, near_zero_eps):
    agree = []
    nonzero = []
    for target, shock_mask in zip(frames, shock_mask_frames):
        target = np.asarray(target, dtype=np.float64)
        shock_mask = np.asarray(shock_mask, dtype=bool)
        neighborhood = shock_mask | np.roll(shock_mask, 1) | np.roll(shock_mask, -1)
        sig = np.abs(target) > near_zero_eps
        pair_mask = neighborhood & np.roll(neighborhood, -1) & sig & np.roll(sig, -1)
        if np.any(pair_mask):
            signs = np.sign(target)
            agree.append(float(np.mean(signs[pair_mask] == np.roll(signs, -1)[pair_mask])))
            nonzero.append(int(np.sum(pair_mask)))
    return {
        "adjacent_same_sign_mean": _safe_mean(agree),
        "adjacent_pair_count_mean": _safe_mean(nonzero),
    }


def _time_bucket_summary(step_rows, bucket_edges):
    rows = []
    if not step_rows:
        return rows

    for idx in range(len(bucket_edges) - 1):
        left = float(bucket_edges[idx])
        right = float(bucket_edges[idx + 1])
        if idx == len(bucket_edges) - 2:
            bucket = [row for row in step_rows if left <= row["time_frac"] <= right]
        else:
            bucket = [row for row in step_rows if left <= row["time_frac"] < right]
        rows.append(
            {
                "bucket": f"{left:.2f}-{right:.2f}",
                "count": len(bucket),
                "target_abs_mean": _safe_mean([row["target_abs_mean"] for row in bucket]),
                "target_abs_p90": _safe_mean([row["target_abs_p90"] for row in bucket]),
                "target_abs_p99": _safe_mean([row["target_abs_p99"] for row in bucket]),
                "shock_mass_frac": _safe_mean([row["target_shock_mass_frac"] for row in bucket]),
                "pred_abs_mean": _safe_mean([row["pred_abs_mean"] for row in bucket]),
                "eff_gate_mean": _safe_mean([row["effective_gate_mean"] for row in bucket]),
            }
        )
    return rows


def _correction_stencil_indices(feature_layout, stencil_size):
    if feature_layout and feature_layout.get("correction_stencil_indices") is not None:
        return [int(idx) for idx in feature_layout.get("correction_stencil_indices", [])]
    center = int(stencil_size) // 2
    return list(range(max(center - 2, 0), min(center + 3, int(stencil_size))))


def _extract_feature_columns(X_raw, stencil_size, phys_dim, feature_layout=None):
    X_raw = np.asarray(X_raw, dtype=np.float32)
    phys_names = physics_feature_names(phys_dim)
    features = {}
    for idx, name in enumerate(phys_names):
        features[name] = X_raw[:, stencil_size + idx].astype(np.float64)

    center = stencil_size // 2
    local_cols = np.asarray(_correction_stencil_indices(feature_layout, stencil_size), dtype=np.int64)
    features[f"stencil{local_cols.size}"] = X_raw[:, local_cols].astype(np.float64)
    features["correction_stencil"] = X_raw[:, local_cols].astype(np.float64)
    features["local_center"] = X_raw[:, center].astype(np.float64)
    return features


def _resolve_region_thresholds(feature_values, analysis_cfg):
    shock_threshold = cfg_get(analysis_cfg, "shock_threshold", None)
    transition_threshold = cfg_get(analysis_cfg, "transition_threshold", None)
    if shock_threshold is None:
        shock_threshold = float(np.quantile(feature_values, float(cfg_get(analysis_cfg, "shock_quantile", 0.9))))
    else:
        shock_threshold = float(shock_threshold)
    if transition_threshold is None:
        transition_threshold = float(
            np.quantile(feature_values, float(cfg_get(analysis_cfg, "transition_quantile", 0.75)))
        )
    else:
        transition_threshold = float(transition_threshold)
    transition_threshold = min(transition_threshold, shock_threshold)
    return transition_threshold, shock_threshold


def _build_region_masks(feature_values, transition_threshold, shock_threshold):
    feature_values = np.asarray(feature_values, dtype=np.float64).reshape(-1)
    shock_mask = feature_values >= shock_threshold
    smooth_mask = feature_values < transition_threshold
    return shock_mask, smooth_mask


def _predict_dataset_batches(predictor, X_raw, batch_size):
    pred_list = []
    raw_gate_list = []
    eff_gate_list = []
    shock_sensor_list = []
    corr_before_gate_list = []
    corr_after_gate_list = []
    for start in range(0, X_raw.shape[0], batch_size):
        stop = min(start + batch_size, X_raw.shape[0])
        pred, _, components = predictor.predict_from_inputs_raw(
            X_raw[start:stop],
            return_debug=True,
            return_components=True,
        )
        pred_list.append(pred.astype(np.float64))
        raw_gate_list.append(components["raw_gate"].astype(np.float64))
        eff_gate_list.append(components["effective_gate"].astype(np.float64))
        shock_sensor_list.append(components["shock_sensor"].astype(np.float64))
        corr_before_gate_list.append(components["corr_before_gate"].astype(np.float64))
        corr_after_gate_list.append(components["corr_after_gate"].astype(np.float64))

    return {
        "pred_corr": np.concatenate(pred_list),
        "raw_gate": np.concatenate(raw_gate_list),
        "effective_gate": np.concatenate(eff_gate_list),
        "shock_sensor": np.concatenate(shock_sensor_list),
        "corr_before_gate": np.concatenate(corr_before_gate_list),
        "corr_after_gate": np.concatenate(corr_after_gate_list),
    }


class _ProbeNet(nn.Module):
    def __init__(self, in_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        layers = []
        prev_dim = int(in_dim)
        if not hidden_dims:
            layers.append(nn.Linear(prev_dim, 1))
        else:
            for hidden in hidden_dims:
                layers.append(nn.Linear(prev_dim, int(hidden)))
                layers.append(nn.SiLU())
                prev_dim = int(hidden)
            layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_regression_probe(
    train_X,
    train_y,
    val_X,
    val_y,
    *,
    hidden_dims,
    epochs,
    batch_size,
    learning_rate,
    device,
):
    train_X = np.asarray(train_X, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32).reshape(-1)
    val_X = np.asarray(val_X, dtype=np.float32)
    val_y = np.asarray(val_y, dtype=np.float32).reshape(-1)

    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    train_Xn = (train_X - mean) / std
    val_Xn = (val_X - mean) / std

    dataset = TensorDataset(torch.from_numpy(train_Xn), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _ProbeNet(train_Xn.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    val_X_t = torch.from_numpy(val_Xn).to(device)
    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(val_X_t).cpu().numpy().astype(np.float64)
    return pred


def _select_probe_subsets(train_X, train_y, val_X, val_y, analysis_cfg):
    rng = np.random.default_rng(12345)
    train_cap = min(int(cfg_get(analysis_cfg, "probe_train_points", 60000)), train_X.shape[0])
    val_cap = min(int(cfg_get(analysis_cfg, "probe_val_points", 20000)), val_X.shape[0])
    train_idx = rng.choice(train_X.shape[0], size=train_cap, replace=False)
    val_idx = rng.choice(val_X.shape[0], size=val_cap, replace=False)
    return train_X[train_idx], train_y[train_idx], val_X[val_idx], val_y[val_idx]


def _run_feature_probes(train_X_raw, train_y, val_X_raw, val_y, stencil_size, analysis_cfg, device, feature_layout):
    train_X_probe, train_y_probe, val_X_probe, val_y_probe = _select_probe_subsets(
        train_X_raw, train_y, val_X_raw, val_y, analysis_cfg
    )
    stencil_cols = np.asarray(_correction_stencil_indices(feature_layout, stencil_size), dtype=np.int64)
    phys_train = train_X_probe[:, stencil_size:]
    phys_val = val_X_probe[:, stencil_size:]
    stencil_train = train_X_probe[:, stencil_cols]
    stencil_val = val_X_probe[:, stencil_cols]
    stencil_name = f"stencil{stencil_cols.size}"

    feature_sets = {
        "physics7": (phys_train, phys_val),
        stencil_name: (stencil_train, stencil_val),
        f"physics7_plus_{stencil_name}": (
            np.hstack([phys_train, stencil_train]),
            np.hstack([phys_val, stencil_val]),
        ),
    }

    sign_threshold = float(
        np.quantile(
            np.abs(train_y_probe.reshape(-1)),
            float(cfg_get(analysis_cfg, "sign_probe_quantile", 0.5)),
        )
    )
    results = {}
    for name, (probe_train_X, probe_val_X) in feature_sets.items():
        linear_pred = _train_regression_probe(
            probe_train_X,
            train_y_probe,
            probe_val_X,
            val_y_probe,
            hidden_dims=[],
            epochs=max(4, int(cfg_get(analysis_cfg, "probe_epochs", 20)) // 2),
            batch_size=int(cfg_get(analysis_cfg, "probe_batch_size", 4096)),
            learning_rate=float(cfg_get(analysis_cfg, "probe_learning_rate", 1e-3)),
            device=device,
        )
        mlp_pred = _train_regression_probe(
            probe_train_X,
            train_y_probe,
            probe_val_X,
            val_y_probe,
            hidden_dims=[64, 32],
            epochs=int(cfg_get(analysis_cfg, "probe_epochs", 20)),
            batch_size=int(cfg_get(analysis_cfg, "probe_batch_size", 4096)),
            learning_rate=float(cfg_get(analysis_cfg, "probe_learning_rate", 1e-3)),
            device=device,
        )

        significant = np.abs(val_y_probe.reshape(-1)) >= sign_threshold
        results[name] = {
            "linear": _pair_metrics(linear_pred, val_y_probe.reshape(-1), near_zero_eps=1e-12),
            "mlp": _pair_metrics(mlp_pred, val_y_probe.reshape(-1), near_zero_eps=1e-12),
            "sign_threshold": sign_threshold,
            "linear_sign_agreement_significant": float(
                np.mean(np.sign(linear_pred[significant]) == np.sign(val_y_probe.reshape(-1)[significant]))
            ),
            "mlp_sign_agreement_significant": float(
                np.mean(np.sign(mlp_pred[significant]) == np.sign(val_y_probe.reshape(-1)[significant]))
            ),
        }
    return results


def _instantiate_predictor(cfg, device, gate_mode):
    return KANPredictor(
        model_path=str(Path(cfg_get(cfg, "paths.model_save_path", "kan_model.pth"))),
        device=device,
        model_cfg=cfg.get("model", {}),
        scaler_cfg=cfg.get("scaler", {}),
        gate_mode=str(gate_mode),
        hard_gate_sensor_threshold=float(cfg_get(cfg, "ablation.hard_gate_sensor_threshold", 0.1)),
        soft_mask_center=float(cfg_get(cfg, "ablation.soft_mask_center", 0.1)),
        soft_mask_width=float(cfg_get(cfg, "ablation.soft_mask_width", 0.08)),
        soft_mask_floor=float(cfg_get(cfg, "ablation.soft_mask_floor", 0.02)),
        correction_clip_abs=cfg_get(cfg, "ablation.correction_clip_abs", None),
        debug_rollout=False,
        debug_t_start=float(cfg_get(cfg, "logging.debug_t_start", 1.0)),
        metadata_cfg=cfg.get("metadata", {}),
    )


def _collect_rollout_trace(cfg, predictor, analysis_cfg, *, gate_mode, correction_scale, alpha):
    eval_cfg = cfg.get("evaluation", {})
    solver_cfg = cfg.get("solver", {})
    ab_cfg = cfg.get("ablation", {})

    N_coarse = int(eval_cfg.get("N_coarse", 128))
    N_ref = int(eval_cfg.get("N_ref", 2048))
    T_final = float(eval_cfg.get("T_final", 1.5))
    cfl = float(eval_cfg.get("cfl", 0.4))
    remove_correction_mean = bool(eval_cfg.get("remove_correction_mean", False))
    ic_cfg = eval_cfg.get("initial_condition", {})
    nu = float(solver_cfg.get("nu", 0.0))
    weno_epsilon = float(solver_cfg.get("weno_epsilon", 1e-6))
    correction_sign_mode = str(ab_cfg.get("correction_sign_mode", "auto"))
    resolved_sign_mode, sign_factor = _resolve_correction_sign_mode(correction_sign_mode, predictor.metadata)
    effective_applied_scale = float(correction_scale) * float(alpha)

    ref_substeps = require_integer_refinement(N_ref, N_coarse)
    x = np.linspace(0, 2 * np.pi, N_ref, endpoint=False)
    u_ref_init = (
        float(ic_cfg.get("sin1_amp", 1.0)) * np.sin(x)
        + float(ic_cfg.get("sin2_amp", 0.5)) * np.sin(2 * x + float(ic_cfg.get("sin2_phase", 0.5)))
        + float(ic_cfg.get("cos5_amp", -0.2)) * np.cos(5 * x)
    )
    u_ref_init /= np.max(np.abs(u_ref_init))

    u_coarse = downsample_periodic(u_ref_init, ratio=ref_substeps)
    u_hybrid = u_coarse.copy()
    u_truth = u_ref_init.copy()
    dx_coarse = 2 * np.pi / N_coarse
    dx_ref = 2 * np.pi / N_ref
    t = 0.0

    target_feature_name = str(cfg_get(analysis_cfg, "shock_feature", "abs_u_x"))
    near_zero_eps = float(cfg_get(analysis_cfg, "target_near_zero_epsilon", 1e-6))
    step_rows = []
    true_frames = []
    shock_frames = []
    pred_frames = []
    applied_frames = []
    raw_gate_frames = []
    eff_gate_frames = []

    while t < T_final - 1e-12:
        dt = cfl * dx_coarse / (max(np.max(np.abs(u_coarse)), np.max(np.abs(u_hybrid))) + 1e-6)
        if t + dt > T_final:
            dt = T_final - t

        for _ in range(ref_substeps):
            u_truth = rk3_step(u_truth, dx_ref, dt / ref_substeps, nu=nu, weno_epsilon=weno_epsilon)

        u_coarse = rk3_step(u_coarse, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)
        u_weno_next = rk3_step(u_hybrid, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)
        inputs_raw = build_model_inputs(
            u_hybrid,
            stencil_size=predictor.stencil_size,
            dx=dx_coarse,
            dt=dt,
            t_stamp=t,
            phys_dim=predictor.phys_dim,
        )
        pred_corr, _, components = predictor.predict_from_inputs_raw(
            inputs_raw,
            return_debug=True,
            return_components=True,
        )
        if remove_correction_mean:
            pred_corr = pred_corr - np.mean(pred_corr)
        applied_corr = effective_applied_scale * sign_factor * pred_corr

        u_truth_down = downsample_periodic(u_truth, ratio=ref_substeps)
        true_corr = u_truth_down - u_weno_next
        feature_names = physics_feature_names(predictor.phys_dim)
        feature_idx = predictor.stencil_size + feature_names.index(target_feature_name)
        shock_feature = inputs_raw[:, feature_idx]
        transition_threshold, shock_threshold = _resolve_region_thresholds(shock_feature, analysis_cfg)
        shock_mask, smooth_mask = _build_region_masks(shock_feature, transition_threshold, shock_threshold)
        target_mass = _mass_by_region(true_corr, shock_mask, smooth_mask)

        step_rows.append(
            {
                "time": float(t + dt),
                "time_frac": float((t + dt) / max(T_final, 1e-12)),
                "gate_mode": gate_mode,
                "correction_scale": float(correction_scale),
                "alpha": float(alpha),
                "effective_applied_scale": float(effective_applied_scale),
                "resolved_sign_mode": resolved_sign_mode,
                "target_abs_mean": float(np.mean(np.abs(true_corr))),
                "target_abs_p90": _safe_quantile(np.abs(true_corr), 0.90),
                "target_abs_p99": _safe_quantile(np.abs(true_corr), 0.99),
                "target_shock_mass_frac": target_mass["shock_mass_frac"],
                "pred_abs_mean": float(np.mean(np.abs(pred_corr))),
                "pred_error_abs_mean": float(np.mean(np.abs(pred_corr - true_corr))),
                "raw_gate_mean": float(np.mean(components["raw_gate"])),
                "effective_gate_mean": float(np.mean(components["effective_gate"])),
                "smooth_applied_abs_mean": _safe_mean(np.abs(applied_corr[smooth_mask])),
                "shock_applied_abs_mean": _safe_mean(np.abs(applied_corr[shock_mask])),
                "l2_base": float(np.sqrt(np.mean((u_coarse - u_truth_down) ** 2))),
            }
        )

        true_frames.append(true_corr.astype(np.float64))
        shock_frames.append(shock_mask.astype(bool))
        pred_frames.append(pred_corr.astype(np.float64))
        applied_frames.append(applied_corr.astype(np.float64))
        raw_gate_frames.append(components["raw_gate"].astype(np.float64))
        eff_gate_frames.append(components["effective_gate"].astype(np.float64))

        u_hybrid = u_weno_next + applied_corr
        step_rows[-1]["l2_hybrid"] = float(np.sqrt(np.mean((u_hybrid - u_truth_down) ** 2)))
        t += dt

    return {
        "step_rows": step_rows,
        "correction_scale": float(correction_scale),
        "alpha": float(alpha),
        "effective_applied_scale": float(effective_applied_scale),
        "final_l2_base": float(step_rows[-1]["l2_base"]),
        "final_l2_hybrid": float(step_rows[-1]["l2_hybrid"]),
        "adjacent_sign": _adjacent_sign_agreement(true_frames, shock_frames, near_zero_eps=near_zero_eps),
        "time_buckets": _time_bucket_summary(step_rows, cfg_get(analysis_cfg, "time_bucket_edges", [0.0, 0.33, 0.66, 1.0])),
    }


def _format_metric_table(rows, headers):
    widths = []
    for idx, header in enumerate(headers):
        widths.append(max(len(header), max(len(str(row[idx])) for row in rows) if rows else 0))
    parts = []
    parts.append(" | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)))
    parts.append("-+-".join("-" * width for width in widths))
    for row in rows:
        parts.append(" | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))
    return "\n".join(parts)


def _stats_row(label, stats):
    return [
        label,
        f"{stats['mean']:.3e}",
        f"{stats['std']:.3e}",
        f"{stats['abs_mean']:.3e}",
        f"{stats['abs_max']:.3e}",
        f"{stats['abs_p50']:.3e}",
        f"{stats['abs_p90']:.3e}",
        f"{stats['abs_p95']:.3e}",
        f"{stats['abs_p99']:.3e}",
    ]


def _pair_row(label, stats):
    return [
        label,
        f"{stats['count']}",
        f"{stats['mse']:.3e}",
        f"{stats['mae']:.3e}",
        f"{stats['cosine_similarity']:.3f}",
        f"{stats['correlation_coeff']:.3f}",
        f"{stats['sign_agreement_ratio']:.3f}",
        f"{stats['pred_abs_mean']:.3e}",
        f"{stats['true_abs_mean']:.3e}",
        f"{stats['pred_to_true_ratio_mean']:.3f}",
    ]


def _gate_mode_row(mode, stats, rollout):
    return [
        mode,
        f"{stats['global']['mae']:.3e}",
        f"{stats['shock']['mae']:.3e}",
        f"{stats['smooth']['pred_abs_mean']:.3e}",
        f"{stats['global']['cosine_similarity']:.3f}",
        f"{rollout['final_l2_hybrid']:.3e}",
        f"{rollout['final_l2_hybrid'] / (rollout['final_l2_base'] + 1e-12):.3f}",
    ]


def _alpha_row(alpha, rollout):
    return [
        f"{alpha:.2f}",
        f"{rollout['effective_applied_scale']:.2f}",
        f"{rollout['final_l2_base']:.3e}",
        f"{rollout['final_l2_hybrid']:.3e}",
        f"{rollout['final_l2_hybrid'] / (rollout['final_l2_base'] + 1e-12):.3f}",
        f"{_safe_mean([row['pred_abs_mean'] for row in rollout['step_rows']]):.3e}",
        f"{_safe_mean([row['smooth_applied_abs_mean'] for row in rollout['step_rows']]):.3e}",
    ]


def _write_plots(plot_path, report_data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    target_abs = np.abs(report_data["val_target"])
    axes[0, 0].hist(np.clip(target_abs, 0.0, np.quantile(target_abs, 0.999)), bins=60, color="steelblue")
    axes[0, 0].set_title("|True one-step correction|")
    axes[0, 0].set_yscale("log")

    binned = report_data["feature_binned"]["abs_u_x"]
    axes[0, 1].plot([row["feature_mean"] for row in binned], [row["target_abs_mean"] for row in binned], marker="o")
    axes[0, 1].set_title("Binned abs_u_x vs |target|")
    axes[0, 1].set_xlabel("abs_u_x mean")
    axes[0, 1].set_ylabel("|target| mean")

    time_buckets = report_data["time_buckets"]
    axes[0, 2].bar(np.arange(len(time_buckets)), [row["target_abs_mean"] for row in time_buckets], color="darkorange")
    axes[0, 2].set_xticks(np.arange(len(time_buckets)))
    axes[0, 2].set_xticklabels([row["bucket"] for row in time_buckets], rotation=20)
    axes[0, 2].set_title("Target magnitude over rollout time")

    gate_modes = list(report_data["gate_mode_metrics"].keys())
    gate_rollout_ratios = [
        report_data["gate_mode_rollouts"][mode]["final_l2_hybrid"]
        / (report_data["gate_mode_rollouts"][mode]["final_l2_base"] + 1e-12)
        for mode in gate_modes
    ]
    axes[1, 0].bar(np.arange(len(gate_modes)), gate_rollout_ratios, color="seagreen")
    axes[1, 0].set_xticks(np.arange(len(gate_modes)))
    axes[1, 0].set_xticklabels(gate_modes, rotation=20)
    axes[1, 0].set_title("Gate mode rollout ratio")

    alpha_values = report_data["alpha_values"]
    alpha_ratios = [
        report_data["alpha_rollouts"][f"{alpha:.2f}"]["final_l2_hybrid"]
        / (report_data["alpha_rollouts"][f"{alpha:.2f}"]["final_l2_base"] + 1e-12)
        for alpha in alpha_values
    ]
    axes[1, 1].plot(alpha_values, alpha_ratios, marker="o", color="firebrick")
    axes[1, 1].set_title("Rollout sensitivity to alpha")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("final hybrid/base ratio")

    shock_gate = report_data["gate_summary"]
    axes[1, 2].bar(
        ["raw-shock", "raw-smooth", "eff-shock", "eff-smooth"],
        [
            shock_gate["raw_gate_shock_mean"],
            shock_gate["raw_gate_smooth_mean"],
            shock_gate["effective_gate_shock_mean"],
            shock_gate["effective_gate_smooth_mean"],
        ],
        color=["navy", "lightblue", "darkgreen", "lightgreen"],
    )
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_title("Gate selectivity")

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=140)
    plt.close(fig)


def _generate_report_text(report_data):
    target_table = _format_metric_table(
        [
            _stats_row("global", report_data["target_stats"]["global"]),
            _stats_row("shock", report_data["target_stats"]["shock"]),
            _stats_row("smooth", report_data["target_stats"]["smooth"]),
        ],
        ["region", "mean", "std", "|.| mean", "|.| max", "p50|.|", "p90|.|", "p95|.|", "p99|.|"],
    )
    pred_table = _format_metric_table(
        [
            _pair_row("global", report_data["prediction_metrics"]["global"]),
            _pair_row("shock", report_data["prediction_metrics"]["shock"]),
            _pair_row("smooth", report_data["prediction_metrics"]["smooth"]),
        ],
        ["region", "count", "MSE", "MAE", "cos", "corr", "sign", "|pred| mean", "|true| mean", "|pred|/|true|"],
    )
    gate_mode_table = _format_metric_table(
        [
            _gate_mode_row(mode, report_data["gate_mode_metrics"][mode], report_data["gate_mode_rollouts"][mode])
            for mode in report_data["gate_mode_order"]
        ],
        ["gate_mode", "global MAE", "shock MAE", "smooth |pred|", "global cos", "rollout L2", "hybrid/base"],
    )
    alpha_table = _format_metric_table(
        [_alpha_row(alpha, report_data["alpha_rollouts"][f"{alpha:.2f}"]) for alpha in report_data["alpha_values"]],
        ["alpha", "eff scale", "base L2", "hybrid L2", "hybrid/base", "mean |pred|", "smooth |applied|"],
    )

    feature_rows = []
    for name, corr_stats in report_data["feature_correlations"].items():
        feature_rows.append(
            [
                name,
                f"{corr_stats['pearson_target']:.3f}",
                f"{corr_stats['spearman_target']:.3f}",
                f"{corr_stats['pearson_abs_target']:.3f}",
                f"{corr_stats['spearman_abs_target']:.3f}",
                f"{corr_stats['shock_pearson_target']:.3f}",
            ]
        )
    feature_table = _format_metric_table(
        feature_rows,
        ["feature", "corr(target)", "rho(target)", "corr(|target|)", "rho(|target|)", "shock corr(target)"],
    )

    probe_rows = []
    for name, stats in report_data["probe_results"].items():
        probe_rows.append(
            [
                name,
                f"{stats['linear']['mae']:.3e}",
                f"{stats['mlp']['mae']:.3e}",
                f"{stats['linear']['correlation_coeff']:.3f}",
                f"{stats['mlp']['correlation_coeff']:.3f}",
                f"{stats['linear_sign_agreement_significant']:.3f}",
                f"{stats['mlp_sign_agreement_significant']:.3f}",
            ]
        )
    probe_table = _format_metric_table(
        probe_rows,
        ["probe set", "linear MAE", "mlp MAE", "linear corr", "mlp corr", "linear sign", "mlp sign"],
    )

    time_table = _format_metric_table(
        [
            [
                row["bucket"],
                f"{row['count']}",
                f"{row['target_abs_mean']:.3e}",
                f"{row['target_abs_p90']:.3e}",
                f"{row['target_abs_p99']:.3e}",
                f"{row['shock_mass_frac']:.3f}",
                f"{row['eff_gate_mean']:.3f}",
            ]
            for row in report_data["time_buckets"]
        ],
        ["bucket", "steps", "|target| mean", "|target| p90", "|target| p99", "shock mass", "eff gate"],
    )

    main_conclusions = "\n".join(f"- {item}" for item in report_data["main_conclusions"])
    next_steps = "\n".join(f"- {item}" for item in report_data["recommended_changes"])
    if report_data["output_cap"]["head_is_bounded"]:
        output_range_lines = (
            f"- Output-head mode: {report_data['output_cap']['head_output_mode']} "
            f"(scale={report_data['output_cap']['head_output_scale']:.3f}, bounded=yes)\n"
            f"- Output-range check: target scaler std={report_data['output_cap']['target_scaler_std']:.3e}, "
            f"physical |cap|={report_data['output_cap']['physical_abs_cap']:.3e}, "
            f"observed |pred| max={report_data['output_cap']['pred_abs_max']:.3e}\n"
            f"- Fraction of targets above that cap: global={report_data['output_cap']['true_above_cap_frac']:.3f}, "
            f"shock={report_data['output_cap']['shock_true_above_cap_frac']:.3f}"
        )
    else:
        output_range_lines = (
            f"- Output-head mode: {report_data['output_cap']['head_output_mode']} "
            f"(scale={report_data['output_cap']['head_output_scale']:.3f}, bounded=no)\n"
            f"- Output-range check: unbounded normalized head; target scaler std={report_data['output_cap']['target_scaler_std']:.3e}, "
            f"observed |pred| max={report_data['output_cap']['pred_abs_max']:.3e}"
        )
    feature_layout_summary = report_data["feature_layout"]

    return f"""# Preliminary Investigation Report

## Summary of inspected components
- Dataset generation: `data/generate.py`
- Training path: `train.py`
- Rollout/evaluation path: `evaluate.py`
- Feature construction: `utils/features.py`
- Model/gate path: `kan/model.py`
- Feature layout: stencil_size={feature_layout_summary['stencil_size']}, correction_indices={feature_layout_summary['correction_stencil_indices']}, gate_stencil={feature_layout_summary['gate_use_stencil_features']}
- Correction head: mode={report_data['output_cap']['head_output_mode']}, scale={report_data['output_cap']['head_output_scale']:.3f}, eval correction_scale={report_data['evaluation_correction_scale']:.3f}

## Target characteristics
```
{target_table}
```
- Shock region threshold on `{report_data['shock_feature']}`: transition < {report_data['transition_threshold']:.3e}, shock >= {report_data['shock_threshold']:.3e}
- Target mass fractions: shock={report_data['target_mass']['shock_mass_frac']:.3f}, transition={report_data['target_mass']['transition_mass_frac']:.3f}, smooth={report_data['target_mass']['smooth_mass_frac']:.3f}
- Sign distribution: positive={report_data['sign_summary']['positive_frac']:.3f}, negative={report_data['sign_summary']['negative_frac']:.3f}, near-zero={report_data['sign_summary']['near_zero_frac']:.3f}
- Near-shock adjacent same-sign ratio on rollout trace: {report_data['rollout_sign_summary']['adjacent_same_sign_mean']:.3f}

### Time dependence
```
{time_table}
```

## Feature sufficiency
```
{feature_table}
```
```
{probe_table}
```
- `abs_u_x` binned |target| mean rises from {report_data['feature_binned']['abs_u_x'][0]['target_abs_mean']:.3e} to {report_data['feature_binned']['abs_u_x'][-1]['target_abs_mean']:.3e}
- `abs_u_xx` binned |target| mean rises from {report_data['feature_binned']['abs_u_xx'][0]['target_abs_mean']:.3e} to {report_data['feature_binned']['abs_u_xx'][-1]['target_abs_mean']:.3e}
- `grad_var` binned |target| mean rises from {report_data['feature_binned']['grad_var'][0]['target_abs_mean']:.3e} to {report_data['feature_binned']['grad_var'][-1]['target_abs_mean']:.3e}

## One-step prediction quality
```
{pred_table}
```
- Global mean |pred|/|true| ratio: {report_data['prediction_metrics']['global']['pred_to_true_ratio_mean']:.3f}
- Shock mean |pred|/|true| ratio: {report_data['prediction_metrics']['shock']['pred_to_true_ratio_mean']:.3f}
- Smooth-region predicted |corr| mean: {report_data['prediction_metrics']['smooth']['pred_abs_mean']:.3e}
- Smooth-region true |corr| mean: {report_data['prediction_metrics']['smooth']['true_abs_mean']:.3e}
{output_range_lines}

## Gate behavior
- Raw gate mean/min/max: {report_data['gate_summary']['raw_gate_mean']:.3f} / {report_data['gate_summary']['raw_gate_min']:.3f} / {report_data['gate_summary']['raw_gate_max']:.3f}
- Effective gate mean/min/max: {report_data['gate_summary']['effective_gate_mean']:.3f} / {report_data['gate_summary']['effective_gate_min']:.3f} / {report_data['gate_summary']['effective_gate_max']:.3f}
- Shock vs smooth raw gate mean: {report_data['gate_summary']['raw_gate_shock_mean']:.3f} vs {report_data['gate_summary']['raw_gate_smooth_mean']:.3f}
- Shock vs smooth effective gate mean: {report_data['gate_summary']['effective_gate_shock_mean']:.3f} vs {report_data['gate_summary']['effective_gate_smooth_mean']:.3f}
- Gate correlation with |target|: raw={report_data['gate_summary']['raw_gate_abs_target_corr']:.3f}, effective={report_data['gate_summary']['effective_gate_abs_target_corr']:.3f}
- Gate correlation with |prediction error|: raw={report_data['gate_summary']['raw_gate_abs_error_corr']:.3f}, effective={report_data['gate_summary']['effective_gate_abs_error_corr']:.3f}
```
{gate_mode_table}
```

## Localization vs smooth-region pollution
- True correction mass fractions: shock={report_data['target_mass']['shock_mass_frac']:.3f}, transition={report_data['target_mass']['transition_mass_frac']:.3f}, smooth={report_data['target_mass']['smooth_mass_frac']:.3f}
- Predicted correction mass fractions: shock={report_data['pred_mass']['shock_mass_frac']:.3f}, transition={report_data['pred_mass']['transition_mass_frac']:.3f}, smooth={report_data['pred_mass']['smooth_mass_frac']:.3f}
- Applied correction mass fractions: shock={report_data['applied_mass']['shock_mass_frac']:.3f}, transition={report_data['applied_mass']['transition_mass_frac']:.3f}, smooth={report_data['applied_mass']['smooth_mass_frac']:.3f}
- Support threshold based on true p90(|corr|) = {report_data['support_threshold']:.3e}
- Active support fraction above true p90 threshold: true={report_data['support_summary']['true_active_frac']:.3f}, pred={report_data['support_summary']['pred_active_frac']:.3f}, applied={report_data['support_summary']['applied_active_frac']:.3f}
- Smooth active support fraction above true p90 threshold: true={report_data['support_summary']['true_smooth_active_frac']:.3f}, pred={report_data['support_summary']['pred_smooth_active_frac']:.3f}, applied={report_data['support_summary']['applied_smooth_active_frac']:.3f}

## Rollout sensitivity to correction scale
```
{alpha_table}
```
- Base evaluation correction_scale={report_data['evaluation_correction_scale']:.3f}; table `eff scale` = correction_scale * alpha

## Main conclusions
{main_conclusions}

## Recommended next architecture adjustments
{next_steps}
"""


def run_analysis(cfg):
    runtime_cfg = cfg.get("runtime", {})
    analysis_cfg = cfg.get("analysis", {})
    path_cfg = cfg.get("paths", {})
    device = _resolve_device(runtime_cfg)
    set_global_seed(runtime_cfg.get("seed", None), bool(runtime_cfg.get("deterministic", False)))

    train_data_path = Path(path_cfg.get("train_data_path", "kan_train_data.npz"))
    with np.load(str(train_data_path), allow_pickle=False) as data:
        X_raw = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32).reshape(-1)
        stencil_size = int(data["stencil_size"])
        phys_dim = int(data["phys_dim"])
        extract_dataset_metadata(data, source_label=str(train_data_path), warn_on_missing=True)

    indices = np.arange(X_raw.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(cfg_get(cfg, "data.split_test_size", 0.2)),
        random_state=int(cfg_get(cfg, "data.split_random_state", 42)),
    )
    X_train, y_train = X_raw[train_idx], y[train_idx]
    X_val, y_val = X_raw[val_idx], y[val_idx]

    predictor = _instantiate_predictor(cfg, device, gate_mode=str(cfg_get(cfg, "ablation.gate_mode", "train_equivalent")))
    feature_layout = dict(getattr(predictor, "feature_layout", {}))
    feature_columns = _extract_feature_columns(X_val, stencil_size, phys_dim, feature_layout=feature_layout)
    shock_feature = str(cfg_get(analysis_cfg, "shock_feature", "abs_u_x"))
    if shock_feature not in feature_columns:
        raise ValueError(f"Unsupported shock feature for analysis: {shock_feature}")
    transition_threshold, shock_threshold = _resolve_region_thresholds(feature_columns[shock_feature], analysis_cfg)
    shock_mask, smooth_mask = _build_region_masks(feature_columns[shock_feature], transition_threshold, shock_threshold)
    near_zero_eps = float(cfg_get(analysis_cfg, "target_near_zero_epsilon", 1e-6))

    val_pred = _predict_dataset_batches(predictor, X_val, int(cfg_get(analysis_cfg, "eval_batch_size", 32768)))
    pred_corr = val_pred["pred_corr"]
    raw_gate = val_pred["raw_gate"]
    effective_gate = val_pred["effective_gate"]
    evaluation_correction_scale = float(cfg_get(cfg, "evaluation.correction_scale", 0.25))
    applied_corr = evaluation_correction_scale * pred_corr.copy()
    abs_error = np.abs(pred_corr - y_val)
    head_output_mode = str(predictor.model_runtime_cfg["correction_head_output_mode"])
    head_output_scale = float(predictor.model_runtime_cfg["correction_head_output_scale"])
    head_is_bounded = bool(predictor.model.correction_head_is_bounded)
    physical_abs_cap = (
        float(abs(predictor.target_scaler.std) * head_output_scale) if head_is_bounded else None
    )

    target_stats = {
        "global": _summary_stats(y_val),
        "shock": _summary_stats(y_val[shock_mask]),
        "smooth": _summary_stats(y_val[smooth_mask]),
    }
    target_mass = _mass_by_region(y_val, shock_mask, smooth_mask)
    sign_summary = {
        "positive_frac": float(np.mean(y_val > near_zero_eps)),
        "negative_frac": float(np.mean(y_val < -near_zero_eps)),
        "near_zero_frac": float(np.mean(np.abs(y_val) <= near_zero_eps)),
        "shock_positive_frac": float(np.mean(y_val[shock_mask] > near_zero_eps)),
        "shock_negative_frac": float(np.mean(y_val[shock_mask] < -near_zero_eps)),
    }

    feature_correlations = {}
    for name in physics_feature_names(phys_dim):
        feat = feature_columns[name]
        feature_correlations[name] = {
            "pearson_target": _pearson_corr(feat, y_val),
            "spearman_target": _spearman_corr(feat, y_val),
            "pearson_abs_target": _pearson_corr(feat, np.abs(y_val)),
            "spearman_abs_target": _spearman_corr(feat, np.abs(y_val)),
            "shock_pearson_target": _pearson_corr(feat[shock_mask], y_val[shock_mask]),
        }

    feature_binned = {
        "abs_u_x": _binned_feature_stats(feature_columns["abs_u_x"], y_val),
        "abs_u_xx": _binned_feature_stats(feature_columns["abs_u_xx"], y_val),
        "grad_var": _binned_feature_stats(feature_columns["grad_var"], y_val),
    }

    probe_results = _run_feature_probes(
        X_train,
        y_train,
        X_val,
        y_val,
        stencil_size,
        analysis_cfg,
        device,
        feature_layout,
    )
    prediction_metrics = _region_metrics(pred_corr, y_val, shock_mask, smooth_mask, near_zero_eps=near_zero_eps)

    gate_summary = {
        "raw_gate_mean": float(np.mean(raw_gate)),
        "raw_gate_min": float(np.min(raw_gate)),
        "raw_gate_max": float(np.max(raw_gate)),
        "effective_gate_mean": float(np.mean(effective_gate)),
        "effective_gate_min": float(np.min(effective_gate)),
        "effective_gate_max": float(np.max(effective_gate)),
        "raw_gate_shock_mean": _safe_mean(raw_gate[shock_mask]),
        "raw_gate_smooth_mean": _safe_mean(raw_gate[smooth_mask]),
        "effective_gate_shock_mean": _safe_mean(effective_gate[shock_mask]),
        "effective_gate_smooth_mean": _safe_mean(effective_gate[smooth_mask]),
        "raw_gate_abs_target_corr": _pearson_corr(raw_gate, np.abs(y_val)),
        "effective_gate_abs_target_corr": _pearson_corr(effective_gate, np.abs(y_val)),
        "raw_gate_abs_error_corr": _pearson_corr(raw_gate, abs_error),
        "effective_gate_abs_error_corr": _pearson_corr(effective_gate, abs_error),
    }
    output_cap = {
        "head_output_mode": head_output_mode,
        "head_output_scale": head_output_scale,
        "head_is_bounded": head_is_bounded,
        "target_scaler_std": float(predictor.target_scaler.std),
        "physical_abs_cap": physical_abs_cap,
        "pred_abs_max": float(np.max(np.abs(pred_corr))),
        "true_above_cap_frac": float(np.mean(np.abs(y_val) >= physical_abs_cap)) if head_is_bounded else None,
        "shock_true_above_cap_frac": float(np.mean(np.abs(y_val[shock_mask]) >= physical_abs_cap)) if head_is_bounded else None,
    }

    gate_mode_metrics = {}
    gate_mode_rollouts = {}
    gate_mode_order = [str(item) for item in cfg_get(analysis_cfg, "gate_modes", ["train_equivalent"])]
    for mode in gate_mode_order:
        mode_predictor = _instantiate_predictor(cfg, device, gate_mode=mode)
        mode_val = _predict_dataset_batches(mode_predictor, X_val, int(cfg_get(analysis_cfg, "eval_batch_size", 32768)))
        gate_mode_metrics[mode] = _region_metrics(mode_val["pred_corr"], y_val, shock_mask, smooth_mask, near_zero_eps=near_zero_eps)
        gate_mode_rollouts[mode] = _collect_rollout_trace(
            cfg,
            mode_predictor,
            analysis_cfg,
            gate_mode=mode,
            correction_scale=evaluation_correction_scale,
            alpha=1.0,
        )

    pred_mass = _mass_by_region(pred_corr, shock_mask, smooth_mask)
    applied_mass = _mass_by_region(applied_corr, shock_mask, smooth_mask)
    support_threshold = float(np.quantile(np.abs(y_val), float(cfg_get(analysis_cfg, "significant_support_quantile", 0.9))))
    support_summary = {
        "true_active_frac": float(np.mean(np.abs(y_val) >= support_threshold)),
        "pred_active_frac": float(np.mean(np.abs(pred_corr) >= support_threshold)),
        "applied_active_frac": float(np.mean(np.abs(applied_corr) >= support_threshold)),
        "true_smooth_active_frac": float(np.mean(np.abs(y_val[smooth_mask]) >= support_threshold)),
        "pred_smooth_active_frac": float(np.mean(np.abs(pred_corr[smooth_mask]) >= support_threshold)),
        "applied_smooth_active_frac": float(np.mean(np.abs(applied_corr[smooth_mask]) >= support_threshold)),
    }

    default_rollout = gate_mode_rollouts[gate_mode_order[0]]
    alpha_values = [float(v) for v in cfg_get(analysis_cfg, "alpha_values", [0.0, 0.25, 0.5, 0.75, 1.0])]
    alpha_gate_mode = str(cfg_get(analysis_cfg, "alpha_gate_mode", gate_mode_order[0]))
    alpha_predictor = _instantiate_predictor(cfg, device, gate_mode=alpha_gate_mode)
    alpha_rollouts = {
        f"{alpha:.2f}": _collect_rollout_trace(
            cfg,
            alpha_predictor,
            analysis_cfg,
            gate_mode=alpha_gate_mode,
            correction_scale=evaluation_correction_scale,
            alpha=alpha,
        )
        for alpha in alpha_values
    }

    main_conclusions = []
    if target_mass["shock_mass_frac"] > 0.5:
        main_conclusions.append(f"The one-step target is strongly localized: {target_mass['shock_mass_frac']:.1%} of |target| mass lies in shock regions.")
    else:
        main_conclusions.append(f"The target is not strongly sparse by the chosen gradient mask: shock mass is {target_mass['shock_mass_frac']:.1%}.")
    if prediction_metrics["shock"]["mae"] > prediction_metrics["smooth"]["mae"]:
        main_conclusions.append("Prediction error remains larger in shock regions than in smooth regions, so the model is not accurately resolving the highest-value correction zones.")
    if prediction_metrics["global"]["pred_abs_mean"] < prediction_metrics["global"]["true_abs_mean"] * 0.8:
        main_conclusions.append(
            f"Predicted correction magnitude is too small where it matters: global |pred| mean is {prediction_metrics['global']['pred_abs_mean']:.3e} vs true {prediction_metrics['global']['true_abs_mean']:.3e}, and shock p99 is {prediction_metrics['shock']['pred_abs_p99']:.3e} vs true {prediction_metrics['shock']['true_abs_p99']:.3e}."
        )
    elif prediction_metrics["global"]["pred_to_true_ratio_mean"] > 1.1:
        main_conclusions.append(
            f"The mean |pred|/|true| ratio is inflated by near-zero targets ({prediction_metrics['global']['pred_to_true_ratio_mean']:.2f}); use tail and regional stats instead of that global mean alone."
        )
    if output_cap["head_is_bounded"] and output_cap["shock_true_above_cap_frac"] > 0.2:
        main_conclusions.append(
            f"The corrective head is range-limited: about {output_cap['shock_true_above_cap_frac']:.1%} of shock targets exceed the current +/-{output_cap['physical_abs_cap']:.3e} physical output range."
        )
    if gate_summary["effective_gate_shock_mean"] <= gate_summary["effective_gate_smooth_mean"] + 0.02:
        main_conclusions.append("The gate is weakly selective: effective gate values are not materially larger in shock regions.")
    else:
        main_conclusions.append(f"The gate is selective but incomplete: effective gate mean is {gate_summary['effective_gate_shock_mean']:.3f} in shocks vs {gate_summary['effective_gate_smooth_mean']:.3f} in smooth regions.")
    best_alpha_key = min(alpha_rollouts, key=lambda key: alpha_rollouts[key]["final_l2_hybrid"])
    if best_alpha_key != "1.00":
        main_conclusions.append(f"Rollout sensitivity is strong: alpha={best_alpha_key} beats the current alpha=1.00 on the standard evaluation rollout.")
    else:
        main_conclusions.append("Global scaling alone does not rescue rollout quality; alpha=1.00 is already near the best tested value.")

    recommended_changes = []
    probe_phys = probe_results["physics7"]["mlp"]["mae"]
    correction_probe_key = f"physics7_plus_stencil{len(feature_layout.get('correction_stencil_indices', []))}"
    probe_phys_stencil = probe_results[correction_probe_key]["mlp"]["mae"]
    if probe_phys_stencil < probe_phys * 0.95:
        recommended_changes.append("Prioritize architectures that expose explicit local stencil values to the corrective head or gating path; the probe shows measurable gain over physics features alone.")
    else:
        recommended_changes.append("Do not widen the feature set blindly; the probe suggests physics features already capture most of the recoverable one-step signal.")
    if gate_mode_rollouts["gate_open"]["final_l2_hybrid"] < gate_mode_rollouts["train_equivalent"]["final_l2_hybrid"]:
        recommended_changes.append("Investigate gate redesign only after confirming why the learned gate suppresses useful corrections; gate-open rollout outperforms the trained gating path.")
    else:
        recommended_changes.append("Keep the next change focused on correction quality rather than removing gating entirely; gate-open does not improve rollout.")
    if output_cap["head_is_bounded"] and output_cap["shock_true_above_cap_frac"] > 0.2:
        recommended_changes.append("Remove or relax the bounded correction-head output range before larger redesigns; the present head cannot represent a large fraction of shock corrections.")
    if alpha_rollouts["0.25"]["final_l2_hybrid"] < alpha_rollouts["1.00"]["final_l2_hybrid"]:
        recommended_changes.append("A conservative correction mechanism is justified next: smaller applied scales are safer than the current full-strength correction.")
    if pred_mass["smooth_mass_frac"] > target_mass["smooth_mass_frac"] + 0.05:
        recommended_changes.append("Bias the next architecture toward stronger spatial locality, because the current model places too much correction mass in smooth regions.")

    report_data = {
        "shock_feature": shock_feature,
        "transition_threshold": transition_threshold,
        "shock_threshold": shock_threshold,
        "target_stats": target_stats,
        "target_mass": target_mass,
        "sign_summary": sign_summary,
        "feature_correlations": feature_correlations,
        "feature_binned": feature_binned,
        "probe_results": probe_results,
        "prediction_metrics": prediction_metrics,
        "gate_summary": gate_summary,
        "output_cap": output_cap,
        "feature_layout": feature_layout,
        "evaluation_correction_scale": evaluation_correction_scale,
        "gate_mode_metrics": gate_mode_metrics,
        "gate_mode_rollouts": gate_mode_rollouts,
        "gate_mode_order": gate_mode_order,
        "pred_mass": pred_mass,
        "applied_mass": applied_mass,
        "support_threshold": support_threshold,
        "support_summary": support_summary,
        "time_buckets": default_rollout["time_buckets"],
        "rollout_sign_summary": default_rollout["adjacent_sign"],
        "alpha_values": alpha_values,
        "alpha_rollouts": alpha_rollouts,
        "main_conclusions": main_conclusions,
        "recommended_changes": recommended_changes,
        "val_target": y_val,
    }

    report_path = _ensure_parent(cfg_get(analysis_cfg, "report_path", "analysis/preliminary_investigation_report.md"))
    summary_json_path = _ensure_parent(cfg_get(analysis_cfg, "summary_json_path", "analysis/preliminary_investigation_summary.json"))
    plot_path = _ensure_parent(cfg_get(analysis_cfg, "plot_path", "analysis/preliminary_investigation_plots.png"))

    report_path.write_text(_generate_report_text(report_data), encoding="utf-8")
    summary_payload = {key: value for key, value in report_data.items() if key not in {"val_target"}}
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    if bool(cfg_get(analysis_cfg, "plot_enabled", True)):
        _write_plots(plot_path, report_data)

    print(f"Report written to: {report_path}")
    print(f"Summary JSON written to: {summary_json_path}")
    if bool(cfg_get(analysis_cfg, 'plot_enabled', True)):
        print(f"Plots written to: {plot_path}")
    print()
    print("Target magnitude summary")
    print(_format_metric_table([
        _stats_row("global", target_stats["global"]),
        _stats_row("shock", target_stats["shock"]),
        _stats_row("smooth", target_stats["smooth"]),
    ], ["region", "mean", "std", "|.| mean", "|.| max", "p50|.|", "p90|.|", "p95|.|", "p99|.|"]))
    print()
    print("Prediction quality summary")
    print(_format_metric_table([
        _pair_row("global", prediction_metrics["global"]),
        _pair_row("shock", prediction_metrics["shock"]),
        _pair_row("smooth", prediction_metrics["smooth"]),
    ], ["region", "count", "MSE", "MAE", "cos", "corr", "sign", "|pred| mean", "|true| mean", "|pred|/|true|"]))
    print()
    print("Gate mode comparison")
    print(_format_metric_table(
        [_gate_mode_row(mode, gate_mode_metrics[mode], gate_mode_rollouts[mode]) for mode in gate_mode_order],
        ["gate_mode", "global MAE", "shock MAE", "smooth |pred|", "global cos", "rollout L2", "hybrid/base"],
    ))
    print()
    print("Alpha rollout comparison")
    print(_format_metric_table(
        [_alpha_row(alpha, alpha_rollouts[f"{alpha:.2f}"]) for alpha in alpha_values],
        ["alpha", "eff scale", "base L2", "hybrid L2", "hybrid/base", "mean |pred|", "smooth |applied|"],
    ))
    return report_data


def _parse_args():
    parser = argparse.ArgumentParser(description="Run preliminary diagnostics for the Hybrid WENO5-KAN one-step correction setup.")
    parser.add_argument("--config", type=str, default=None, help="Optional config path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    run_analysis(cfg)
