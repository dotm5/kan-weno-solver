import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from kan import GatedKAN, HybridScaler, HybridCorrectionLoss, TargetAffineScaler, resolve_model_runtime_config
from utils.config import cfg_get, load_config, set_global_seed
from utils.features import FEATURE_LAYOUT_VERSION, build_feature_layout_metadata, physics_feature_names
from utils.metadata import (
    build_checkpoint_metadata,
    default_correction_head_metadata,
    default_target_scaling_metadata,
    extract_dataset_metadata,
    validate_one_step_metadata,
)


class PDEDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _gate_terms(gate, shock_metric, smooth_threshold=0.1, margin=0.05):
    """Gate regularizers to prevent collapse after target scaling."""
    smooth_mask = (shock_metric < smooth_threshold).float().unsqueeze(1)
    shock_mask = 1.0 - smooth_mask

    smooth_count = smooth_mask.sum() + 1e-6
    shock_count = shock_mask.sum() + 1e-6

    gate_smooth_mean = (gate * smooth_mask).sum() / smooth_count
    gate_shock_mean = (gate * shock_mask).sum() / shock_count

    sparsity_core = gate_smooth_mean
    separation_core = F.relu((gate_smooth_mean + margin) - gate_shock_mean)
    shock_open_core = ((1.0 - gate) * shock_mask).sum() / shock_count

    return gate_smooth_mean, gate_shock_mean, sparsity_core, separation_core, shock_open_core


def _resolve_device(runtime_cfg):
    dev = str(runtime_cfg.get("device", "auto")).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _resolve_auto_flag(value, auto_default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() == "auto":
        return auto_default
    return bool(value)


def _prepare_dataset_metadata(
    data,
    *,
    data_path: Path,
    stencil_size: int,
    inferred_phys_dim: int,
    cfg,
    model_runtime_cfg,
):
    warn_on_missing = bool(cfg_get(cfg, "metadata.warn_on_missing", True))
    strict_metadata = bool(cfg_get(cfg, "metadata.strict", True))
    cfg_steps_ahead = int(cfg_get(cfg, "data_generation.steps_ahead", 1))
    if cfg_steps_ahead != 1:
        raise ValueError(
            "Training config must use strict one-step correction: "
            f"data_generation.steps_ahead={cfg_steps_ahead}."
        )

    dataset_metadata = extract_dataset_metadata(
        data,
        source_label=str(data_path),
        warn_on_missing=warn_on_missing,
    )
    dataset_metadata.setdefault("steps_ahead", int(data["steps_ahead"]) if "steps_ahead" in data.files else 1)
    dataset_metadata.setdefault("stencil_size", stencil_size)
    dataset_metadata.setdefault("phys_dim", inferred_phys_dim)
    dataset_metadata.setdefault("physics_feature_names", physics_feature_names(inferred_phys_dim))
    dataset_metadata.setdefault("feature_layout_version", FEATURE_LAYOUT_VERSION)
    expected_feature_layout = build_feature_layout_metadata(
        stencil_size=stencil_size,
        phys_dim=inferred_phys_dim,
        physics_feature_names=physics_feature_names(inferred_phys_dim),
        use_stencil_features=bool(model_runtime_cfg["use_stencil_features"]),
        stencil_radius=int(model_runtime_cfg["stencil_radius"]),
        gate_use_stencil_features=bool(model_runtime_cfg["gate_use_stencil_features"]),
    )
    dataset_metadata.setdefault("feature_layout", expected_feature_layout)
    expected_feature_layout_for_validation = (
        expected_feature_layout
        if dataset_metadata.get("feature_layout_version") == FEATURE_LAYOUT_VERSION
        and not bool(dataset_metadata.get("feature_layout_inferred", False))
        else None
    )

    try:
        validate_one_step_metadata(
            dataset_metadata,
            source_label=str(data_path),
            expected_steps_ahead=cfg_steps_ahead,
            expected_stencil_size=stencil_size,
            expected_phys_dim=inferred_phys_dim,
            expected_feature_names=physics_feature_names(inferred_phys_dim),
            expected_feature_layout=expected_feature_layout_for_validation,
        )
    except ValueError:
        if strict_metadata:
            raise
        warnings.warn(
            f"{data_path}: metadata validation failed, continuing because metadata.strict=false.",
            RuntimeWarning,
            stacklevel=2,
        )
    return dataset_metadata


def train_with_config(cfg):
    runtime_cfg = cfg.get("runtime", {})
    path_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    opt_cfg = cfg.get("optimizer", {})
    sched_cfg = cfg.get("scheduler", {})
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    solver_cfg = cfg.get("solver", {})
    scaler_cfg = cfg.get("scaler", {})
    target_scaler_cfg = cfg.get("target_scaler", {})
    strict_metadata = bool(cfg_get(cfg, "metadata.strict", True))

    set_global_seed(runtime_cfg.get("seed", None), bool(runtime_cfg.get("deterministic", False)))

    device = _resolve_device(runtime_cfg)
    print(f"Using Device: {device}")

    data_path = Path(path_cfg.get("train_data_path", "kan_train_data.npz"))
    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    # 严格 one-step 语义：训练数据必须直接存储 next-step correction target。
    data = np.load(str(data_path), allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    stencil_size = int(data["stencil_size"])
    inferred_phys_dim = int(X.shape[1] - stencil_size)
    model_runtime_cfg = resolve_model_runtime_config(
        stencil_size=stencil_size,
        artifact_model_cfg=model_cfg,
        fallback_model_cfg={},
        metadata=None,
        prefer_legacy_when_missing=False,
        warn_on_legacy=True,
        source_label="train-config",
    )
    dataset_metadata = _prepare_dataset_metadata(
        data,
        data_path=data_path,
        stencil_size=stencil_size,
        inferred_phys_dim=inferred_phys_dim,
        cfg=cfg,
        model_runtime_cfg=model_runtime_cfg,
    )
    phys_dim = int(dataset_metadata.get("phys_dim", inferred_phys_dim))
    print(
        "Dataset metadata: "
        f"steps_ahead={dataset_metadata.get('steps_ahead')}, "
        f"phys_dim={phys_dim}, "
        f"feature_layout={dataset_metadata.get('feature_layout_version')}, "
        f"corr_head={model_runtime_cfg['correction_head_output_mode']}"
    )

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X,
        y,
        test_size=float(data_cfg.get("split_test_size", 0.2)),
        random_state=int(data_cfg.get("split_random_state", 42)),
    )

    input_scaler = HybridScaler(
        stencil_size=stencil_size,
        eps=float(scaler_cfg.get("eps", 1e-8)),
        clip_percentile_abs_features=float(scaler_cfg.get("clip_percentile_abs_features", 99.5)),
    )
    X_train = input_scaler.fit(X_train_raw).transform(X_train_raw)
    X_val = input_scaler.transform(X_val_raw)

    target_scaler = TargetAffineScaler(
        eps=float(target_scaler_cfg.get("eps", 1e-12)),
        min_std=float(target_scaler_cfg.get("min_std", 1e-8)),
        clip_z=target_scaler_cfg.get("clip_z", None),
    ).fit(y_train_raw)

    y_train = target_scaler.transform(y_train_raw).astype(np.float32)
    y_val = target_scaler.transform(y_val_raw).astype(np.float32)

    print(
        "Target affine scaling: "
        f"mean={target_scaler.mean:.3e}, std={target_scaler.std:.3e}, clip_z={target_scaler.clip_z}"
    )

    pin_memory = _resolve_auto_flag(runtime_cfg.get("pin_memory", "auto"), device.type == "cuda")
    num_workers = int(runtime_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        PDEDataset(X_train, y_train),
        batch_size=int(train_cfg.get("batch_size", 4096)),
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        PDEDataset(X_val, y_val),
        batch_size=int(train_cfg.get("batch_size", 4096)),
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    model = GatedKAN(
        stencil_size=stencil_size,
        phys_dim=phys_dim,
        hidden_dim=int(model_runtime_cfg["hidden_dim"]),
        shape_grid_size=int(model_runtime_cfg["shape_grid_size"]),
        shape_spline_order=int(model_runtime_cfg["shape_spline_order"]),
        correction_head_output_mode=str(model_runtime_cfg["correction_head_output_mode"]),
        correction_head_output_scale=float(model_runtime_cfg["correction_head_output_scale"]),
        use_stencil_features=bool(model_runtime_cfg["use_stencil_features"]),
        stencil_radius=int(model_runtime_cfg["stencil_radius"]),
        gate_use_stencil_features=bool(model_runtime_cfg["gate_use_stencil_features"]),
        gate_hidden_dims=tuple(model_runtime_cfg["gate_hidden_dims"]),
        gate_temperature=float(model_runtime_cfg["gate_temperature"]),
        gate_bias_init=float(model_runtime_cfg["gate_bias_init"]),
        shock_indicator_threshold=float(model_runtime_cfg["shock_indicator_threshold"]),
        curvature_eps=float(model_runtime_cfg["curvature_eps"]),
    ).to(device)

    if str(opt_cfg.get("type", "adamw")).lower() != "adamw":
        raise ValueError(f"Unsupported optimizer type: {opt_cfg.get('type')}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
    )

    if str(sched_cfg.get("type", "reduce_on_plateau")).lower() != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler type: {sched_cfg.get('type')}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(sched_cfg.get("mode", "min")),
        patience=int(sched_cfg.get("patience", 15)),
        factor=float(sched_cfg.get("factor", 0.5)),
    )

    use_amp = _resolve_auto_flag(runtime_cfg.get("use_amp", "auto"), device.type == "cuda")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    shock_feature_idx = stencil_size + (1 if phys_dim > 1 else 0)

    target_mean_t = torch.tensor(target_scaler.mean, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_scaler.std, dtype=torch.float32, device=device)

    smooth_l1_beta = float(loss_cfg.get("smooth_l1_beta", 1.0))
    shock_weight = float(loss_cfg.get("shock_weight", 5.0))
    smooth_threshold = float(loss_cfg.get("gate_smooth_threshold", 0.1))
    smooth_penalty_weight = float(loss_cfg.get("smooth_penalty_weight", 0.0))
    locality_penalty_weight = float(loss_cfg.get("locality_penalty_weight", 0.0))
    separation_margin = float(loss_cfg.get("gate_separation_margin", 0.05))
    gate_sparsity_w = float(loss_cfg.get("gate_sparsity_weight", 2e-2))
    gate_separation_w = float(loss_cfg.get("gate_separation_weight", 5e-1))
    gate_shock_open_w = float(loss_cfg.get("gate_shock_open_weight", 1e-1))
    gate_reg_scale_factor = float(loss_cfg.get("gate_reg_scale_factor", 0.2))
    gate_reg_scale_min = float(loss_cfg.get("gate_reg_scale_min", 1.0))
    gate_reg_scale_max = float(loss_cfg.get("gate_reg_scale_max", 25.0))
    phys_weight_factor = float(loss_cfg.get("phys_weight_factor", 5.0))
    correction_loss_fn = HybridCorrectionLoss(
        smooth_l1_beta=smooth_l1_beta,
        shock_weight=shock_weight,
        smooth_threshold=smooth_threshold,
        smooth_penalty_weight=smooth_penalty_weight,
        locality_penalty_weight=locality_penalty_weight,
    )

    accumulation_steps = int(train_cfg.get("accumulation_steps", 4))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 20))
    epochs = int(train_cfg.get("epochs", 100))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_pred_sum = 0.0
        train_pred_total_sum = 0.0
        train_gate_sep_sum = 0.0

        for i, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            shock_metric = batch_X[:, shock_feature_idx]

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp and device.type == "cuda"):
                pred_z, gate = model(batch_X)
                pred_phys = pred_z * target_std_t + target_mean_t

                pred_terms = correction_loss_fn(
                    pred_z,
                    batch_y,
                    shock_metric=shock_metric,
                    pred_phys=pred_phys,
                )
                pred_loss = pred_terms["total"]

                gate_smooth, gate_shock, sparsity_core, separation_core, shock_open_core = _gate_terms(
                    gate,
                    shock_metric,
                    smooth_threshold=smooth_threshold,
                    margin=separation_margin,
                )
                gate_reg = (
                    gate_sparsity_w * sparsity_core
                    + gate_separation_w * separation_core
                    + gate_shock_open_w * shock_open_core
                )

                gate_reg_scale = (
                    torch.clamp(pred_loss.detach(), min=gate_reg_scale_min, max=gate_reg_scale_max)
                    * gate_reg_scale_factor
                )
                loss = (pred_loss + gate_reg_scale * gate_reg) / accumulation_steps

            grad_scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_sum += loss.item() * accumulation_steps
            train_pred_sum += pred_terms["weighted_smooth_l1"].item()
            train_pred_total_sum += pred_loss.item()
            train_gate_sep_sum += (gate_shock - gate_smooth).item()

            del (
                batch_X,
                batch_y,
                shock_metric,
                pred_z,
                pred_phys,
                gate,
                pred_terms,
                pred_loss,
                gate_smooth,
                gate_shock,
                sparsity_core,
                separation_core,
                shock_open_core,
                gate_reg,
                gate_reg_scale,
                loss,
            )

        model.eval()
        val_pred_sum = 0.0
        val_pred_total_sum = 0.0
        val_phys_wmse_sum = 0.0
        val_gate_sep_sum = 0.0
        batch_count = 0

        with torch.no_grad(), torch.amp.autocast(
            device_type=autocast_device,
            enabled=use_amp and device.type == "cuda",
        ):
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                shock_metric = batch_X[:, shock_feature_idx]

                pred_z, gate = model(batch_X)
                pred_phys = pred_z * target_std_t + target_mean_t
                pred_terms = correction_loss_fn(
                    pred_z,
                    batch_y,
                    shock_metric=shock_metric,
                    pred_phys=pred_phys,
                )
                pred_loss = pred_terms["weighted_smooth_l1"]
                val_pred_sum += pred_loss.item() * batch_X.size(0)
                val_pred_total_sum += pred_terms["total"].item() * batch_X.size(0)
                target_phys = batch_y * target_std_t + target_mean_t
                weights = 1.0 + phys_weight_factor * torch.abs(target_phys)
                wmse_phys = torch.mean(weights * (pred_phys - target_phys) ** 2)
                val_phys_wmse_sum += wmse_phys.item() * batch_X.size(0)

                gate_smooth, gate_shock, _, _, _ = _gate_terms(
                    gate,
                    shock_metric,
                    smooth_threshold=smooth_threshold,
                    margin=separation_margin,
                )
                val_gate_sep_sum += (gate_shock - gate_smooth).item() * batch_X.size(0)
                batch_count += batch_X.size(0)

                del (
                    batch_X,
                    batch_y,
                    shock_metric,
                    pred_z,
                    gate,
                    pred_terms,
                    pred_loss,
                    pred_phys,
                    target_phys,
                    weights,
                    wmse_phys,
                    gate_smooth,
                    gate_shock,
                )

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        avg_train_pred = train_pred_sum / max(len(train_loader), 1)
        avg_train_pred_total = train_pred_total_sum / max(len(train_loader), 1)
        avg_train_gate_sep = train_gate_sep_sum / max(len(train_loader), 1)

        avg_val_pred = val_pred_sum / max(batch_count, 1)
        avg_val_pred_total = val_pred_total_sum / max(batch_count, 1)
        avg_val_phys_wmse = val_phys_wmse_sum / max(batch_count, 1)
        avg_val_gate_sep = val_gate_sep_sum / max(batch_count, 1)

        scheduler.step(avg_val_pred)

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:3d} | Loss(total): {avg_train_loss:.2e} | Pred(z,w): {avg_train_pred:.2e} | "
                f"Pred(all): {avg_train_pred_total:.2e} | Val Pred(z,w): {avg_val_pred:.2e} | "
                f"Val Pred(all): {avg_val_pred_total:.2e} | Val W-MSE(phys): {avg_val_phys_wmse:.2e} | "
                f"Gate Sep train/val: {avg_train_gate_sep:.3f}/{avg_val_gate_sep:.3f}"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    save_target_scaler = bool(cfg_get(cfg, "checkpoint.save_target_scaler", True))
    feature_layout_metadata = build_feature_layout_metadata(
        stencil_size=stencil_size,
        phys_dim=phys_dim,
        physics_feature_names=physics_feature_names(phys_dim),
        use_stencil_features=bool(model_runtime_cfg["use_stencil_features"]),
        stencil_radius=int(model_runtime_cfg["stencil_radius"]),
        gate_use_stencil_features=bool(model_runtime_cfg["gate_use_stencil_features"]),
    )
    correction_head_metadata = default_correction_head_metadata(
        output_mode=str(model_runtime_cfg["correction_head_output_mode"]),
        output_scale=float(model_runtime_cfg["correction_head_output_scale"]),
    )
    checkpoint_metadata = build_checkpoint_metadata(
        dataset_metadata=dataset_metadata,
        solver_cfg=solver_cfg,
        target_scaler_cfg=target_scaler_cfg,
        target_scaler_enabled=save_target_scaler,
        correction_head_cfg=correction_head_metadata,
        feature_layout=feature_layout_metadata,
    )
    try:
        validate_one_step_metadata(
            checkpoint_metadata,
            source_label="checkpoint-save",
            expected_steps_ahead=1,
            expected_stencil_size=stencil_size,
            expected_phys_dim=phys_dim,
            expected_feature_names=physics_feature_names(phys_dim),
            expected_target_scaling=default_target_scaling_metadata(target_scaler_enabled=save_target_scaler),
            expected_feature_layout=feature_layout_metadata,
            expected_correction_head=correction_head_metadata,
        )
    except ValueError:
        if strict_metadata:
            raise
        warnings.warn(
            "Checkpoint metadata validation failed, continuing because metadata.strict=false.",
            RuntimeWarning,
            stacklevel=2,
        )

    save_dict = {
        "model_state_dict": model.state_dict(),
        "scaler_state": input_scaler.state_dict(),
        "stencil_size": stencil_size,
        "phys_dim": phys_dim,
        "steps_ahead": int(dataset_metadata.get("steps_ahead", 1)),
        "config": cfg,
        "metadata": checkpoint_metadata,
    }
    if save_target_scaler:
        save_dict["target_scaler_state"] = target_scaler.state_dict()

    model_save_path = Path(path_cfg.get("model_save_path", "kan_model.pth"))
    torch.save(save_dict, str(model_save_path))
    print(f"Model saved to '{model_save_path}'")


def train(data_path="kan_train_data.npz", epochs=100, batch_size=4096, lr=1e-3, accumulation_steps=4, device=None):
    """向后兼容入口：保留旧签名，内部转为配置驱动。"""
    cfg = load_config(None)
    cfg["paths"]["train_data_path"] = data_path
    cfg["training"]["epochs"] = epochs
    cfg["training"]["batch_size"] = batch_size
    cfg["optimizer"]["lr"] = lr
    cfg["training"]["accumulation_steps"] = accumulation_steps
    if device is not None:
        cfg["runtime"]["device"] = str(device)
    return train_with_config(cfg)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid WENO5-KAN model.")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选）")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    train_with_config(cfg)
