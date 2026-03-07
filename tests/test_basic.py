import warnings

import numpy as np
import pytest
import torch

from data.generate import build_paired_initial_state, generate_dataset, get_one_step_pair
from evaluate import KANPredictor, run_evaluation
from kan import GatedKAN, HybridScaler, TargetAffineScaler
from solvers.weno import weno5_flux_splitting
from train import train_with_config
from utils.config import load_config
from utils.features import downsample_periodic, physics_feature_names, require_integer_refinement
from utils.metadata import (
    ONE_STEP_TARGET_SEMANTICS,
    build_checkpoint_metadata,
    build_dataset_metadata,
    deserialize_metadata,
)


def test_gated_kan_shape():
    """Verify GatedKAN output and component shapes."""
    stencil_size = 9
    phys_dim = 7
    batch_size = 4
    model = GatedKAN(stencil_size=stencil_size, phys_dim=phys_dim)

    x = torch.randn(batch_size, stencil_size + phys_dim)
    correction, gate = model(x)
    raw_corr, raw_gate, gated_corr = model.forward_components(x)

    assert correction.shape == (batch_size, 1)
    assert gate.shape == (batch_size, 1)
    assert raw_corr.shape == (batch_size, 1)
    assert raw_gate.shape == (batch_size, 1)
    assert gated_corr.shape == (batch_size, 1)
    assert torch.allclose(correction, gated_corr)
    assert (gate >= 0).all() and (gate <= 1).all()


def test_weno5_flux_splitting():
    """Basic sanity check for WENO5 flux splitting."""
    N = 64
    u = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    hat_f = weno5_flux_splitting(u)

    assert hat_f.shape == (N,)
    assert not np.isnan(hat_f).any()


def test_one_step_pair_and_dataset_metadata(tmp_path):
    """Generated targets must match strict one-step correction semantics."""
    N_coarse = 8
    N_fine = 32
    ratio = require_integer_refinement(N_fine, N_coarse)
    dx_coarse = 2 * np.pi / N_coarse
    dx_ref = 2 * np.pi / N_fine

    np.random.seed(0)
    u_ref, u_coarse = build_paired_initial_state(
        N_coarse,
        N_fine,
        num_modes=2,
        amplitude_min=0.2,
        amplitude_max=0.4,
    )
    dt = 0.1 * dx_coarse / (np.max(np.abs(u_coarse)) + 1e-6)

    u_ref_next, u_weno_next, target = get_one_step_pair(
        u_ref,
        u_coarse,
        dx_ref=dx_ref,
        dx_coarse=dx_coarse,
        dt=dt,
        substeps_ratio=ratio,
        nu=0.0,
        weno_epsilon=1e-6,
    )

    assert np.allclose(target, downsample_periodic(u_ref_next, ratio=ratio) - u_weno_next)

    data_path = tmp_path / "one_step_data.npz"
    generate_dataset(
        num_samples=1,
        N_coarse=N_coarse,
        N_fine=N_fine,
        steps_ahead=1,
        stencil_size=5,
        cfl=0.2,
        num_sessions=1,
        sample_stride=1,
        seed=123,
        output_path=str(data_path),
        num_modes=2,
        amplitude_min=0.2,
        amplitude_max=0.4,
        warmup_factor=0,
        max_rollout_factor=1,
    )

    data = np.load(str(data_path), allow_pickle=False)
    metadata = deserialize_metadata(data["metadata_json"])
    assert int(data["steps_ahead"]) == 1
    assert int(data["phys_dim"]) == 7
    assert metadata["target_semantics"] == ONE_STEP_TARGET_SEMANTICS
    assert metadata["target_definition"] == "u_ref_next - u_weno_next"


def test_train_smoke_and_checkpoint_metadata(tmp_path):
    """Training should warn on missing metadata but still save one-step checkpoint metadata."""
    data_path = tmp_path / "tiny_data.npz"
    model_path = tmp_path / "tiny_model.pth"

    stencil_size = 9
    phys_dim = 7
    num_samples = 32
    X = np.random.randn(num_samples, stencil_size + phys_dim).astype(np.float32)
    y = np.random.randn(num_samples, 1).astype(np.float32)
    np.savez(data_path, X=X, y=y, stencil_size=stencil_size, steps_ahead=1)

    cfg = load_config(None)
    cfg["paths"]["train_data_path"] = str(data_path)
    cfg["paths"]["model_save_path"] = str(model_path)
    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["use_amp"] = False
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 8
    cfg["training"]["accumulation_steps"] = 1
    cfg["training"]["log_every"] = 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_with_config(cfg)

    assert any("metadata_json missing" in str(item.message) for item in caught)

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "scaler_state" in checkpoint
    assert "target_scaler_state" in checkpoint
    assert "config" in checkpoint
    assert "metadata" in checkpoint
    assert checkpoint["metadata"]["steps_ahead"] == 1


def test_train_rejects_multistep_dataset(tmp_path):
    """Legacy multistep datasets must fail fast under strict one-step training."""
    data_path = tmp_path / "bad_data.npz"
    X = np.random.randn(16, 16).astype(np.float32)
    y = np.random.randn(16, 1).astype(np.float32)
    np.savez(data_path, X=X, y=y, stencil_size=9, steps_ahead=2)

    cfg = load_config(None)
    cfg["paths"]["train_data_path"] = str(data_path)
    cfg["paths"]["model_save_path"] = str(tmp_path / "bad_model.pth")
    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["use_amp"] = False
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 8
    cfg["training"]["accumulation_steps"] = 1

    with pytest.raises(ValueError, match="steps_ahead"):
        train_with_config(cfg)


def _write_synthetic_checkpoint(path):
    stencil_size = 9
    phys_dim = 7
    X = np.random.randn(64, stencil_size + phys_dim).astype(np.float32)
    y = np.random.randn(64, 1).astype(np.float32)

    scaler = HybridScaler(stencil_size=stencil_size)
    scaler.fit(X)
    target_scaler = TargetAffineScaler().fit(y)
    model = GatedKAN(stencil_size=stencil_size, phys_dim=phys_dim)

    dataset_metadata = build_dataset_metadata(
        steps_ahead=1,
        stencil_size=stencil_size,
        phys_dim=phys_dim,
        physics_feature_names=physics_feature_names(phys_dim),
        solver_cfg={"nu": 0.0, "weno_epsilon": 1e-6},
        generation_cfg={"N_coarse": 16, "N_fine": 64, "cfl": 0.2, "sample_stride": 1, "num_sessions": 1},
    )
    checkpoint_metadata = build_checkpoint_metadata(
        dataset_metadata=dataset_metadata,
        solver_cfg={"nu": 0.0, "weno_epsilon": 1e-6},
        target_scaler_cfg={"clip_z": None, "min_std": 1e-8},
        target_scaler_enabled=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_state": scaler.state_dict(),
            "target_scaler_state": target_scaler.state_dict(),
            "stencil_size": stencil_size,
            "phys_dim": phys_dim,
            "steps_ahead": 1,
            "config": {"model": {}},
            "metadata": checkpoint_metadata,
        },
        str(path),
    )


def test_predictor_gate_modes_consistent(tmp_path):
    """train_equivalent and raw_gate_only should agree; masks should only reduce the gate."""
    ckpt_path = tmp_path / "predictor_ckpt.pth"
    _write_synthetic_checkpoint(ckpt_path)

    u = np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))
    dx = 2 * np.pi / 16
    dt = 0.1 * dx

    pred_train = KANPredictor(str(ckpt_path), device="cpu", gate_mode="train_equivalent")
    pred_raw = KANPredictor(str(ckpt_path), device="cpu", gate_mode="raw_gate_only")
    pred_soft = KANPredictor(str(ckpt_path), device="cpu", gate_mode="soft_mask")
    pred_hard = KANPredictor(str(ckpt_path), device="cpu", gate_mode="hard_mask")
    pred_open = KANPredictor(str(ckpt_path), device="cpu", gate_mode="gate_open")

    corr_train, dbg_train, _ = pred_train.predict(u, dx, dt, return_debug=True, return_components=True)
    corr_raw, dbg_raw, _ = pred_raw.predict(u, dx, dt, return_debug=True, return_components=True)
    _, dbg_soft, _ = pred_soft.predict(u, dx, dt, return_debug=True, return_components=True)
    _, dbg_hard, _ = pred_hard.predict(u, dx, dt, return_debug=True, return_components=True)
    _, dbg_open, _ = pred_open.predict(u, dx, dt, return_debug=True, return_components=True)

    assert np.allclose(corr_train, corr_raw, atol=1e-6)
    assert pytest.approx(dbg_open["effective_gate_mean"], rel=0, abs=1e-6) == 1.0
    assert dbg_soft["effective_gate_mean"] <= dbg_train["effective_gate_mean"] + 1e-8
    assert dbg_hard["effective_gate_mean"] <= dbg_train["effective_gate_mean"] + 1e-8


def test_run_evaluation_auto_sign_uses_plus(tmp_path):
    """Auto sign should resolve to plus for the strict one-step target definition."""
    ckpt_path = tmp_path / "eval_ckpt.pth"
    _write_synthetic_checkpoint(ckpt_path)

    cfg = load_config(None)
    cfg["paths"]["model_save_path"] = str(ckpt_path)
    cfg["paths"]["evaluation_plot_path"] = str(tmp_path / "eval.png")
    cfg["runtime"]["device"] = "cpu"
    cfg["evaluation"]["N_coarse"] = 16
    cfg["evaluation"]["N_ref"] = 64
    cfg["evaluation"]["T_final"] = 0.05
    cfg["evaluation"]["cfl"] = 0.2
    cfg["evaluation"]["remove_correction_mean"] = False
    cfg["plotting"]["enabled"] = False
    cfg["logging"]["debug_sign_metrics"] = True
    cfg["logging"]["debug_gate_metrics"] = True
    cfg["ablation"]["gate_mode"] = "train_equivalent"
    cfg["ablation"]["correction_sign_mode"] = "auto"

    result = run_evaluation(cfg)
    assert result["resolved_sign_mode"] == "plus"
    assert result["final_l2_base"] >= 0.0
    assert result["final_l2_hybrid"] >= 0.0
