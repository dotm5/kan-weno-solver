import tempfile
from pathlib import Path

import matplotlib

from data.generate import generate_dataset
from evaluate import run_evaluation
from train import train_with_config
from utils.config import load_config

# Use non-interactive backend for matplotlib in CI
matplotlib.use("Agg")


def run_ci_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_path = tmp_path / "kan_train_data.npz"
        model_path = tmp_path / "kan_model.pth"
        plot_path = tmp_path / "evaluation_result.png"

        print("=== CI Pipeline: Dataset Generation ===")
        generate_dataset(
            num_samples=8,
            N_coarse=64,
            N_fine=256,
            steps_ahead=1,
            output_path=str(data_path),
            num_sessions=2,
            warmup_factor=1,
            max_rollout_factor=1,
        )

        cfg = load_config(None)
        cfg["paths"]["train_data_path"] = str(data_path)
        cfg["paths"]["model_save_path"] = str(model_path)
        cfg["paths"]["evaluation_plot_path"] = str(plot_path)
        cfg["runtime"]["device"] = "cpu"
        cfg["runtime"]["use_amp"] = False
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 32
        cfg["training"]["accumulation_steps"] = 1
        cfg["training"]["log_every"] = 1
        cfg["plotting"]["enabled"] = False
        cfg["evaluation"]["N_coarse"] = 64
        cfg["evaluation"]["N_ref"] = 256
        cfg["evaluation"]["T_final"] = 0.05
        cfg["evaluation"]["cfl"] = 0.2

        print("=== CI Pipeline: Training ===")
        train_with_config(cfg)

        print("=== CI Pipeline: Evaluation ===")
        run_evaluation(cfg)


if __name__ == "__main__":
    run_ci_pipeline()
