from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    data_path: str = 'kan_train_data.npz'
    stencil_size: int = 9
    phys_dim: int = 3
    test_size: float = 0.2
    batch_size: int = 16
    num_workers: int = 4

@dataclass
class ModelConfig:
    hidden_dim: int = 32
    grid_size: int = 10
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0

@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    factor: float = 0.5
    device: str = "cuda" if "cuda" else "cpu" # Placeholder, will be resolved at runtime
    save_dir: str = "checkpoints"
    experiment_name: str = "kan_pde_experiment"

@dataclass
class PhysicsConfig:
    shock_weight: float = 5.0
    gate_sparsity: float = 5e-4
    smooth_threshold: float = 0.1
