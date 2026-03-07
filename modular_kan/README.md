# Modular KAN PDE Solver Framework

A highly decoupled, modular PyTorch framework for solving Partial Differential Equations (PDEs) using Kolmogorov-Arnold Networks (KAN). This architecture emphasizes clean interfaces, dependency injection, and physics-informed learning.

## 🚀 Architecture Overview

The framework is strictly decoupled into functional modules:

- **`config.py`**: Centralized configuration using Python dataclasses. Defines data, model, training, and physics parameters.
- **`models.py`**: Implementation of `KANLinear`, `KAN`, and the `GatedKAN` architecture (which utilizes a gate network for physics-informed selection).
- **`physics.py`**: Contains `KANPhysicsLoss`, implementing Weighted MSE (shock-focusing) and physics-driven sparsity.
- **`dataset.py`**: Handles data loading, train/val splitting, and the `HybridScaler` (StandardScaling for stencils + Percentile scaling for physics).
- **`trainer.py`**: The execution engine. It is agnostic to the specific PDE or model architecture, interacting only via the `Solver` interface.
- **`solvers.py`**: The "glue" layer. `KANSolver` binds a specific model, loss function, and optimizer logic together.
- **`callbacks.py`**: Observability and lifecycle management (Logging, Checkpointing).

## 🛠 Key Features

- **Strict Decoupling**: The Trainer doesn't know about the PDE; the Model doesn't know about the Optimizer.
- **Physics-Informed Gating**: Uses a `GatedKAN` approach to apply corrections only where needed (e.g., near shocks).
- **Hybrid Scaling**: Specialized scaling logic to handle high-gradient regions in PDE solutions.
- **Extensible Callbacks**: Easily add new monitoring or checkpointing logic.

## 🏃 Getting Started

### Prerequisites
Ensure you have the requirements installed:
```bash
pip install torch numpy scikit-learn matplotlib
```

### Running Training
Execute the training loop using the module entry point:
```bash
python -m modular_kan.main
```

## ⚙️ Configuration

Configurations are managed in `modular_kan/config.py`. You can customize:

| Config Class | Key Parameters |
| :--- | :--- |
| **DataConfig** | `stencil_size`, `batch_size`, `test_size` |
| **ModelConfig** | `hidden_dim`, `grid_size`, `spline_order` |
| **PhysicsConfig** | `shock_weight`, `gate_sparsity`, `smooth_threshold` |
| **TrainingConfig** | `epochs`, `learning_rate`, `device`, `save_dir` |

## 🧪 Solver Interface

To solve a different PDE, simply inherit from the `Solver` base class in `trainer.py` and implement:
- `training_step()`
- `validation_step()`
- `configure_optimizers()`

This allows the same `Trainer` to be reused across entirely different physics problems.

## 📊 Monitoring

Training progress is logged to the console via the `LoggerCallback`. Best models and scaler states are automatically saved to the directory specified in `TrainingConfig.save_dir`.
