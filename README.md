# Hybrid WENO5-KAN Solver for Burgers' Equation

This project implements a **hybrid numerical solver** that integrates the high-order **WENO5** (Weighted Essential Non-Oscillatory) scheme with **Gated Kolmogorov-Arnold Networks (Gated KAN)**.

The hybrid approach uses KAN to learn and correct numerical truncation errors on coarse grids, achieving high-resolution fidelity with significantly lower computational overhead.

## 🚀 Key Features

*   **Core Physics Engine**: Vectorized WENO5 implementation with Lax-Friedrichs flux splitting and TVD-RK3 time integration.
*   **Gated KAN Architecture (v4.1)**: A physics-gated network that dynamically applies corrections based on local flow gradients and shock metrics.
*   **Physics-Consistent Learning**: A custom loss function incorporating shock-weighting and sparsity constraints to ensure stability near discontinuities.
*   **Modular Design**: Clean separation between the neural network core, numerical solvers, data pipelines, and testing suites.

---

## 📂 Project Structure

```text
.
├── kan/                   # Neural Network Package
│   ├── model.py           # Gated KAN & KANLinear definitions
│   ├── scalers.py         # Physics-aware HybridScaler
│   └── losses.py          # Physics-Consistent Loss functions
├── solvers/               # Numerical Physics Solvers
│   └── weno.py            # WENO5 + RK3 implementation
├── data/                  # Data Generation Pipeline
│   └── generate.py        # High-fidelity data generation logic
├── tests/                 # Professional Test Suite
│   ├── conftest.py        # Shared pytest fixtures
│   ├── test_baseline.py   # Solver accuracy & conservation tests
│   ├── test_dataset.py    # Data integrity tests
│   ├── test_training.py   # Model convergence & I/O tests
│   └── convergence.py     # WENO5 order-of-accuracy verification
├── train.py               # Unified training entry point
├── evaluate.py            # Performance evaluation & ablation study
├── dev_log.md             # Development history (Chinese)
└── requirements.txt       # Project dependencies
```

---

## 🛠️ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Workflow

**Step 1: Generate Training Data**
Generate accumulated error data (Truth vs. Baseline) using a 9-point stencil.
```bash
python -m data.generate
```

**Step 2: Train the Model**
Train the Gated KAN model using the physics-consistent loss.
```bash
python train.py
```

**Step 3: Evaluate Performance**
Run a full rollout simulation to compare Baseline WENO5 vs. Hybrid WENO5-KAN.
```bash
python evaluate.py
```

---

## 🧪 Testing

The project uses `pytest` for quality assurance. The suite covers numerical conservation, model convergence, and data integrity.

Run all tests:
```bash
pytest
```

Run convergence verification:
```bash
python -m tests.convergence
```

---

## 📈 Results

The **Gated KAN** model effectively identifies shock regions and applies localized corrections. This results in:
1.  **Reduced Numerical Dissipation**: Sharper shock fronts compared to standard coarse-grid WENO5.
2.  **Phase Error Correction**: Improved wave propagation speed accuracy over long-term integration.
3.  **Physical Consistency**: The gating mechanism ensures corrections are suppressed in smooth regions where the baseline solver is already accurate.
