# Hybrid WENO5-KAN Solver for Burgers' Equation

This project implements a **hybrid numerical solver** that integrates the high-order **WENO5** (Weighted Essential Non-Oscillatory) scheme with **Kolmogorov-Arnold Networks (KAN)**.

The goal is to use KAN to learn and correct the truncation errors of the WENO5 solver on coarse grids, effectively achieving high-resolution accuracy with low-resolution computational cost.

## Key Features

* **Core Solver**: Vectorized WENO5 implementation with Lax-Friedrichs flux splitting and TVD RK3 time integration.
* **Residual Learning**: Uses KAN to predict numerical errors based on local stencils.
* **Advanced Strategy**: Implements **Multi-step Tendency Learning** (predicting cumulative error over 10 steps) to improve stability.
* **Physics-Aware**: Includes conservation enforcement and shock-capturing weighted loss functions.

##  Project Structure

* `burgers_weno.py`: The core numerical engine (WENO5 + RK3).
* `generate_residual_dataset.py`: Generates training data (coarse grid vs. fine grid ground truth).
* `train_kan_residual.py`: Trains the KAN model to predict error residuals.
* `rollout_comparison.py`: Runs the ablation study and visualizes the performance (Baseline vs. Hybrid).

##  Quick Start

### 1. Prerequisites
Ensure you have `torch`, `numpy`, `matplotlib`, and `scikit-learn` installed.

```bash
pip install -r requirements.txt
```

### 2. Workflow

**Step 1: Generate Dataset**
Run this script to generate 10-step accumulated error data. It uses a 9-point stencil.
```bash
python generate_residual_dataset.py
````

_Output: `kan_train_data_multistep.npz`_

**Step 2: Train KAN Model**

Train the residual correction network using a Weighted MSE loss (focusing on shocks).

Bash

```
python train_kan_residual.py
```

_Output: `kan_model_multistep.pth`_

**Step 3: Verification (Rollout)**

Run the comparative simulation to see the hybrid solver in action.

Bash

```
python rollout_comparison.py
```

_Output: `rollout_v2_result.png`_

##  Results

The hybrid solver (WENO5 + KAN) demonstrates significant improvement over the standard coarse-grid WENO5 solver, particularly in maintaining shock sharpness and reducing phase error over long integration times.

