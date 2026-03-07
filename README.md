# Hybrid WENO5-KAN Solver for Burgers' Equation

Hybrid solver for 1D Burgers' equation:
- numerical backbone: WENO5 + Lax-Friedrichs + TVD-RK3
- learned correction: Gated KAN
- objective: keep smooth-region stability while activating correction in shock-relevant regions

## What Changed (Current Workflow)

- YAML-first configuration system:
  - default: `config/default_config.yaml`
  - override: `--config config/xxx.yaml` (recursive merge on top of default)
- Gate pipeline improvements:
  - richer gate features (time embedding + gradient descriptors)
  - configurable gate inference modes: `original`, `raw_gate_only`, `gate_open`, `soft_mask`
  - rollout diagnostics for `raw_gate`, `effective_gate`, `mask_active_ratio`, correction magnitude
- Target Affine Scaling:
  - train in normalized target space (`N(0,1)`-like)
  - inverse-transform during evaluation to physical correction scale
  - scaler state saved to checkpoint (`target_scaler_state`)
- Data generation upgrade:
  - single-session -> multi-session sampling
  - rolling progress bar + ETA in `data/generate.py`

## Project Structure

```text
.
├── config/
│   ├── default_config.yaml      # 默认完整配置（含中文注释）
│   └── example_config.yaml      # 只覆盖部分字段的示例
├── data/
│   └── generate.py              # 多会话数据生成 + 进度条/ETA
├── kan/
│   ├── model.py                 # Gated KAN / gate 输入构造
│   ├── scalers.py               # HybridScaler
│   └── losses.py
├── solvers/
│   └── weno.py                  # WENO5 + RK3
├── utils/
│   └── config.py                # YAML 加载、递归合并、seed
├── train.py                     # 配置驱动训练
├── evaluate.py                  # 配置驱动 rollout + 诊断绘图
├── rollout.py                   # evaluate 包装入口
├── dev_log.md                   # 开发日志
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## End-to-End Workflow

1. Inspect/prepare config
```bash
# 使用默认配置
python train.py

# 使用自定义覆盖配置
python train.py --config config/example_config.yaml
```

2. Generate dataset (multi-session)
```bash
# 默认配置生成
python -m data.generate

# 自定义配置生成
python -m data.generate --config config/example_config.yaml
```

3. Train
```bash
python train.py --config config/example_config.yaml
```

4. Evaluate / rollout
```bash
python evaluate.py --config config/example_config.yaml
# 或
python rollout.py --config config/example_config.yaml
```

## Gate Evaluation / Ablation Workflow

Configure `ablation.gate_mode` in YAML:
- `original`: `effective_gate = raw_gate * hard_mask`
- `raw_gate_only`: `effective_gate = raw_gate`
- `gate_open`: `effective_gate = 1`
- `soft_mask` (default): `effective_gate = raw_gate * soft_mask`

Useful debug options:
- `logging.debug_rollout: true`
- `logging.debug_t_start: 1.0`

When debug is enabled, evaluation logs:
- raw/effective gate stats
- mask active ratio
- shock indicator stats
- correction magnitude vs baseline update magnitude

Plot output (`paths.evaluation_plot_path`) includes:
- top panel: baseline vs hybrid L2 error
- bottom panel: raw/effective gate, mask ratio, correction magnitude

## Data Features

Current physics feature vector:

```text
[u_x, |u_x|, |u_xx|, grad_var, dt, sin(t), cos(t)]
```

## Checkpoint Contents

Saved by `train.py`:
- `model_state_dict`
- `scaler_state` (input scaler)
- `target_scaler_state` (target affine scaler, if enabled)
- `stencil_size`, `phys_dim`, `steps_ahead`
- `config` snapshot

## Testing

```bash
python -m pytest
python -m tests.convergence
```
