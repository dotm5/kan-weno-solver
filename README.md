# Hybrid WENO5-KAN Solver for Burgers' Equation

> **Status: Deprecated / Archived**
>
> 本项目已确认当前技术路线不可行，因此不再继续维护或推进。
> 仓库仅作为阶段性实验记录保留，不建议继续基于该项目进行开发、复现或扩展。

## 项目状态说明

本项目原计划构建一个用于一维 Burgers 方程的混合求解器：

- 数值主干：WENO5 + Lax-Friedrichs + TVD-RK3
- 学习修正模块：Gated KAN
- 目标：在保持光滑区域稳定性的同时，在激波相关区域激活学习修正项

经过实验验证后，当前路线未能达到预期目标。主要问题在于：

- 学习修正项难以在长时间 rollout 中稳定发挥作用
- gate 机制与物理区域选择之间的匹配效果不理想
- 修正项在激波区域的收益不足以抵消其带来的不稳定风险
- 当前框架难以证明相较于纯数值方法具有可靠优势

因此，本项目已被废弃。

## 原始设计概览

项目曾包含以下功能尝试：

- YAML-first 配置系统
  - 默认配置：`config/default_config.yaml`
  - 覆盖配置：`--config config/xxx.yaml`
  - 支持基于默认配置的递归合并

- Gate pipeline 改进
  - 时间嵌入特征
  - 梯度描述符
  - 多种 gate 推理模式：
    - `original`
    - `raw_gate_only`
    - `gate_open`
    - `soft_mask`

- Rollout 诊断
  - `raw_gate`
  - `effective_gate`
  - `mask_active_ratio`
  - correction magnitude

- Target Affine Scaling
  - 在归一化目标空间中训练
  - 评估时反变换回物理修正尺度
  - scaler 状态保存到 checkpoint

- 数据生成流程升级
  - 从 single-session 扩展到 multi-session sampling
  - `data/generate.py` 中加入进度条和 ETA

## Project Structure

```text
.
├── config/
│   ├── default_config.yaml      # 默认完整配置
│   └── example_config.yaml      # 部分字段覆盖示例
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
````

## Historical Usage

以下命令仅作为历史记录保留，不保证仍然可用，也不建议继续使用。

### Installation

```bash
pip install -r requirements.txt
```

### Generate Dataset

```bash
python -m data.generate
python -m data.generate --config config/example_config.yaml
```

### Train

```bash
python train.py
python train.py --config config/example_config.yaml
```

### Evaluate / Rollout

```bash
python evaluate.py --config config/example_config.yaml
python rollout.py --config config/example_config.yaml
```

### Testing

```bash
python -m pytest
python -m tests.convergence
```

## 原始物理特征

项目中使用过的 physics feature vector：

```text
[u_x, |u_x|, |u_xx|, grad_var, dt, sin(t), cos(t)]
```

## Checkpoint 内容

`train.py` 曾保存以下内容：

* `model_state_dict`
* `scaler_state`
* `target_scaler_state`
* `stencil_size`
* `phys_dim`
* `steps_ahead`
* `config` snapshot

