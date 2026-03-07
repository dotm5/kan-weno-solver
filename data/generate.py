import argparse
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from solvers.weno import rk3_step
from utils.config import load_config, set_global_seed


def generate_random_state(
    N,
    num_modes=6,
    amplitude_min=0.1,
    amplitude_max=1.0,
    amplitude_decay_power=1.0,
):
    """Generate random wave with high-frequency components."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u = np.zeros_like(x)
    for k in range(1, num_modes + 1):
        amp = np.random.uniform(amplitude_min, amplitude_max) / (k ** amplitude_decay_power)
        phase = np.random.uniform(0, 2 * np.pi)
        u += amp * np.sin(k * x + phase)
    u = u / (np.max(np.abs(u)) + 1e-8)
    return u


def get_multistep_error(u_coarse_init, N_coarse, N_fine, dt, steps, nu=0.0, weno_epsilon=1e-6):
    """Calculate accumulated error over 'steps'."""
    u_fine = np.interp(
        np.linspace(0, 2 * np.pi, N_fine, endpoint=False),
        np.linspace(0, 2 * np.pi, N_coarse, endpoint=False),
        u_coarse_init,
    )

    substeps_ratio = N_fine // N_coarse
    dt_fine = dt / substeps_ratio

    for _ in range(steps * substeps_ratio):
        u_fine = rk3_step(u_fine, 2 * np.pi / N_fine, dt_fine, nu=nu, weno_epsilon=weno_epsilon)

    u_truth_down = u_fine[::substeps_ratio]
    u_weno = u_coarse_init.copy()
    dx_coarse = 2 * np.pi / N_coarse

    for _ in range(steps):
        u_weno = rk3_step(u_weno, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)

    return u_truth_down - u_weno


def compute_physics_features(u, dx, dt, t_stamp):
    """Physics features: [u_x, |u_x|, |u_xx|, grad_var, dt, sin(t), cos(t)]."""
    u_x = np.gradient(u, dx)
    abs_ux = np.abs(u_x)
    u_xx = np.gradient(u_x, dx)
    abs_uxx = np.abs(u_xx)
    grad_var = np.sqrt(((np.roll(u_x, -1) - u_x) ** 2 + (np.roll(u_x, 1) - u_x) ** 2) * 0.5)
    dt_feat = np.full_like(u_x, dt)
    t_sin = np.full_like(u_x, np.sin(t_stamp))
    t_cos = np.full_like(u_x, np.cos(t_stamp))
    return np.stack([u_x, abs_ux, abs_uxx, grad_var, dt_feat, t_sin, t_cos], axis=1)


def build_sample(u_n, dx_coarse, dt, t_stamp, stencil_size):
    """Build one full-grid training sample block from the current PDE state."""
    pad_size = stencil_size // 2
    phys_feats = compute_physics_features(u_n, dx_coarse, dt, t_stamp)
    u_padded = np.pad(u_n, (pad_size, pad_size), mode="wrap")
    stencils = sliding_window_view(u_padded, window_shape=stencil_size)
    return np.hstack([stencils, phys_feats])


def rollout_session(
    N_coarse,
    N_fine,
    steps_ahead,
    stencil_size,
    cfl,
    samples_per_session,
    sample_stride,
    nu=0.0,
    weno_epsilon=1e-6,
    num_modes=6,
    amplitude_min=0.1,
    amplitude_max=1.0,
    amplitude_decay_power=1.0,
    warmup_factor=4,
    max_rollout_factor=3,
):
    """Run one trajectory session and collect multiple supervised samples."""
    dx_coarse = 2 * np.pi / N_coarse
    u_n = generate_random_state(
        N_coarse,
        num_modes=num_modes,
        amplitude_min=amplitude_min,
        amplitude_max=amplitude_max,
        amplitude_decay_power=amplitude_decay_power,
    )
    dt = cfl * dx_coarse / (np.max(np.abs(u_n)) + 1e-6)

    warmup_steps = np.random.randint(0, steps_ahead * warmup_factor + 1)
    t_stamp = 0.0
    for _ in range(warmup_steps):
        u_n = rk3_step(u_n, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)
        t_stamp += dt
        dt = cfl * dx_coarse / (np.max(np.abs(u_n)) + 1e-6)

    X_blocks = []
    y_blocks = []
    stride_count = 0

    max_rollout_steps = max(samples_per_session * sample_stride * max_rollout_factor, samples_per_session + 1)
    for _ in range(max_rollout_steps):
        if stride_count % sample_stride == 0:
            err = get_multistep_error(
                u_n,
                N_coarse,
                N_fine,
                dt,
                steps_ahead,
                nu=nu,
                weno_epsilon=weno_epsilon,
            )
            X_blocks.append(build_sample(u_n, dx_coarse, dt, t_stamp, stencil_size))
            y_blocks.append(err)
            if len(X_blocks) >= samples_per_session:
                break

        u_n = rk3_step(u_n, dx_coarse, dt, nu=nu, weno_epsilon=weno_epsilon)
        t_stamp += dt
        dt = cfl * dx_coarse / (np.max(np.abs(u_n)) + 1e-6)
        stride_count += 1

    if not X_blocks:
        err = get_multistep_error(
            u_n,
            N_coarse,
            N_fine,
            dt,
            steps_ahead,
            nu=nu,
            weno_epsilon=weno_epsilon,
        )
        X_blocks.append(build_sample(u_n, dx_coarse, dt, t_stamp, stencil_size))
        y_blocks.append(err)

    return np.vstack(X_blocks), np.concatenate(y_blocks).reshape(-1, 1)


def _format_seconds(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def generate_dataset(
    num_samples=5000,
    N_coarse=128,
    N_fine=2048,
    steps_ahead=10,
    stencil_size=9,
    cfl=0.5,
    num_sessions=64,
    sample_stride=1,
    seed=None,
    output_path="kan_train_data.npz",
    nu=0.0,
    weno_epsilon=1e-6,
    num_modes=6,
    amplitude_min=0.1,
    amplitude_max=1.0,
    amplitude_decay_power=1.0,
    warmup_factor=4,
    max_rollout_factor=3,
    progress_bar_width=30,
):
    print(f"Generating Dataset: {num_samples} samples, Lookahead={steps_ahead} steps...")
    print(f"Multi-session mode: sessions={num_sessions}, sample_stride={sample_stride}")

    if seed is not None:
        np.random.seed(seed)

    X_list = []
    y_list = []

    total_points = 0
    target_points = num_samples * N_coarse
    sessions_done = 0
    start_time = time.time()

    def print_progress(force=False):
        progress = min(total_points / max(target_points, 1), 1.0)
        elapsed = time.time() - start_time
        speed = total_points / max(elapsed, 1e-8)
        eta = (target_points - total_points) / max(speed, 1e-8)

        bar_width = int(progress_bar_width)
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)

        msg = (
            f"\r[{bar}] {progress * 100:6.2f}% "
            f"points={total_points}/{target_points} "
            f"sessions={sessions_done} "
            f"elapsed={_format_seconds(elapsed)} "
            f"ETA={_format_seconds(eta)}"
        )
        print(msg, end="\n" if force else "", flush=True)

    while total_points < target_points:
        sessions_done += 1
        remaining_samples = int(np.ceil((target_points - total_points) / N_coarse))
        sessions_left = max(num_sessions - sessions_done + 1, 1)
        samples_this_session = max(1, int(np.ceil(remaining_samples / sessions_left)))

        X_session, y_session = rollout_session(
            N_coarse=N_coarse,
            N_fine=N_fine,
            steps_ahead=steps_ahead,
            stencil_size=stencil_size,
            cfl=cfl,
            samples_per_session=samples_this_session,
            sample_stride=sample_stride,
            nu=nu,
            weno_epsilon=weno_epsilon,
            num_modes=num_modes,
            amplitude_min=amplitude_min,
            amplitude_max=amplitude_max,
            amplitude_decay_power=amplitude_decay_power,
            warmup_factor=warmup_factor,
            max_rollout_factor=max_rollout_factor,
        )

        X_list.append(X_session)
        y_list.append(y_session)
        total_points += X_session.shape[0]
        print_progress(force=False)

    X_data = np.vstack(X_list)
    y_data = np.vstack(y_list)

    X_data = X_data[:target_points]
    y_data = y_data[:target_points]
    total_points = X_data.shape[0]
    print_progress(force=True)

    output_path = Path(output_path)
    np.savez(str(output_path), X=X_data, y=y_data, steps_ahead=steps_ahead, stencil_size=stencil_size)
    print(f"Dataset saved to '{output_path}'. Shape: {X_data.shape}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate KAN training data in multi-session mode.")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选）")

    # 兼容旧参数：仅在显式传入时覆盖配置值
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--n-coarse", type=int, default=None)
    parser.add_argument("--n-fine", type=int, default=None)
    parser.add_argument("--steps-ahead", type=int, default=None)
    parser.add_argument("--stencil-size", type=int, default=None)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--num-sessions", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _override_if_not_none(target_dict, key, value):
    if value is not None:
        target_dict[key] = value


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)

    runtime_cfg = cfg.get("runtime", {})
    gen_cfg = dict(cfg.get("data_generation", {}))
    solver_cfg = cfg.get("solver", {})
    paths_cfg = cfg.get("paths", {})

    # CLI override（最小侵入）
    _override_if_not_none(gen_cfg, "num_samples", args.num_samples)
    _override_if_not_none(gen_cfg, "N_coarse", args.n_coarse)
    _override_if_not_none(gen_cfg, "N_fine", args.n_fine)
    _override_if_not_none(gen_cfg, "steps_ahead", args.steps_ahead)
    _override_if_not_none(gen_cfg, "stencil_size", args.stencil_size)
    _override_if_not_none(gen_cfg, "cfl", args.cfl)
    _override_if_not_none(gen_cfg, "num_sessions", args.num_sessions)
    _override_if_not_none(gen_cfg, "sample_stride", args.sample_stride)
    _override_if_not_none(gen_cfg, "seed", args.seed)

    set_global_seed(runtime_cfg.get("seed", None), bool(runtime_cfg.get("deterministic", False)))

    generate_dataset(
        num_samples=int(gen_cfg.get("num_samples", 5000)),
        N_coarse=int(gen_cfg.get("N_coarse", 128)),
        N_fine=int(gen_cfg.get("N_fine", 2048)),
        steps_ahead=int(gen_cfg.get("steps_ahead", 10)),
        stencil_size=int(gen_cfg.get("stencil_size", 9)),
        cfl=float(gen_cfg.get("cfl", 0.5)),
        num_sessions=int(gen_cfg.get("num_sessions", 64)),
        sample_stride=int(gen_cfg.get("sample_stride", 1)),
        seed=gen_cfg.get("seed", None),
        output_path=paths_cfg.get("data_output_path", "kan_train_data.npz"),
        nu=float(solver_cfg.get("nu", 0.0)),
        weno_epsilon=float(solver_cfg.get("weno_epsilon", 1e-6)),
        num_modes=int(gen_cfg.get("num_modes", 6)),
        amplitude_min=float(gen_cfg.get("amplitude_min", 0.1)),
        amplitude_max=float(gen_cfg.get("amplitude_max", 1.0)),
        amplitude_decay_power=float(gen_cfg.get("amplitude_decay_power", 1.0)),
        warmup_factor=int(gen_cfg.get("warmup_factor", 4)),
        max_rollout_factor=int(gen_cfg.get("max_rollout_factor", 3)),
        progress_bar_width=int(gen_cfg.get("progress_bar_width", 30)),
    )
