import numpy as np
import argparse
from solvers.weno import rk3_step
from numpy.lib.stride_tricks import sliding_window_view


def generate_random_state(N, num_modes=6):
    """Generate random wave with high-frequency components."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u = np.zeros_like(x)
    for k in range(1, num_modes + 1):
        amp = np.random.uniform(0.1, 1.0) / (k ** 1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        u += amp * np.sin(k * x + phase)
    u = u / (np.max(np.abs(u)) + 1e-8)
    return u


def get_multistep_error(u_coarse_init, N_coarse, N_fine, dt, steps):
    """Calculate accumulated error over 'steps'."""
    u_fine = np.interp(
        np.linspace(0, 2 * np.pi, N_fine, endpoint=False),
        np.linspace(0, 2 * np.pi, N_coarse, endpoint=False),
        u_coarse_init,
    )

    substeps_ratio = N_fine // N_coarse
    dt_fine = dt / substeps_ratio

    for _ in range(steps * substeps_ratio):
        u_fine = rk3_step(u_fine, 2 * np.pi / N_fine, dt_fine, nu=0.0)

    u_truth_down = u_fine[::substeps_ratio]
    u_weno = u_coarse_init.copy()
    dx_coarse = 2 * np.pi / N_coarse

    for _ in range(steps):
        u_weno = rk3_step(u_weno, dx_coarse, dt, nu=0.0)

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
):
    """Run one trajectory session and collect multiple supervised samples."""
    dx_coarse = 2 * np.pi / N_coarse
    u_n = generate_random_state(N_coarse)
    dt = cfl * dx_coarse / (np.max(np.abs(u_n)) + 1e-6)

    # Session warmup introduces temporal diversity within each trajectory.
    warmup_steps = np.random.randint(0, steps_ahead * 4 + 1)
    t_stamp = 0.0
    for _ in range(warmup_steps):
        u_n = rk3_step(u_n, dx_coarse, dt, nu=0.0)
        t_stamp += dt
        dt = cfl * dx_coarse / (np.max(np.abs(u_n)) + 1e-6)

    X_blocks = []
    y_blocks = []
    stride_count = 0

    max_rollout_steps = max(samples_per_session * sample_stride * 3, samples_per_session + 1)
    for _ in range(max_rollout_steps):
        if stride_count % sample_stride == 0:
            err = get_multistep_error(u_n, N_coarse, N_fine, dt, steps_ahead)
            X_blocks.append(build_sample(u_n, dx_coarse, dt, t_stamp, stencil_size))
            y_blocks.append(err)
            if len(X_blocks) >= samples_per_session:
                break

        u_n = rk3_step(u_n, dx_coarse, dt, nu=0.0)
        t_stamp += dt
        dt = cfl * dx_coarse / (np.max(np.abs(u_n)) + 1e-6)
        stride_count += 1

    if not X_blocks:
        err = get_multistep_error(u_n, N_coarse, N_fine, dt, steps_ahead)
        X_blocks.append(build_sample(u_n, dx_coarse, dt, t_stamp, stencil_size))
        y_blocks.append(err)

    return np.vstack(X_blocks), np.concatenate(y_blocks).reshape(-1, 1)


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
        )

        X_list.append(X_session)
        y_list.append(y_session)
        total_points += X_session.shape[0]

        if sessions_done % 8 == 0 or total_points >= target_points:
            progress = min(total_points / target_points, 1.0)
            print(f"  Session {sessions_done}: progress {progress:.1%}")

    X_data = np.vstack(X_list)
    y_data = np.vstack(y_list)

    # Trim any extra rows if final session overshoots target.
    X_data = X_data[:target_points]
    y_data = y_data[:target_points]

    filename = "kan_train_data.npz"
    np.savez(filename, X=X_data, y=y_data, steps_ahead=steps_ahead, stencil_size=stencil_size)
    print(f"Dataset saved to '{filename}'. Shape: {X_data.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KAN training data in multi-session mode.")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--n-coarse", type=int, default=128)
    parser.add_argument("--n-fine", type=int, default=2048)
    parser.add_argument("--steps-ahead", type=int, default=10)
    parser.add_argument("--stencil-size", type=int, default=9)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--num-sessions", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        N_coarse=args.n_coarse,
        N_fine=args.n_fine,
        steps_ahead=args.steps_ahead,
        stencil_size=args.stencil_size,
        cfl=args.cfl,
        num_sessions=args.num_sessions,
        sample_stride=args.sample_stride,
        seed=args.seed,
    )
