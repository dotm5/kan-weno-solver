import numpy as np


def weno5_flux_splitting(u, epsilon=1e-6, linear_weights=(0.1, 0.6, 0.3)):
    """
    向量化的 WENO5 通量重构 (Lax-Friedrichs Flux Splitting)。

    Args:
        u (np.ndarray): 守恒量 u (形状: [N])
        epsilon (float): WENO 权重稳定项
        linear_weights (tuple): 三个线性权重, 默认 (0.1, 0.6, 0.3)

    Returns:
        hat_f (np.ndarray): 界面处 i+1/2 的数值通量 (形状: [N])
    """
    alpha = np.max(np.abs(u))

    f_val = 0.5 * u ** 2
    fp = 0.5 * (f_val + alpha * u)
    fm = 0.5 * (f_val - alpha * u)

    vp0 = np.roll(fp, 2)
    vp1 = np.roll(fp, 1)
    vp2 = fp
    vp3 = np.roll(fp, -1)
    vp4 = np.roll(fp, -2)

    beta0_p = (13 / 12) * (vp0 - 2 * vp1 + vp2) ** 2 + (1 / 4) * (vp0 - 4 * vp1 + 3 * vp2) ** 2
    beta1_p = (13 / 12) * (vp1 - 2 * vp2 + vp3) ** 2 + (1 / 4) * (vp1 - vp3) ** 2
    beta2_p = (13 / 12) * (vp2 - 2 * vp3 + vp4) ** 2 + (1 / 4) * (3 * vp2 - 4 * vp3 + vp4) ** 2

    d0_p, d1_p, d2_p = linear_weights
    alpha0_p = d0_p / (epsilon + beta0_p) ** 2
    alpha1_p = d1_p / (epsilon + beta1_p) ** 2
    alpha2_p = d2_p / (epsilon + beta2_p) ** 2
    sum_alpha_p = alpha0_p + alpha1_p + alpha2_p

    w0_p = alpha0_p / sum_alpha_p
    w1_p = alpha1_p / sum_alpha_p
    w2_p = alpha2_p / sum_alpha_p

    q0_p = (2 * vp0 - 7 * vp1 + 11 * vp2) / 6
    q1_p = (-vp1 + 5 * vp2 + 2 * vp3) / 6
    q2_p = (2 * vp2 + 5 * vp3 - vp4) / 6

    hat_fp = w0_p * q0_p + w1_p * q1_p + w2_p * q2_p

    vm0 = np.roll(fm, -3)
    vm1 = np.roll(fm, -2)
    vm2 = np.roll(fm, -1)
    vm3 = fm
    vm4 = np.roll(fm, 1)

    beta0_m = (13 / 12) * (vm0 - 2 * vm1 + vm2) ** 2 + (1 / 4) * (vm0 - 4 * vm1 + 3 * vm2) ** 2
    beta1_m = (13 / 12) * (vm1 - 2 * vm2 + vm3) ** 2 + (1 / 4) * (vm1 - vm3) ** 2
    beta2_m = (13 / 12) * (vm2 - 2 * vm3 + vm4) ** 2 + (1 / 4) * (3 * vm2 - 4 * vm3 + vm4) ** 2

    d0_m, d1_m, d2_m = linear_weights
    alpha0_m = d0_m / (epsilon + beta0_m) ** 2
    alpha1_m = d1_m / (epsilon + beta1_m) ** 2
    alpha2_m = d2_m / (epsilon + beta2_m) ** 2
    sum_alpha_m = alpha0_m + alpha1_m + alpha2_m

    w0_m = alpha0_m / sum_alpha_m
    w1_m = alpha1_m / sum_alpha_m
    w2_m = alpha2_m / sum_alpha_m

    q0_m = (2 * vm0 - 7 * vm1 + 11 * vm2) / 6
    q1_m = (-vm1 + 5 * vm2 + 2 * vm3) / 6
    q2_m = (2 * vm2 + 5 * vm3 - vm4) / 6

    hat_fm = w0_m * q0_m + w1_m * q1_m + w2_m * q2_m

    hat_f = hat_fp + hat_fm
    return hat_f


def compute_rhs(u, dx, nu=0.0, weno_epsilon=1e-6, linear_weights=(0.1, 0.6, 0.3)):
    """
    计算 du/dt = - d(hat_f)/dx + nu * d2u/dx2
    """
    hat_f = weno5_flux_splitting(u, epsilon=weno_epsilon, linear_weights=linear_weights)
    rhs_adv = -(hat_f - np.roll(hat_f, 1)) / dx

    if nu > 1e-9:
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx ** 2)
        return rhs_adv + nu * u_xx
    return rhs_adv


def rk3_step(u, dx, dt, nu=0.0, weno_epsilon=1e-6, linear_weights=(0.1, 0.6, 0.3)):
    """
    TVD Runge-Kutta 3 阶时间积分
    """
    rhs1 = compute_rhs(u, dx, nu=nu, weno_epsilon=weno_epsilon, linear_weights=linear_weights)
    u1 = u + dt * rhs1

    rhs2 = compute_rhs(u1, dx, nu=nu, weno_epsilon=weno_epsilon, linear_weights=linear_weights)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs2)

    rhs3 = compute_rhs(u2, dx, nu=nu, weno_epsilon=weno_epsilon, linear_weights=linear_weights)
    u_new = (1 / 3) * u + (2 / 3) * (u2 + dt * rhs3)

    return u_new
