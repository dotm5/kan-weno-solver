import numpy as np

def weno5_flux_splitting(u):
    """
    向量化的 WENO5 通量重构 (Lax-Friedrichs Flux Splitting)。
    
    原理:
    将物理通量 f(u) 分裂为 f+(u) 和 f-(u)。
    f+(u) 对应特征值 > 0 (波向右传)，使用左偏模板重构。
    f-(u) 对应特征值 < 0 (波向左传)，使用右偏模板重构。
    
    Args:
        u (np.ndarray): 守恒量 u (形状: [N])
        
    Returns:
        hat_f (np.ndarray): 界面处 i+1/2 的数值通量 (形状: [N])
    """
    # 1. 通量分裂 (Lax-Friedrichs)
    # f(u) = u^2 / 2
    # alpha = max|u| (全局最大特征速度)
    alpha = np.max(np.abs(u))
    
    # f+ 和 f-
    # f+ = 0.5 * (f(u) + alpha * u)
    # f- = 0.5 * (f(u) - alpha * u)
    f_val = 0.5 * u**2
    fp = 0.5 * (f_val + alpha * u)
    fm = 0.5 * (f_val - alpha * u)

    # 2. 准备模板 (利用 np.roll 实现向量化移位)
    # 我们需要重构 i+1/2 处的通量
    
    # --- 正通量 fp (左偏模板) ---
    # 模板: i-2, i-1, i, i+1, i+2
    vp0 = np.roll(fp, 2)   # i-2
    vp1 = np.roll(fp, 1)   # i-1
    vp2 = fp               # i
    vp3 = np.roll(fp, -1)  # i+1
    vp4 = np.roll(fp, -2)  # i+2
    
    # 光滑度指标 (Beta)
    beta0_p = (13/12) * (vp0 - 2*vp1 + vp2)**2 + (1/4) * (vp0 - 4*vp1 + 3*vp2)**2
    beta1_p = (13/12) * (vp1 - 2*vp2 + vp3)**2 + (1/4) * (vp1 - vp3)**2
    beta2_p = (13/12) * (vp2 - 2*vp3 + vp4)**2 + (1/4) * (3*vp2 - 4*vp3 + vp4)**2
    
    # WENO 权重
    epsilon = 1e-6
    d0_p, d1_p, d2_p = 0.1, 0.6, 0.3
    alpha0_p = d0_p / (epsilon + beta0_p)**2
    alpha1_p = d1_p / (epsilon + beta1_p)**2
    alpha2_p = d2_p / (epsilon + beta2_p)**2
    sum_alpha_p = alpha0_p + alpha1_p + alpha2_p
    
    w0_p = alpha0_p / sum_alpha_p
    w1_p = alpha1_p / sum_alpha_p
    w2_p = alpha2_p / sum_alpha_p
    
    # 候选通量 (Candidate Fluxes)
    q0_p = (2*vp0 - 7*vp1 + 11*vp2) / 6
    q1_p = (-vp1 + 5*vp2 + 2*vp3) / 6
    q2_p = (2*vp2 + 5*vp3 - vp4) / 6
    
    hat_fp = w0_p * q0_p + w1_p * q1_p + w2_p * q2_p

    # --- 负通量 fm (右偏模板) ---
    # 镜像对称: 直接利用 fp 的逻辑，但输入序列反转或重新索引
    # 这里我们显式写出对应 i+1/2 的右偏模板: i+3, i+2, i+1, i, i-1
    vm0 = np.roll(fm, -3) # i+3
    vm1 = np.roll(fm, -2) # i+2
    vm2 = np.roll(fm, -1) # i+1
    vm3 = fm              # i
    vm4 = np.roll(fm, 1)  # i-1
    
    # 光滑度指标
    beta0_m = (13/12) * (vm0 - 2*vm1 + vm2)**2 + (1/4) * (vm0 - 4*vm1 + 3*vm2)**2
    beta1_m = (13/12) * (vm1 - 2*vm2 + vm3)**2 + (1/4) * (vm1 - vm3)**2
    beta2_m = (13/12) * (vm2 - 2*vm3 + vm4)**2 + (1/4) * (3*vm2 - 4*vm3 + vm4)**2
    
    # WENO 权重 (d系数与正通量相反: 0.3, 0.6, 0.1)
    d0_m, d1_m, d2_m = 0.1, 0.6, 0.3 # 注意：这里的d定义对应于beta的顺序
    alpha0_m = d0_m / (epsilon + beta0_m)**2
    alpha1_m = d1_m / (epsilon + beta1_m)**2
    alpha2_m = d2_m / (epsilon + beta2_m)**2
    sum_alpha_m = alpha0_m + alpha1_m + alpha2_m
    
    w0_m = alpha0_m / sum_alpha_m
    w1_m = alpha1_m / sum_alpha_m
    w2_m = alpha2_m / sum_alpha_m
    
    # 候选通量
    q0_m = (2*vm0 - 7*vm1 + 11*vm2) / 6
    q1_m = (-vm1 + 5*vm2 + 2*vm3) / 6
    q2_m = (2*vm2 + 5*vm3 - vm4) / 6
    
    hat_fm = w0_m * q0_m + w1_m * q1_m + w2_m * q2_m

    # 3. 组合数值通量
    hat_f = hat_fp + hat_fm
    return hat_f

def compute_rhs(u, dx, nu=0.0):
    """
    计算 du/dt = - d(hat_f)/dx + nu * d2u/dx2
    """
    # 1. 对流项 (WENO5)
    # hat_f[i] 是 i+1/2 处的通量
    # hat_f[i-1] 是 i-1/2 处的通量 (通过 roll 获得)
    hat_f = weno5_flux_splitting(u)
    rhs_adv = -(hat_f - np.roll(hat_f, 1)) / dx
    
    # 2. 粘性项 (二阶中心差分，仅当 nu > 0 时计算)
    if nu > 1e-9:
        # u_{i+1} - 2u_i + u_{i-1}
        u_xx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)
        return rhs_adv + nu * u_xx
    else:
        return rhs_adv

def rk3_step(u, dx, dt, nu=0.0):
    """
    TVD Runge-Kutta 3 阶时间积分
    """
    # Stage 1
    rhs1 = compute_rhs(u, dx, nu)
    u1 = u + dt * rhs1
    
    # Stage 2
    rhs2 = compute_rhs(u1, dx, nu)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs2)
    
    # Stage 3
    rhs3 = compute_rhs(u2, dx, nu)
    u_new = (1/3) * u + (2/3) * (u2 + dt * rhs3)
    
    return u_new