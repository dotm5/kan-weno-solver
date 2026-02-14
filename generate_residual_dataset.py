import numpy as np
from burgers_weno import rk3_step
from numpy.lib.stride_tricks import sliding_window_view

def generate_random_state(N, num_modes=6):
    """生成包含高频成分的随机波形"""
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    u = np.zeros_like(x)
    for k in range(1, num_modes + 1):
        # 随机幅值和相位
        # 衰减系数设为 1.0，保留更多高频激波特征
        amp = np.random.uniform(0.1, 1.0) / (k**1.0) 
        phase = np.random.uniform(0, 2*np.pi)
        u += amp * np.sin(k * x + phase)
    # 归一化到 [-1, 1]
    u = u / (np.max(np.abs(u)) + 1e-8)
    return u

def get_multistep_error(u_coarse_init, N_coarse, N_fine, dt, steps):
    """
    计算 'steps' 步后的累积误差 (Accumulated Error)
    Error = Truth(t + steps*dt) - Baseline(t + steps*dt)
    """
    # 1. 准备 Truth (高精度，细网格，子步进)
    # 必须先插值到细网格才能跑 Truth
    u_fine = np.interp(np.linspace(0, 2*np.pi, N_fine, endpoint=False), 
                       np.linspace(0, 2*np.pi, N_coarse, endpoint=False), 
                       u_coarse_init)
    
    substeps_ratio = N_fine // N_coarse
    dt_fine = dt / substeps_ratio
    
    # Truth 跑 steps * ratio 步
    for _ in range(steps * substeps_ratio):
        u_fine = rk3_step(u_fine, 2*np.pi/N_fine, dt_fine, nu=0.0)
    
    # 降采样回粗网格
    u_truth_down = u_fine[::substeps_ratio]

    # 2. 准备 Baseline (粗网格 WENO5)
    u_weno = u_coarse_init.copy()
    dx_coarse = 2*np.pi / N_coarse
    
    for _ in range(steps):
        u_weno = rk3_step(u_weno, dx_coarse, dt, nu=0.0)
        
    return u_truth_down - u_weno

def main():
    # --- 参数配置 ---
    N_coarse = 128
    N_fine = 2048
    num_samples = 5000  # 样本数量
    cfl = 0.5           # 稍微加大 CFL 让误差明显一点
    STEPS_AHEAD = 10    # [关键] 预测未来 10 步的累积漂移
    STENCIL_SIZE = 9    # [关键] 增加视野到 9 点 (上帝视角)
    
    print(f"Generating Dataset: {num_samples} samples, Lookahead={STEPS_AHEAD} steps, Stencil={STENCIL_SIZE}...")
    
    X_list = []
    y_list = []
    
    dx_coarse = 2 * np.pi / N_coarse
    pad_size = STENCIL_SIZE // 2
    
    for i in range(num_samples):
        if i % 100 == 0: print(f"  Progress: {i}/{num_samples}")
        
        # 1. 生成初始状态 u^n
        u_coarse_n = generate_random_state(N_coarse)
        
        # 2. 决定 dt
        max_speed = np.max(np.abs(u_coarse_n)) + 1e-6
        dt = cfl * dx_coarse / max_speed
        
        # 3. 计算 10 步后的累积误差 (Ground Truth)
        accumulated_error = get_multistep_error(u_coarse_n, N_coarse, N_fine, dt, STEPS_AHEAD)
        
        # 4. 提取特征 (u^n 的 9点 Stencil)
        # 注意：输入依然是 t=n 时刻的状态，去预测 t=n+10 的误差
        u_padded = np.pad(u_coarse_n, (pad_size, pad_size), mode='wrap')
        stencils = sliding_window_view(u_padded, window_shape=STENCIL_SIZE)
        
        X_list.append(stencils)
        y_list.append(accumulated_error)
        
    X_data = np.vstack(X_list)
    y_data = np.concatenate(y_list).reshape(-1, 1)
    
    filename = "kan_train_data_multistep.npz"
    # 存入元数据，方便训练时读取配置
    np.savez(filename, 
             X=X_data, 
             y=y_data, 
             steps_ahead=STEPS_AHEAD,
             stencil_size=STENCIL_SIZE)
             
    print(f"Dataset saved to '{filename}'")
    print(f"X shape: {X_data.shape}, y shape: {y_data.shape}")

if __name__ == "__main__":
    main()