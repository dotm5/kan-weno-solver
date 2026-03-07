import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solvers.weno import rk3_step

def solve_burgers_inviscid(N, T_final, cfl=0.1):
    """辅助函数：运行一次完整模拟"""
    L = 2 * np.pi
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    
    # 初始条件: u = 0.5 + sin(x)，加 0.5 避免在 0 附近徘徊，
    # 虽然 Lax-Friedrichs 可以处理 0，但偏移后能测试通量分裂逻辑
    u = 0.5 + np.sin(x)
    
    t = 0.0
    while t < T_final:
        max_speed = np.max(np.abs(u))
        if max_speed < 1e-8: max_speed = 1.0
        
        dt = cfl * dx / max_speed
        if t + dt > T_final: dt = T_final - t
        
        # 关键：nu=0.0 用于测试纯 WENO5 精度
        u = rk3_step(u, dx, dt, nu=0.0)
        t += dt
        
    return x, u

def main():
    # 参数设置：必须在激波形成前 (T=0.5 安全)
    T_final = 0.5 
    N_list = [32, 64, 128, 256, 512]
    N_ref = 2048
    
    print(f"--- Running WENO5 Convergence Test (T={T_final}, Inviscid) ---")
    
    # 1. 计算参考解
    print(f"Computing Reference (N={N_ref})...")
    _, u_ref = solve_burgers_inviscid(N_ref, T_final)
    
    errors_l2 = []
    dx_list = []
    
    # 2. 循环测试
    for N in N_list:
        dx = 2 * np.pi / N
        dx_list.append(dx)
        
        _, u_num = solve_burgers_inviscid(N, T_final)
        
        # 3. 误差计算 (降采样，无插值误差)
        step = N_ref // N
        if N_ref % N != 0: raise ValueError("N_ref must be multiple of N")
        
        u_exact = u_ref[::step]
        error = u_num - u_exact
        
        l2_err = np.sqrt(np.sum(error**2) * dx)
        errors_l2.append(l2_err)
        
        print(f"N={N:3d} | L2 Error={l2_err:.4e}")

    # 3. 计算阶数
    orders = []
    for i in range(1, len(errors_l2)):
        r = np.log(errors_l2[i-1]/errors_l2[i]) / np.log(dx_list[i-1]/dx_list[i])
        orders.append(r)
    
    print("\nCalculated Orders:", [f"{o:.2f}" for o in orders])
    
    # 4. 绘图 (可选)
    plt.loglog(dx_list, errors_l2, 'o-', label='Error')
    plt.loglog(dx_list, [errors_l2[0]*(dx/dx_list[0])**5 for dx in dx_list], 'k--', label='Order 5')
    plt.legend()
    plt.title("WENO5 Convergence")
    plt.grid(True, which="both")
    plt.savefig("convergence_check.png")
    print("Plot saved to convergence_check.png")

if __name__ == "__main__":
    main()