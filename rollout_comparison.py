import torch
import numpy as np
import matplotlib.pyplot as plt
from burgers_weno import rk3_step
from train_kan_residual import KAN, KANLinear # 确保能导入这些类
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# 1. KAN 预测器 (适配多步趋势)
# ==========================================
class KANPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # [Fix] weights_only=False 允许加载 Numpy 数组
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_scale = checkpoint['scaler_scale'] + 1e-8
        
        # 读取模型元数据
        self.stencil_size = checkpoint.get('stencil_size', 9)
        self.steps_ahead = checkpoint.get('steps_ahead', 1)
        
        print(f"Model loaded. Stencil={self.stencil_size}, Trained on {self.steps_ahead}-step error.")
        
        # 动态初始化结构
        self.model = KAN([self.stencil_size, 32, 1], grid_size=10, spline_order=3).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_tendency(self, u_current):
        # 1. Padding 适配 Stencil
        pad = self.stencil_size // 2
        u_padded = np.pad(u_current, (pad, pad), mode='wrap')
        stencils = sliding_window_view(u_padded, window_shape=self.stencil_size)
        
        # 2. Normalize
        stencils_norm = (stencils - self.scaler_mean) / self.scaler_scale
        inputs = torch.FloatTensor(stencils_norm).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            preds = self.model(inputs).cpu().numpy().flatten()
            
        # [关键逻辑]：将累积误差转化为“每步修正趋势”
        # 假设误差在 short horizon 内是线性的，分摊到每一步
        return preds / self.steps_ahead

# ==========================================
# 2. Rollout 对比实验
# ==========================================
def run_rollout():
    # 参数
    N_coarse = 128
    N_ref = 2048
    T_final = 1.5
    cfl = 0.4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = KANPredictor('kan_model_multistep.pth', device)
    except FileNotFoundError:
        print("Error: Model file 'kan_model_multistep.pth' not found.")
        return

    # 初始化 (未见过的测试波形)
    x = np.linspace(0, 2*np.pi, N_ref, endpoint=False)
    u_ref = 1.0 * np.sin(x) + 0.5 * np.sin(2*x + 0.5) - 0.2 * np.cos(5*x)
    u_ref = u_ref / np.max(np.abs(u_ref))
    
    u_coarse = u_ref[::(N_ref//N_coarse)].copy()
    u_hybrid = u_coarse.copy()
    u_truth = u_ref.copy()
    
    t = 0.0
    dx_coarse = 2*np.pi / N_coarse
    dx_ref = 2*np.pi / N_ref
    ref_substeps = N_ref // N_coarse
    
    history = {'time': [], 'l2_base': [], 'l2_hybrid': [], 'tv_hybrid': [], 'corr_mag': []}
    
    print("Starting Multi-step KAN Rollout...")
    
    while t < T_final:
        # CFL
        max_speed = max(np.max(np.abs(u_coarse)), np.max(np.abs(u_hybrid))) + 1e-6
        dt = cfl * dx_coarse / max_speed
        if t + dt > T_final: dt = T_final - t
        
        # 1. Truth (Sub-stepping)
        dt_ref = dt / ref_substeps
        for _ in range(ref_substeps):
            u_truth = rk3_step(u_truth, dx_ref, dt_ref, nu=0.0)
        u_truth_down = u_truth[::ref_substeps]
        
        # 2. Baseline
        u_coarse = rk3_step(u_coarse, dx_coarse, dt, nu=0.0)
        
        # 3. Hybrid
        # A. 物理步
        u_phys = rk3_step(u_hybrid, dx_coarse, dt, nu=0.0)
        
        # B. 神经网络倾向修正
        correction = predictor.predict_tendency(u_hybrid)
        
        # C. 安全机制 (Conservation + Clip)
        correction -= np.mean(correction)
        # 阈值可以小一点，因为这是单步的修正量，应该是微小的
        correction = np.clip(correction, -0.01, 0.01) 
        
        u_hybrid = u_phys + correction
        
        # 记录
        t += dt
        l2_base = np.sqrt(np.mean((u_coarse - u_truth_down)**2))
        l2_hybr = np.sqrt(np.mean((u_hybrid - u_truth_down)**2))
        tv_hybr = np.sum(np.abs(np.diff(u_hybrid))) + np.abs(u_hybrid[-1] - u_hybrid[0])
        
        history['time'].append(t)
        history['l2_base'].append(l2_base)
        history['l2_hybrid'].append(l2_hybr)
        history['tv_hybrid'].append(tv_hybr)
        history['corr_mag'].append(np.mean(np.abs(correction)))
        
        if l2_hybr > 10.0: 
            print("Hybrid Solver Exploded!")
            break

    # --- 绘图 ---
    print("Plotting results...")
    fig = plt.figure(figsize=(18, 6))
    
    # 图 1: L2 Error
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(history['time'], history['l2_base'], 'k--', label='Baseline')
    ax1.plot(history['time'], history['l2_hybrid'], 'r-', label='Hybrid KAN')
    ax1.set_yscale('log')
    ax1.set_title('L2 Error (Log Scale)')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    
    # 图 2: Correction Magnitude
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(history['time'], history['corr_mag'], 'g-')
    ax2.set_title('Mean Per-Step Correction')
    ax2.set_ylabel('|Correction|')
    ax2.grid(True)
    
    # 图 3: Total Variation
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(history['time'], history['tv_hybrid'], 'purple')
    ax3.set_title('Hybrid Total Variation')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('rollout_v2_result.png')
    print("Results saved to 'rollout_v2_result.png'")

if __name__ == '__main__':
    run_rollout()