import torch
import numpy as np
import matplotlib.pyplot as plt
from burgers_weno import rk3_step
from numpy.lib.stride_tricks import sliding_window_view
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from kan import GatedKAN, HybridScaler

# ==========================================
# 3. KAN 预测器 (v4.1)
# ==========================================
class KANPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # Weights_only=False 以加载 scaler 的 numpy 数据
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        self.stencil_size = checkpoint.get('stencil_size', 9)
        self.phys_dim = checkpoint.get('phys_dim', 3)
        self.steps_ahead = checkpoint.get('steps_ahead', 10)
        
        print(f"Loading Model v4.1... (Steps Ahead: {self.steps_ahead})")
        
        # 1. 恢复 Scaler
        self.scaler = HybridScaler(stencil_size=self.stencil_size)
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        # 2. 恢复模型
        self.model = GatedKAN(stencil_size=self.stencil_size, phys_dim=self.phys_dim).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, u_current, dx, dt):
        # 1. Stencil
        pad = self.stencil_size // 2
        u_padded = np.pad(u_current, (pad, pad), mode='wrap')
        stencils = sliding_window_view(u_padded, window_shape=self.stencil_size)
        
        # 2. Physics Features (实时计算)
        u_x = np.gradient(u_current, dx)
        abs_ux = np.abs(u_x)
        dt_feat = np.full_like(u_x, dt)
        
        # Stack: (N, 9+3)
        phys_feats = np.stack([u_x, abs_ux, dt_feat], axis=1)
        inputs_raw = np.hstack([stencils, phys_feats])
        
        # 3. Normalize (使用训练好的参数)
        inputs_norm = self.scaler.transform(inputs_raw)
        inputs_tensor = torch.FloatTensor(inputs_norm).to(self.device)
        
        # 4. Inference
        with torch.no_grad():
            preds, gates = self.model(inputs_tensor)
            preds = preds.cpu().numpy().flatten()
            gates = gates.cpu().numpy().flatten()
            
        # 5. Tendency Conversion
        return preds / self.steps_ahead, gates

# ==========================================
# 4. 主程序
# ==========================================
def run_rollout():
    # 参数
    N_coarse = 128
    N_ref = 2048
    T_final = 1.5
    cfl = 0.4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = KANPredictor('kan_model_v4.1.pth', device)
    except FileNotFoundError:
        print("Model file not found. Please train v4.1 first.")
        return

    # 初始化
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
    
    history = {
        'time': [], 'l2_base': [], 'l2_hybrid': [], 
        'gate_mean': [], 'gate_max': []
    }
    
    print("Starting Rollout v4.1...")
    
    while t < T_final:
        max_speed = max(np.max(np.abs(u_coarse)), np.max(np.abs(u_hybrid))) + 1e-6
        dt = cfl * dx_coarse / max_speed
        if t + dt > T_final: dt = T_final - t
        
        # 1. Truth
        dt_ref = dt / ref_substeps
        for _ in range(ref_substeps):
            u_truth = rk3_step(u_truth, dx_ref, dt_ref, nu=0.0)
        u_truth_down = u_truth[::ref_substeps]
        
        # 2. Baseline
        u_coarse = rk3_step(u_coarse, dx_coarse, dt, nu=0.0)
        
        # 3. Hybrid
        u_phys = rk3_step(u_hybrid, dx_coarse, dt, nu=0.0)
        
        # [NEW] 传入 dt，让模型自适应时间步
        correction, gates = predictor.predict(u_hybrid, dx_coarse, dt)
        
        # 安全归零 (虽然模型本身应该已经学会了)
        correction -= np.mean(correction)
        
        u_hybrid = u_phys + correction
        
        t += dt
        l2_base = np.sqrt(np.mean((u_coarse - u_truth_down)**2))
        l2_hybr = np.sqrt(np.mean((u_hybrid - u_truth_down)**2))
        
        history['time'].append(t)
        history['l2_base'].append(l2_base)
        history['l2_hybrid'].append(l2_hybr)
        history['gate_mean'].append(np.mean(gates))
        history['gate_max'].append(np.max(gates))
        
        if l2_hybr > 10.0: break

    # 绘图
    plt.figure(figsize=(15, 5))
    
    # Error
    plt.subplot(1, 3, 1)
    plt.plot(history['time'], history['l2_base'], 'k--', label='Baseline')
    plt.plot(history['time'], history['l2_hybrid'], 'r-', label='Hybrid v4.1')
    plt.yscale('log')
    plt.title('L2 Error')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Gate Activity
    plt.subplot(1, 3, 2)
    plt.plot(history['time'], history['gate_mean'], label='Mean Gate')
    plt.plot(history['time'], history['gate_max'], label='Max Gate')
    plt.title('Limiter Activity (Gate)')
    plt.legend()
    plt.grid(True)
    
    # Final Waveform
    plt.subplot(1, 3, 3)
    plt.plot(u_truth_down, 'k-', alpha=0.3, label='Truth')
    plt.plot(u_hybrid, 'r-', label='Hybrid')
    plt.title(f'Waveform at T={t:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rollout_v4.1_result.png')
    print("Done. Saved 'rollout_v4.1_result.png'.")

if __name__ == '__main__':
    run_rollout()