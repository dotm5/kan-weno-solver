import torch
import numpy as np
import matplotlib.pyplot as plt
from burgers_weno import rk3_step
from numpy.lib.stride_tricks import sliding_window_view
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 必须复制的模型类定义 (必须与训练代码一致)
# ==========================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_base = scale_base
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h) + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1/2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if self.scale_spline is not None else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GatedKAN(nn.Module):
    def __init__(self, stencil_size=9, phys_dim=3, hidden_dim=32):
        super(GatedKAN, self).__init__()
        self.stencil_size = stencil_size
        self.shape_net = KAN([stencil_size + phys_dim, hidden_dim, 1], grid_size=10, spline_order=3)
        self.gate_net = nn.Sequential(
            nn.Linear(phys_dim, 16), nn.SiLU(),
            nn.Linear(16, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x_stencil = x[:, :self.stencil_size]
        x_physics = x[:, self.stencil_size:]
        raw_correction = F.softsign(self.shape_net(x)) * 0.5
        gate = self.gate_net(x_physics)
        return raw_correction * gate, gate

# ==========================================
# 2. 混合归一化器 (推理模式)
# ==========================================
class HybridScaler:
    def __init__(self, stencil_size=9):
        self.stencil_size = stencil_size
        self.stencil_scaler = StandardScaler()
        # 初始化占位
        self.phys_ux_mean = 0.0
        self.phys_ux_std = 1.0
        self.phys_abs_max = 1.0
        self.phys_dt_max = 1.0

    def load_state_dict(self, state):
        """[关键] 从保存的参数中恢复"""
        self.stencil_scaler.mean_ = state['stencil_mean']
        self.stencil_scaler.scale_ = state['stencil_scale']
        # Sklearn 的 StandardScaler 需要设定 var_ 才能正常工作，虽然 transform 用不到
        self.stencil_scaler.var_ = state['stencil_scale'] ** 2 
        
        params = state['phys_params']
        self.phys_ux_mean = params[0]
        self.phys_ux_std = params[1]
        self.phys_abs_max = params[2]
        self.phys_dt_max = params[3]
        print(f"Scaler Loaded. |u_x| max threshold: {self.phys_abs_max:.4f}")

    def transform(self, X):
        X_stencil = X[:, :self.stencil_size]
        X_stencil_norm = self.stencil_scaler.transform(X_stencil)
        
        u_x = X[:, self.stencil_size]
        abs_ux = X[:, self.stencil_size+1]
        dt = X[:, self.stencil_size+2]
        
        u_x_norm = (u_x - self.phys_ux_mean) / self.phys_ux_std
        # 使用加载的阈值进行截断
        abs_ux_norm = np.clip(abs_ux / self.phys_abs_max, 0.0, 1.0)
        dt_norm = dt / self.phys_dt_max
        
        X_phys_norm = np.stack([u_x_norm, abs_ux_norm, dt_norm], axis=1)
        return np.hstack([X_stencil_norm, X_phys_norm])

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