import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 

# ==========================================
# 0. 混合归一化器 (v4.1 抗离群值版)
# ==========================================
class HybridScaler:
    def __init__(self, stencil_size=9):
        self.stencil_size = stencil_size
        self.stencil_scaler = StandardScaler()
        # Physics Params
        self.phys_ux_mean = 0.0
        self.phys_ux_std = 1.0
        self.phys_abs_max = 1.0
        self.phys_dt_max = 1.0

    def fit(self, X):
        """只在训练集上调用"""
        # 1. Stencil (Standardization)
        X_stencil = X[:, :self.stencil_size]
        self.stencil_scaler.fit(X_stencil)
        
        # 2. Physics: [u_x, |u_x|, dt]
        u_x = X[:, self.stencil_size]
        abs_ux = X[:, self.stencil_size+1]
        dt = X[:, self.stencil_size+2]
        
        # u_x: 有正有负，用均值方差归一化
        self.phys_ux_mean = np.mean(u_x)
        self.phys_ux_std = np.std(u_x) + 1e-8
        
        # [Critical Upgrade] |u_x|: 使用 99.9 分位数而不是 Max
        # 防止个别极端激波导致大部分数据被压缩到 0
        self.phys_abs_max = np.percentile(abs_ux, 99.9) + 1e-8
        
        # dt: 使用 Max (dt 通常是离散的几个值，Max 没问题)
        self.phys_dt_max = np.max(dt) + 1e-8
        
        print(f"Scaler Fitted: |u_x| max clip threshold = {self.phys_abs_max:.4f}")
        return self

    def transform(self, X):
        """应用变换"""
        X_stencil = X[:, :self.stencil_size]
        X_stencil_norm = self.stencil_scaler.transform(X_stencil)
        
        u_x = X[:, self.stencil_size]
        abs_ux = X[:, self.stencil_size+1]
        dt = X[:, self.stencil_size+2]
        
        u_x_norm = (u_x - self.phys_ux_mean) / self.phys_ux_std
        
        # 这里的 clip 很重要，防止推理时出现比训练集还大的极端值
        abs_ux_norm = np.clip(abs_ux / self.phys_abs_max, 0.0, 1.0)
        dt_norm = dt / self.phys_dt_max
        
        X_phys_norm = np.stack([u_x_norm, abs_ux_norm, dt_norm], axis=1)
        return np.hstack([X_stencil_norm, X_phys_norm])

    def state_dict(self):
        return {
            'stencil_mean': self.stencil_scaler.mean_,
            'stencil_scale': self.stencil_scaler.scale_,
            'phys_params': (self.phys_ux_mean, self.phys_ux_std, self.phys_abs_max, self.phys_dt_max)
        }

# ==========================================
# 1. KAN 核心组件 (保持不变)
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

# ==========================================
# 2. 物理门控网络 (v4.1 Final)
# ==========================================
class GatedKAN(nn.Module):
    def __init__(self, stencil_size=9, phys_dim=3, hidden_dim=32):
        super(GatedKAN, self).__init__()
        self.stencil_size = stencil_size
        self.phys_dim = phys_dim
        
        # Shape Net: 全状态输入
        self.shape_net = KAN([stencil_size + phys_dim, hidden_dim, 1], grid_size=10, spline_order=3)
        
        # Gate Net: 物理门控
        self.gate_net = nn.Sequential(
            nn.Linear(phys_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # 初始化 Gate 偏置为 -2.0 (Gate ≈ 0.12)
        nn.init.constant_(self.gate_net[-2].bias, -2.0)

    def forward(self, x):
        x_stencil = x[:, :self.stencil_size]
        x_physics = x[:, self.stencil_size:]
        
        raw_correction = self.shape_net(x) 
        
        # Softsign 限幅
        raw_correction = F.softsign(raw_correction) * 0.5
        
        gate = self.gate_net(x_physics)
        
        return raw_correction * gate, gate

# ==========================================
# 3. 物理一致损失函数 (v4.0)
# ==========================================
class PhysicsConsistentLoss(nn.Module):
    def __init__(self, shock_weight=5.0, gate_sparsity=5e-4, smooth_threshold=0.1):
        super().__init__()
        self.shock_weight = shock_weight
        self.gate_sparsity = gate_sparsity
        self.smooth_threshold = smooth_threshold 
    
    def forward(self, pred, target, gate_val, shock_metric):
        # 1. Weighted MSE
        weights = 1.0 + self.shock_weight * torch.abs(target)
        mse_loss = torch.mean(weights * (pred - target)**2)
        
        # 2. Physics-Driven Sparsity
        # shock_metric 是归一化后的 |u_x| (0~1)
        smooth_mask = (shock_metric < self.smooth_threshold).float()
        sparsity_loss = self.gate_sparsity * torch.mean(gate_val * smooth_mask)
        
        return mse_loss + sparsity_loss, mse_loss, sparsity_loss

# ==========================================
# 4. 主训练流程 (v4.1 Final)
# ==========================================
def main():
    DATA_PATH = 'kan_v3.1_data.npz' 
    BATCH_SIZE = 1024
    EPOCHS = 200
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using Device: {DEVICE}")

    # 1. Load Data
    try:
        data = np.load(DATA_PATH)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.float32)
        
        STENCIL_SIZE = int(data['stencil_size'])
        # [Fix 1] 显式读取 STEPS_AHEAD，修复保存时的 NameError
        STEPS_AHEAD = int(data['steps_ahead']) 
        PHYS_DIM = 3
        
        print(f"Data Loaded. Shape: {X.shape}. Steps Ahead: {STEPS_AHEAD}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Split FIRST (Anti-Leakage)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Hybrid Normalization (Fit on Train ONLY)
    # [Upgrade] 内部使用了 Percentile Scaling
    scaler = HybridScaler(stencil_size=STENCIL_SIZE)
    
    X_train = scaler.fit(X_train_raw).transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    
    # 验证集物理特征 (|u_x|)
    val_shock_metric = X_val_t[:, STENCIL_SIZE + 1]
    # 基于 Percentile 后的值，0.1 意味着相对较大的梯度
    val_smooth_mask = (val_shock_metric < 0.1).float()
    val_shock_mask = 1.0 - val_smooth_mask

    # 4. Model Init
    model = GatedKAN(stencil_size=STENCIL_SIZE, phys_dim=PHYS_DIM).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    criterion = PhysicsConsistentLoss(shock_weight=5.0, gate_sparsity=5e-4, smooth_threshold=0.1)

    # 5. Training Loop
    print("Starting v4.1 Training (Robust & Monitored)...")
    
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, X_train_t.size()[0], BATCH_SIZE):
            optimizer.zero_grad()
            indices = permutation[i:i+BATCH_SIZE]
            
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]
            batch_shock_metric = batch_X[:, STENCIL_SIZE + 1]
            
            pred, gate_val = model(batch_X)
            
            loss, mse, sparsity = criterion(pred, batch_y, gate_val, batch_shock_metric)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_gate = model(X_val_t)
            
            weights = 1.0 + 5.0 * torch.abs(y_val_t)
            val_weighted_mse = torch.mean(weights * (val_pred - y_val_t)**2)
            
            gate_std = torch.std(val_gate).item()
            
            mean_gate_smooth = torch.sum(val_gate.flatten() * val_smooth_mask) / (torch.sum(val_smooth_mask) + 1e-6)
            mean_gate_shock = torch.sum(val_gate.flatten() * val_shock_mask) / (torch.sum(val_shock_mask) + 1e-6)
            
            # [Upgrade] Gate Separation Monitor
            gate_sep = mean_gate_shock - mean_gate_smooth
            
        scheduler.step(val_weighted_mse)
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:3d} | Loss: {avg_loss:.1e} | Val W-MSE: {val_weighted_mse:.1e} | "
                  f"Gate Sep: {gate_sep:.3f} (Sh:{mean_gate_shock:.2f} - Sm:{mean_gate_smooth:.2f})")

    # 6. Save Model
    scaler_state = scaler.state_dict()
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': scaler_state, 
        'stencil_size': STENCIL_SIZE,
        'phys_dim': PHYS_DIM,
        'steps_ahead': STEPS_AHEAD
    }, 'kan_model_v4.1.pth')
    
    print("Model v4.1 Saved. (Ready for Paper)")

if __name__ == '__main__':
    main()