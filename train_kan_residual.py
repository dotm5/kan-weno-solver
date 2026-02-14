import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，防止 VSCode 崩溃
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Efficient KAN Layer (修正版)
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
        # 扩展 Grid 以支持 spline 计算
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
            self.spline_weight.data.copy_(
                (self.scale_spline if self.scale_spline is not None else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )

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
            self.layers.append(
                KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# 2. 自定义 Loss：关注激波 (Shock-Focusing Loss)
# ==========================================
class WeightedMSELoss(nn.Module):
    def __init__(self, weight_factor=10.0):
        super().__init__()
        self.factor = weight_factor
    
    def forward(self, pred, target):
        # 误差大的地方（通常是激波），权重线性增加
        weights = 1.0 + self.factor * torch.abs(target)
        # 也可以尝试平方权重：weights = 1.0 + self.factor * (target**2)
        loss = torch.mean(weights * (pred - target)**2)
        return loss

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    DATA_PATH = 'kan_train_data_multistep.npz'
    BATCH_SIZE = 512 
    EPOCHS = 150
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using Device: {DEVICE}")

    # 1. 加载数据
    try:
        data = np.load(DATA_PATH)
        X = data['X'] 
        y = data['y']
        # 读取元数据
        STENCIL_SIZE = int(data['stencil_size'])
        STEPS_AHEAD = int(data['steps_ahead'])
        print(f"Data Loaded. Stencil Size: {STENCIL_SIZE}, Prediction Horizon: {STEPS_AHEAD}")
    except Exception as e:
        print(f"Error loading data: {e}. Did you run 'generate_residual_dataset.py'?")
        return
    
    # 2. 预处理
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 3. 划分
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    # 4. 模型初始化 (动态调整输入维度)
    # 输入维度根据 Stencil 调整 (例如 9)
    # 隐层加宽到 32 以处理更多信息
    model = KAN([STENCIL_SIZE, 32, 1], grid_size=10, spline_order=3).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    # 移除了 verbose=True 参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    # 使用加权 Loss，重点攻克激波误差
    criterion = WeightedMSELoss(weight_factor=20.0) 
    # 验证时使用标准 MSE
    val_criterion = nn.MSELoss()

    # 5. 训练循环
    print("Starting Training with Weighted Loss...")
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, X_train_t.size()[0], BATCH_SIZE):
            optimizer.zero_grad()
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = val_criterion(val_out, y_val_t)
            
        scheduler.step(val_loss)
        train_losses.append(avg_loss)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Train(W-MSE): {avg_loss:.2e} | Val(MSE): {val_loss.item():.2e} | LR: {current_lr:.2e}")

    # 6. 保存模型 (存入 stencil_size 以便 rollout 读取)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler_X.mean_,
        'scaler_scale': scaler_X.scale_,
        'stencil_size': STENCIL_SIZE,
        'steps_ahead': STEPS_AHEAD
    }, 'kan_model_multistep.pth')
    
    print("Model saved to 'kan_model_multistep.pth'")
    
    # 简单画图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train (Weighted)')
    plt.plot(val_losses, label='Val (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Convergence')
    plt.savefig('kan_multistep_training.png')
    print("Plot saved.")

if __name__ == '__main__':
    main()