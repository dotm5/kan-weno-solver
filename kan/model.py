import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        if hasattr(self.gate_net[-2], 'bias'):
             nn.init.constant_(self.gate_net[-2].bias, -2.0)

    def forward(self, x):
        x_stencil = x[:, :self.stencil_size]
        x_physics = x[:, self.stencil_size:]
        
        raw_correction = self.shape_net(x) 
        
        # Softsign 限幅
        raw_correction = F.softsign(raw_correction) * 0.5
        
        gate = self.gate_net(x_physics)
        
        return raw_correction * gate, gate
