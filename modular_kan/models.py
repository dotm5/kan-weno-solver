import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from .config import ModelConfig

class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Linear Layer.
    Uses B-splines for learnable activation functions.
    """
    def __init__(self, in_features: int, out_features: int, config: ModelConfig):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = config.grid_size
        self.spline_order = config.spline_order
        self.scale_noise = config.scale_noise
        self.scale_base = config.scale_base
        self.scale_spline = config.scale_spline
        
        # Base weights (residual connection)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = nn.SiLU()
        
        # Grid for B-splines
        h = (1.0 - (-1.0)) / self.grid_size # Assuming grid_range=[-1, 1]
        grid_range = [-1, 1]
        grid = ((torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * h) + grid_range[0])
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        
        # Spline weights
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, self.grid_size + self.spline_order))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * self.scale_noise / self.grid_size
            # Initialize spline weights to approximate identity or noise
            # Simplified initialization for this implementation
            # In a full implementation, we'd solve lstsq for identity approximation
            self.spline_weight.data.normal_(0, 0.1) 

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline bases for input x.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + 
                    (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        spline_bases = self.b_splines(x).view(x.size(0), -1)
        spline_output = F.linear(spline_bases, self.spline_weight.view(self.out_features, -1))
        
        return base_output + spline_output

class KAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network.
    """
    def __init__(self, layers_hidden: List[int], config: ModelConfig):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, config))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GatedKAN(nn.Module):
    """
    Physics-Informed Gated KAN.
    Consists of a Shape Net (KAN) and a Gate Net (MLP).
    """
    def __init__(self, stencil_size: int, phys_dim: int, config: ModelConfig):
        super(GatedKAN, self).__init__()
        self.stencil_size = stencil_size
        self.phys_dim = phys_dim
        
        # Shape Net: Full state input -> Correction
        # Input: Stencil + Physics features
        self.shape_net = KAN([stencil_size + phys_dim, config.hidden_dim, 1], config)
        
        # Gate Net: Physics features -> Gate value (0-1)
        self.gate_net = nn.Sequential(
            nn.Linear(phys_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Initialize Gate bias to start closed (approx 0.12)
        if hasattr(self.gate_net[-2], 'bias'):
             nn.init.constant_(self.gate_net[-2].bias, -2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_physics = x[:, self.stencil_size:]
        
        raw_correction = self.shape_net(x) 
        
        # Softsign limiting
        raw_correction = F.softsign(raw_correction) * 0.5
        
        gate = self.gate_net(x_physics)
        
        return raw_correction * gate, gate
