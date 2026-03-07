import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .config import PhysicsConfig

class PhysicsLoss(nn.Module, ABC):
    """
    Abstract base class for physics-informed loss functions.
    """
    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor, gate_val: torch.Tensor, shock_metric: torch.Tensor) -> torch.Tensor:
        pass

class KANPhysicsLoss(PhysicsLoss):
    """
    Physics-consistent loss for KAN-based PDE solvers.
    Combines Weighted MSE (focusing on shocks) and Physics-Driven Sparsity (gate regularization).
    """
    def __init__(self, config: PhysicsConfig):
        super().__init__(config)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, gate_val: torch.Tensor, shock_metric: torch.Tensor):
        # 1. Weighted MSE
        # Error in shock regions (high target value) is penalized more heavily.
        weights = 1.0 + self.config.shock_weight * torch.abs(target)
        mse_loss = torch.mean(weights * (pred - target)**2)
        
        # 2. Physics-Driven Sparsity
        # Enforce sparsity (gate close to 0) in smooth regions.
        # shock_metric is usually |u_x| normalized.
        smooth_mask = (shock_metric < self.config.smooth_threshold).float()
        sparsity_loss = self.config.gate_sparsity * torch.mean(gate_val * smooth_mask)
        
        total_loss = mse_loss + sparsity_loss
        return total_loss, mse_loss, sparsity_loss
