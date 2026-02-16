import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_factor=10.0):
        super().__init__()
        self.factor = weight_factor
    
    def forward(self, pred, target):
        # 误差大的地方（通常是激波），权重线性增加
        weights = 1.0 + self.factor * torch.abs(target)
        loss = torch.mean(weights * (pred - target)**2)
        return loss

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
