import torch
import torch.nn as nn
import torch.nn.functional as F

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


class HybridCorrectionLoss(nn.Module):
    """Prediction loss with modest locality bias for one-step correction."""

    def __init__(
        self,
        *,
        smooth_l1_beta=1.0,
        shock_weight=5.0,
        smooth_threshold=0.1,
        smooth_penalty_weight=0.0,
        locality_penalty_weight=0.0,
    ):
        super().__init__()
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.shock_weight = float(shock_weight)
        self.smooth_threshold = float(smooth_threshold)
        self.smooth_penalty_weight = float(smooth_penalty_weight)
        self.locality_penalty_weight = float(locality_penalty_weight)

    def forward(
        self,
        pred_z,
        target_z,
        *,
        shock_metric,
        pred_phys,
    ):
        shock_metric = shock_metric.reshape(-1, 1)
        smooth_mask = (shock_metric < self.smooth_threshold).to(pred_z.dtype)
        pointwise = F.smooth_l1_loss(
            pred_z,
            target_z,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        weights = 1.0 + self.shock_weight * shock_metric
        weighted_smooth_l1 = torch.mean(pointwise * weights)

        smooth_penalty = torch.mean(torch.abs(pred_phys) * smooth_mask) * self.smooth_penalty_weight
        locality_penalty = torch.mean(torch.abs(pred_phys) * (1.0 - shock_metric)) * self.locality_penalty_weight

        total = weighted_smooth_l1 + smooth_penalty + locality_penalty
        return {
            "total": total,
            "weighted_smooth_l1": weighted_smooth_l1,
            "smooth_penalty": smooth_penalty,
            "locality_penalty": locality_penalty,
        }
