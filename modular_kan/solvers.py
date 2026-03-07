import torch
import torch.optim as optim
from typing import Dict, Any, Tuple

from .trainer import Solver
from .models import GatedKAN
from .physics import KANPhysicsLoss
from .config import TrainingConfig, ModelConfig, PhysicsConfig

class KANSolver(Solver):
    """
    Concrete Solver for KAN-based PDE problems.
    Binds the GatedKAN model with the PhysicsConsistentLoss.
    """
    def __init__(self, 
                 model_config: ModelConfig, 
                 physics_config: PhysicsConfig, 
                 training_config: TrainingConfig,
                 stencil_size: int,
                 phys_dim: int):
        super().__init__()
        self.config = training_config
        self.model = GatedKAN(stencil_size=stencil_size, phys_dim=phys_dim, config=model_config)
        self.loss_fn = KANPhysicsLoss(physics_config)
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=self.config.patience, 
            factor=self.config.factor
        )

    def _common_step(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Unpack batch: input, target, shock_metric
        # Assumes PDEDataset returns (X, y, shock_metric)
        x, y, shock_metric = batch
        
        # Forward pass
        pred, gate_val = self.model(x)
        
        # Loss calculation
        loss, mse_loss, sparsity_loss = self.loss_fn(pred, y, gate_val, shock_metric)
        
        return loss, {
            'loss': loss,
            'mse': mse_loss,
            'sparsity': sparsity_loss,
            'gate_mean': gate_val.mean()
        }

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, metrics = self._common_step(batch)
        return metrics

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, metrics = self._common_step(batch)
        return metrics
