import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader
from .callbacks import Callback
from .config import TrainingConfig

class Solver(nn.Module, ABC):
    """
    Abstract Interface for a Problem Solver.
    Defines how a single batch is processed for training and validation.
    This decouples the Trainer from the specific physics/model logic.
    """
    def __init__(self):
        super().__init__()
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: Any = None
    
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        Returns a dictionary containing 'loss' and any other metrics.
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.
        Returns a dictionary containing metrics.
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """
        Setup optimizer and scheduler.
        """
        pass

class Trainer:
    """
    Generic Trainer that orchestrates the training loop.
    Agnostic to the underlying Model/Physics.
    """
    def __init__(
        self,
        solver: Solver,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: List[Callback] = None
    ):
        self.solver = solver
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        self.current_epoch = 0
        self.device = torch.device(config.device)
        
        self.solver.to(self.device)
        self.solver.configure_optimizers()

    def fit(self):
        print(f"Starting training on {self.device}")
        self._callback("on_train_begin")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            self._callback("on_epoch_begin")
            
            # Training Loop
            self.solver.train()
            train_logs = self._run_epoch(self.train_loader, mode="train")
            
            # Validation Loop
            self.solver.eval()
            val_logs = {}
            with torch.no_grad():
                val_logs = self._run_epoch(self.val_loader, mode="val")
            
            # Scheduler Step
            if self.solver.scheduler:
                # Assuming ReduceLROnPlateau, looking at 'val_loss'
                # If using other schedulers, this logic might need adjustment or be moved to solver
                if isinstance(self.solver.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.solver.scheduler.step(val_logs.get('val_loss', 0.0))
                else:
                    self.solver.scheduler.step()

            # Merge logs
            epoch_logs = {**train_logs, **val_logs}
            self._callback("on_epoch_end", logs=epoch_logs)

        self._callback("on_train_end")

    def _run_epoch(self, loader: DataLoader, mode: str) -> Dict[str, float]:
        total_loss = 0.0
        metrics_sum = {}
        count = 0
        
        for batch_idx, batch in enumerate(loader):
            self._callback("on_batch_begin")
            
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            elif isinstance(batch, dict):
                 batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            if mode == "train":
                self.solver.optimizer.zero_grad()
                outputs = self.solver.training_step(batch, batch_idx)
                loss = outputs['loss']
                loss.backward()
                # Optional: Clip grad norm could be a config or callback
                torch.nn.utils.clip_grad_norm_(self.solver.parameters(), 1.0)
                self.solver.optimizer.step()
            else:
                outputs = self.solver.validation_step(batch, batch_idx)
            
            # Aggregate metrics
            batch_size = loader.batch_size # Approximate
            count += 1 # Or batch_size if using weighted average
            
            for k, v in outputs.items():
                val = v.item() if torch.is_tensor(v) else v
                metrics_sum[k] = metrics_sum.get(k, 0.0) + val
                
            self._callback("on_batch_end")

        # Average metrics
        avg_metrics = {f"{mode}_{k}": v / count for k, v in metrics_sum.items()}
        return avg_metrics

    def _callback(self, method_name: str, logs: Dict[str, Any] = None):
        for cb in self.callbacks:
            getattr(cb, method_name)(self, logs)

    def save_checkpoint(self, path: str):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.solver.model.state_dict(),
            'optimizer_state_dict': self.solver.optimizer.state_dict(),
            'config': self.config
        }, path)
