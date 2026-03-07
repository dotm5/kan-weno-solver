from typing import Dict, Any, Optional
import os
import torch

class Callback:
    """
    Base class for callbacks.
    """
    def on_train_begin(self, trainer, logs: Dict[str, Any] = None): pass
    def on_train_end(self, trainer, logs: Dict[str, Any] = None): pass
    def on_epoch_begin(self, trainer, logs: Dict[str, Any] = None): pass
    def on_epoch_end(self, trainer, logs: Dict[str, Any] = None): pass
    def on_batch_begin(self, trainer, logs: Dict[str, Any] = None): pass
    def on_batch_end(self, trainer, logs: Dict[str, Any] = None): pass

class LoggerCallback(Callback):
    """
    Simple console logger.
    """
    def on_epoch_end(self, trainer, logs: Dict[str, Any] = None):
        epoch = trainer.current_epoch
        if epoch % 1 == 0: # Log every epoch
            msg = f"Epoch {epoch:3d} | "
            for k, v in logs.items():
                if isinstance(v, float):
                    msg += f"{k}: {v:.2e} | "
            print(msg)

class CheckpointCallback(Callback):
    """
    Saves model checkpoints based on a monitored metric.
    """
    def __init__(self, save_dir: str, monitor: str = 'val_loss', mode: str = 'min'):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, logs: Dict[str, Any] = None):
        current = logs.get(self.monitor)
        if current is None:
            return

        save = False
        if self.mode == 'min' and current < self.best_score:
            self.best_score = current
            save = True
        elif self.mode == 'max' and current > self.best_score:
            self.best_score = current
            save = True

        if save:
            path = os.path.join(self.save_dir, f"best_model.pth")
            # Using the trainer's save method or accessing the model directly
            # Assuming trainer exposes 'model' or 'solver.model'
            if hasattr(trainer, 'save_checkpoint'):
                trainer.save_checkpoint(path)
            elif hasattr(trainer.solver, 'model'):
                torch.save(trainer.solver.model.state_dict(), path)
            print(f"Saved best model to {path} with {self.monitor}: {current:.4e}")
