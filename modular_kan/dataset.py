import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

from .config import DataConfig

class HybridScaler:
    """
    Scales the input data using a combination of StandardScaling for stencil features
    and physics-informed scaling for physics features.
    """
    def __init__(self, stencil_size: int = 9):
        self.stencil_size = stencil_size
        self.stencil_scaler = StandardScaler()
        # Physics Params
        self.phys_ux_mean = 0.0
        self.phys_ux_std = 1.0
        self.phys_abs_max = 1.0
        self.phys_dt_max = 1.0

    def fit(self, X: np.ndarray) -> 'HybridScaler':
        """Fit the scaler to the training data."""
        # 1. Stencil (Standardization)
        X_stencil = X[:, :self.stencil_size]
        self.stencil_scaler.fit(X_stencil)
        
        # 2. Physics: [u_x, |u_x|, dt]
        u_x = X[:, self.stencil_size]
        abs_ux = X[:, self.stencil_size+1]
        dt = X[:, self.stencil_size+2]
        
        # u_x: Standardize
        self.phys_ux_mean = np.mean(u_x)
        self.phys_ux_std = np.std(u_x) + 1e-8
        
        # |u_x|: Use 99.9 percentile to handle outliers (shocks)
        self.phys_abs_max = np.percentile(abs_ux, 99.9) + 1e-8
        
        # dt: Use Max
        self.phys_dt_max = np.max(dt) + 1e-8
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted scaler."""
        X_stencil = X[:, :self.stencil_size]
        X_stencil_norm = self.stencil_scaler.transform(X_stencil)
        
        u_x = X[:, self.stencil_size]
        abs_ux = X[:, self.stencil_size+1]
        dt = X[:, self.stencil_size+2]
        
        u_x_norm = (u_x - self.phys_ux_mean) / self.phys_ux_std
        
        # Clip to prevent extreme values during inference
        abs_ux_norm = np.clip(abs_ux / self.phys_abs_max, 0.0, 1.0)
        dt_norm = dt / self.phys_dt_max
        
        X_phys_norm = np.stack([u_x_norm, abs_ux_norm, dt_norm], axis=1)
        return np.hstack([X_stencil_norm, X_phys_norm])

    def state_dict(self) -> Dict[str, Any]:
        return {
            'stencil_mean': self.stencil_scaler.mean_,
            'stencil_scale': self.stencil_scaler.scale_,
            'phys_params': (self.phys_ux_mean, self.phys_ux_std, self.phys_abs_max, self.phys_dt_max)
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.stencil_scaler.mean_ = state['stencil_mean']
        self.stencil_scaler.scale_ = state['stencil_scale']
        self.stencil_scaler.var_ = state['stencil_scale'] ** 2 
        
        params = state['phys_params']
        self.phys_ux_mean = params[0]
        self.phys_ux_std = params[1]
        self.phys_abs_max = params[2]
        self.phys_dt_max = params[3]

class PDEDataset(Dataset):
    """
    Dataset wrapper for PDE training data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, stencil_size: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.stencil_size = stencil_size

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: input, target, shock_metric
        # shock_metric is used for physics-informed loss (smooth vs shock)
        return self.X[idx], self.y[idx], self.X[idx, self.stencil_size + 1]

class PDEDataModule:
    """
    Manages data loading, splitting, and scaling.
    """
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = HybridScaler(stencil_size=config.stencil_size)

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, HybridScaler]:
        data = np.load(self.config.data_path)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.float32)
        
        # Check consistency
        if int(data['stencil_size']) != self.config.stencil_size:
            print(f"Warning: Config stencil_size ({self.config.stencil_size}) does not match data ({data['stencil_size']}). Using data's value.")
            self.config.stencil_size = int(data['stencil_size'])
            self.scaler = HybridScaler(stencil_size=self.config.stencil_size)

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42
        )

        X_train = self.scaler.fit(X_train_raw).transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)

        train_dataset = PDEDataset(X_train, y_train, self.config.stencil_size)
        val_dataset = PDEDataset(X_val, y_val, self.config.stencil_size)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers
        )

        return train_loader, val_loader, self.scaler
