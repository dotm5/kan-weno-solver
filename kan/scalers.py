import numpy as np
from sklearn.preprocessing import StandardScaler


class TargetAffineScaler:
    """Affine target scaler shared by training and evaluation."""

    def __init__(self, eps=1e-12, min_std=1e-8, clip_z=None):
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.clip_z = None if clip_z is None else float(clip_z)
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        self.mean = float(np.mean(y))
        raw_std = float(np.std(y))
        self.std = max(raw_std, self.min_std)
        if raw_std < self.min_std:
            print(
                f"Warning: target std={raw_std:.3e} < min_std={self.min_std:.3e}, using std floor."
            )
        return self

    def transform(self, y):
        y = np.asarray(y, dtype=np.float64)
        z = (y - self.mean) / max(self.std, self.eps)
        if self.clip_z is not None:
            z = np.clip(z, -self.clip_z, self.clip_z)
        return z

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.float64)
        return y * self.std + self.mean

    def state_dict(self):
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "min_std": float(self.min_std),
            "clip_z": self.clip_z,
            "eps": float(self.eps),
        }

    def load_state_dict(self, state):
        self.eps = float(state.get("eps", self.eps))
        self.min_std = float(state.get("min_std", self.min_std))
        self.clip_z = state.get("clip_z", self.clip_z)
        self.mean = float(state.get("mean", 0.0))
        std = float(state.get("std", 1.0))
        self.std = max(std, self.min_std)


class HybridScaler:
    def __init__(self, stencil_size=9, eps=1e-8, clip_percentile_abs_features=99.5):
        self.stencil_size = stencil_size
        self.stencil_scaler = StandardScaler()
        self.eps = eps
        self.clip_percentile_abs_features = float(clip_percentile_abs_features)

        self.phys_dim = 0
        self.phys_mean = np.zeros(0, dtype=np.float32)
        self.phys_std = np.ones(0, dtype=np.float32)
        self.phys_upper = np.ones(0, dtype=np.float32)
        self.phys_use_clip = np.zeros(0, dtype=bool)

    def fit(self, X):
        """Fit on train split only."""
        X_stencil = X[:, :self.stencil_size]
        self.stencil_scaler.fit(X_stencil)

        X_phys = X[:, self.stencil_size:]
        self.phys_dim = X_phys.shape[1]

        self.phys_mean = np.zeros(self.phys_dim, dtype=np.float32)
        self.phys_std = np.ones(self.phys_dim, dtype=np.float32)
        self.phys_upper = np.ones(self.phys_dim, dtype=np.float32)
        self.phys_use_clip = np.zeros(self.phys_dim, dtype=bool)

        for idx in range(self.phys_dim):
            col = X_phys[:, idx]
            if idx == 0:
                self.phys_mean[idx] = float(np.mean(col))
                self.phys_std[idx] = float(np.std(col) + self.eps)
            elif idx in (1, 2, 3):
                self.phys_upper[idx] = float(np.percentile(col, self.clip_percentile_abs_features) + self.eps)
                self.phys_use_clip[idx] = True
            elif idx == 4:
                self.phys_upper[idx] = float(np.max(col) + self.eps)
                self.phys_use_clip[idx] = True
            elif idx in (5, 6):
                # sin/cos timestamp embedding is already bounded.
                pass
            else:
                self.phys_mean[idx] = float(np.mean(col))
                self.phys_std[idx] = float(np.std(col) + self.eps)

        if self.phys_dim > 1:
            print(f"Scaler Fitted: |u_x| clip threshold = {self.phys_upper[1]:.4f}")
        return self

    def transform(self, X):
        X_stencil = X[:, :self.stencil_size]
        X_stencil_norm = self.stencil_scaler.transform(X_stencil)

        X_phys = X[:, self.stencil_size:]
        if X_phys.shape[1] != self.phys_dim:
            raise ValueError(
                f"Physics feature dim mismatch: got {X_phys.shape[1]}, expected {self.phys_dim}."
            )

        X_phys_norm = np.empty_like(X_phys, dtype=np.float32)
        for idx in range(self.phys_dim):
            col = X_phys[:, idx]
            if self.phys_use_clip[idx]:
                X_phys_norm[:, idx] = np.clip(col / self.phys_upper[idx], 0.0, 1.0)
            elif idx in (5, 6):
                X_phys_norm[:, idx] = np.clip(col, -1.0, 1.0)
            else:
                X_phys_norm[:, idx] = (col - self.phys_mean[idx]) / self.phys_std[idx]

        return np.hstack([X_stencil_norm, X_phys_norm])

    def state_dict(self):
        return {
            'stencil_mean': self.stencil_scaler.mean_,
            'stencil_scale': self.stencil_scaler.scale_,
            'phys_dim': int(self.phys_dim),
            'phys_mean': self.phys_mean,
            'phys_std': self.phys_std,
            'phys_upper': self.phys_upper,
            'phys_use_clip': self.phys_use_clip,
            'eps': self.eps,
            'clip_percentile_abs_features': self.clip_percentile_abs_features,
        }

    def load_state_dict(self, state):
        self.stencil_scaler.mean_ = state['stencil_mean']
        self.stencil_scaler.scale_ = state['stencil_scale']
        self.stencil_scaler.var_ = state['stencil_scale'] ** 2
        self.stencil_scaler.n_features_in_ = len(state['stencil_mean'])

        self.eps = float(state.get('eps', self.eps))
        self.clip_percentile_abs_features = float(
            state.get('clip_percentile_abs_features', self.clip_percentile_abs_features)
        )

        if 'phys_dim' in state:
            self.phys_dim = int(state['phys_dim'])
            self.phys_mean = np.asarray(state['phys_mean'], dtype=np.float32)
            self.phys_std = np.asarray(state['phys_std'], dtype=np.float32)
            self.phys_upper = np.asarray(state['phys_upper'], dtype=np.float32)
            self.phys_use_clip = np.asarray(state['phys_use_clip'], dtype=bool)
        else:
            # Backward compatibility for old checkpoints with 3 physics features.
            ux_mean, ux_std, abs_max, dt_max = state['phys_params']
            self.phys_dim = 3
            self.phys_mean = np.array([ux_mean, 0.0, 0.0], dtype=np.float32)
            self.phys_std = np.array([ux_std, 1.0, 1.0], dtype=np.float32)
            self.phys_upper = np.array([1.0, abs_max, dt_max], dtype=np.float32)
            self.phys_use_clip = np.array([False, True, True], dtype=bool)

        if self.phys_dim > 1:
            print(f"Scaler Loaded. |u_x| clip threshold: {self.phys_upper[1]:.4f}")
