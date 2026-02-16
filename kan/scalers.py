import numpy as np
from sklearn.preprocessing import StandardScaler

class HybridScaler:
    def __init__(self, stencil_size=9):
        self.stencil_size = stencil_size
        self.stencil_scaler = StandardScaler()
        # Physics Params
        self.phys_ux_mean = 0.0
        self.phys_ux_std = 1.0
        self.phys_abs_max = 1.0
        self.phys_dt_max = 1.0

    def fit(self, X):
        """只在训练集上调用"""
        # 1. Stencil (Standardization)
        X_stencil = X[:, :self.stencil_size]
        self.stencil_scaler.fit(X_stencil)
        
        # 2. Physics: [u_x, |u_x|, dt]
        u_x = X[:, self.stencil_size]
        abs_ux = X[:, self.stencil_size+1]
        dt = X[:, self.stencil_size+2]
        
        # u_x: 有正有负，用均值方差归一化
        self.phys_ux_mean = np.mean(u_x)
        self.phys_ux_std = np.std(u_x) + 1e-8
        
        # [Critical Upgrade] |u_x|: 使用 99.9 分位数而不是 Max
        # 防止个别极端激波导致大部分数据被压缩到 0
        self.phys_abs_max = np.percentile(abs_ux, 99.9) + 1e-8
        
        # dt: 使用 Max (dt 通常是离散的几个值，Max 没问题)
        self.phys_dt_max = np.max(dt) + 1e-8
        
        print(f"Scaler Fitted: |u_x| max clip threshold = {self.phys_abs_max:.4f}")
        return self

    def transform(self, X):
        """应用变换"""
        X_stencil = X[:, :self.stencil_size]
        X_stencil_norm = self.stencil_scaler.transform(X_stencil)
        
        u_x = X[:, self.stencil_size]
        abs_ux = X[:, self.stencil_size+1]
        dt = X[:, self.stencil_size+2]
        
        u_x_norm = (u_x - self.phys_ux_mean) / self.phys_ux_std
        
        # 这里的 clip 很重要，防止推理时出现比训练集还大的极端值
        abs_ux_norm = np.clip(abs_ux / self.phys_abs_max, 0.0, 1.0)
        dt_norm = dt / self.phys_dt_max
        
        X_phys_norm = np.stack([u_x_norm, abs_ux_norm, dt_norm], axis=1)
        return np.hstack([X_stencil_norm, X_phys_norm])

    def state_dict(self):
        return {
            'stencil_mean': self.stencil_scaler.mean_,
            'stencil_scale': self.stencil_scaler.scale_,
            'phys_params': (self.phys_ux_mean, self.phys_ux_std, self.phys_abs_max, self.phys_dt_max)
        }

    def load_state_dict(self, state):
        """[关键] 从保存的参数中恢复"""
        self.stencil_scaler.mean_ = state['stencil_mean']
        self.stencil_scaler.scale_ = state['stencil_scale']
        # Sklearn 的 StandardScaler 需要设定 var_ 才能正常工作，虽然 transform 用不到
        if hasattr(self.stencil_scaler, 'var_') or True:
             self.stencil_scaler.var_ = state['stencil_scale'] ** 2 
        
        params = state['phys_params']
        self.phys_ux_mean = params[0]
        self.phys_ux_std = params[1]
        self.phys_abs_max = params[2]
        self.phys_dt_max = params[3]
        print(f"Scaler Loaded. |u_x| max threshold: {self.phys_abs_max:.4f}")
