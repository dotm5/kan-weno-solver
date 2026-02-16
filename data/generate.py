import numpy as np
from solvers.weno import rk3_step
from numpy.lib.stride_tricks import sliding_window_view

def generate_random_state(N, num_modes=6):
    """Generate random wave with high-frequency components"""
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    u = np.zeros_like(x)
    for k in range(1, num_modes + 1):
        amp = np.random.uniform(0.1, 1.0) / (k**1.0) 
        phase = np.random.uniform(0, 2*np.pi)
        u += amp * np.sin(k * x + phase)
    u = u / (np.max(np.abs(u)) + 1e-8)
    return u

def get_multistep_error(u_coarse_init, N_coarse, N_fine, dt, steps):
    """Calculate accumulated error over 'steps'"""
    u_fine = np.interp(np.linspace(0, 2*np.pi, N_fine, endpoint=False), 
                       np.linspace(0, 2*np.pi, N_coarse, endpoint=False), 
                       u_coarse_init)
    
    substeps_ratio = N_fine // N_coarse
    dt_fine = dt / substeps_ratio
    
    for _ in range(steps * substeps_ratio):
        u_fine = rk3_step(u_fine, 2*np.pi/N_fine, dt_fine, nu=0.0)
    
    u_truth_down = u_fine[::substeps_ratio]
    u_weno = u_coarse_init.copy()
    dx_coarse = 2*np.pi / N_coarse
    
    for _ in range(steps):
        u_weno = rk3_step(u_weno, dx_coarse, dt, nu=0.0)
        
    return u_truth_down - u_weno

def generate_dataset(num_samples=5000, N_coarse=128, N_fine=2048, steps_ahead=10, stencil_size=9, cfl=0.5):
    print(f"Generating Dataset: {num_samples} samples, Lookahead={steps_ahead} steps...")
    
    X_list = []
    y_list = []
    
    dx_coarse = 2 * np.pi / N_coarse
    pad_size = stencil_size // 2
    
    for i in range(num_samples):
        if i % 500 == 0: print(f"  Progress: {i}/{num_samples}")
        
        u_n = generate_random_state(N_coarse)
        dt = cfl * dx_coarse / (np.max(np.abs(u_n)) + 1e-6)
        
        err = get_multistep_error(u_n, N_coarse, N_fine, dt, steps_ahead)
        
        # Physics features for v4.1 (Gated KAN)
        u_x = np.gradient(u_n, dx_coarse)
        abs_ux = np.abs(u_x)
        dt_feat = np.full_like(u_x, dt)
        
        u_padded = np.pad(u_n, (pad_size, pad_size), mode='wrap')
        stencils = sliding_window_view(u_padded, window_shape=stencil_size)
        
        # Stack stencil + physics features [stencil(9), u_x, |u_x|, dt]
        phys_feats = np.stack([u_x, abs_ux, dt_feat], axis=1)
        X_combined = np.hstack([stencils, phys_feats])
        
        X_list.append(X_combined)
        y_list.append(err)
        
    X_data = np.vstack(X_list)
    y_data = np.concatenate(y_list).reshape(-1, 1)
    
    filename = "kan_train_data.npz"
    np.savez(filename, X=X_data, y=y_data, steps_ahead=steps_ahead, stencil_size=stencil_size)
    print(f"Dataset saved to '{filename}'. Shape: {X_data.shape}")

if __name__ == "__main__":
    generate_dataset()
