import torch
import numpy as np
import matplotlib.pyplot as plt
from solvers.weno import rk3_step
from numpy.lib.stride_tricks import sliding_window_view
from kan import GatedKAN, HybridScaler

class KANPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.stencil_size = checkpoint.get('stencil_size', 9)
        self.phys_dim = checkpoint.get('phys_dim', 3)
        self.steps_ahead = checkpoint.get('steps_ahead', 10)
        
        self.scaler = HybridScaler(stencil_size=self.stencil_size)
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        self.model = GatedKAN(stencil_size=self.stencil_size, phys_dim=self.phys_dim).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, u_current, dx, dt):
        pad = self.stencil_size // 2
        u_padded = np.pad(u_current, (pad, pad), mode='wrap')
        stencils = sliding_window_view(u_padded, window_shape=self.stencil_size)
        
        u_x = np.gradient(u_current, dx)
        abs_ux = np.abs(u_x)
        dt_feat = np.full_like(u_x, dt)
        
        inputs_raw = np.hstack([stencils, np.stack([u_x, abs_ux, dt_feat], axis=1)])
        inputs_norm = self.scaler.transform(inputs_raw)
        inputs_tensor = torch.FloatTensor(inputs_norm).to(self.device)
        
        with torch.no_grad():
            preds, gates = self.model(inputs_tensor)
            return preds.cpu().numpy().flatten() / self.steps_ahead, gates.cpu().numpy().flatten()

def run_evaluation(model_path='kan_model.pth'):
    N_coarse, N_ref = 128, 2048
    T_final, cfl = 1.5, 0.4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = KANPredictor(model_path, device)
    except:
        print("Model not found. Run training first.")
        return

    x = np.linspace(0, 2*np.pi, N_ref, endpoint=False)
    u_ref_init = 1.0 * np.sin(x) + 0.5 * np.sin(2*x + 0.5) - 0.2 * np.cos(5*x)
    u_ref_init /= np.max(np.abs(u_ref_init))
    
    u_coarse = u_ref_init[::(N_ref//N_coarse)].copy()
    u_hybrid = u_coarse.copy()
    u_truth = u_ref_init.copy()
    
    t, dx_coarse, dx_ref = 0.0, 2*np.pi/N_coarse, 2*np.pi/N_ref
    ref_substeps = N_ref // N_coarse
    
    history = {'time': [], 'l2_base': [], 'l2_hybrid': []}
    
    print("Starting Rollout Evaluation...")
    while t < T_final:
        dt = cfl * dx_coarse / (max(np.max(np.abs(u_coarse)), np.max(np.abs(u_hybrid))) + 1e-6)
        if t + dt > T_final: dt = T_final - t
        
        for _ in range(ref_substeps): u_truth = rk3_step(u_truth, dx_ref, dt/ref_substeps, nu=0.0)
        u_coarse = rk3_step(u_coarse, dx_coarse, dt, nu=0.0)
        u_phys = rk3_step(u_hybrid, dx_coarse, dt, nu=0.0)
        
        corr, _ = predictor.predict(u_hybrid, dx_coarse, dt)
        u_hybrid = u_phys + (corr - np.mean(corr))
        
        t += dt
        u_truth_down = u_truth[::ref_substeps]
        history['time'].append(t)
        history['l2_base'].append(np.sqrt(np.mean((u_coarse - u_truth_down)**2)))
        history['l2_hybrid'].append(np.sqrt(np.mean((u_hybrid - u_truth_down)**2)))

    plt.figure(figsize=(10, 5))
    plt.plot(history['time'], history['l2_base'], 'k--', label='Baseline (WENO5)')
    plt.plot(history['time'], history['l2_hybrid'], 'r-', label='Hybrid (WENO5+KAN)')
    plt.yscale('log'); plt.legend(); plt.title('L2 Error Comparison'); plt.grid(True)
    plt.savefig('evaluation_result.png')
    print("Evaluation complete. Results saved to 'evaluation_result.png'.")

if __name__ == "__main__":
    run_evaluation()
