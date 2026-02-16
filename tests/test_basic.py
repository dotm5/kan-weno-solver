import torch
import numpy as np
import pytest
from kan.model import GatedKAN
from solvers.weno import weno5_flux_splitting

def test_gated_kan_shape():
    """Verify GatedKAN output shapes."""
    stencil_size = 9
    phys_dim = 3
    batch_size = 4
    model = GatedKAN(stencil_size=stencil_size, phys_dim=phys_dim)
    
    # Mock input
    x = torch.randn(batch_size, stencil_size + phys_dim)
    correction, gate = model(x)
    
    assert correction.shape == (batch_size, 1)
    assert gate.shape == (batch_size, 1)
    assert (gate >= 0).all() and (gate <= 1).all()

def test_weno5_flux_splitting():
    """Basic sanity check for WENO5 flux splitting."""
    N = 64
    u = np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
    hat_f = weno5_flux_splitting(u)
    
    assert hat_f.shape == (N,)
    assert not np.isnan(hat_f).any()

def test_train_smoke():
    """Lightweight smoke test for training loop."""
    from train import train
    import numpy as np
    import os
    
    # Create tiny synthetic dataset
    data_path = 'tiny_data.npz'
    stencil_size = 9
    phys_dim = 3
    num_samples = 100
    X = np.random.randn(num_samples, stencil_size + phys_dim).astype(np.float32)
    y = np.random.randn(num_samples, 1).astype(np.float32)
    # train.py expects stencil_size and steps_ahead in the npz
    np.savez(data_path, X=X, y=y, stencil_size=stencil_size, steps_ahead=1)
    
    try:
        # Run for 1 epoch, small batch
        train(data_path=data_path, epochs=1, batch_size=10, device=torch.device('cpu'))
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists('kan_model.pth'):
            os.remove('kan_model.pth')
