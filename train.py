import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from kan import GatedKAN, HybridScaler, PhysicsConsistentLoss

def train(data_path='kan_train_data.npz', epochs=200, batch_size=1024, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 1. Load Data
    data = np.load(data_path)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    stencil_size = int(data['stencil_size'])
    steps_ahead = int(data['steps_ahead'])
    phys_dim = 3
    
    # 2. Split and Scale
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = HybridScaler(stencil_size=stencil_size)
    X_train = scaler.fit(X_train_raw).transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    val_shock_metric = X_val_t[:, stencil_size + 1]
    val_smooth_mask = (val_shock_metric < 0.1).float()
    val_shock_mask = 1.0 - val_smooth_mask

    # 3. Model & Optimizer
    model = GatedKAN(stencil_size=stencil_size, phys_dim=phys_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    criterion = PhysicsConsistentLoss()

    # 4. Loop
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train_t.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train_t[indices], y_train_t[indices]
            pred, gate_val = model(batch_X)
            loss, _, _ = criterion(pred, batch_y, gate_val, batch_X[:, stencil_size + 1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            val_pred, val_gate = model(X_val_t)
            weights = 1.0 + 5.0 * torch.abs(y_val_t)
            val_mse = torch.mean(weights * (val_pred - y_val_t)**2)
            
            mean_gate_smooth = torch.sum(val_gate * val_smooth_mask) / (torch.sum(val_smooth_mask) + 1e-6)
            mean_gate_shock = torch.sum(val_gate * val_shock_mask) / (torch.sum(val_shock_mask) + 1e-6)
            
        scheduler.step(val_mse)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss/(X_train_t.size(0)/batch_size):.1e} | Val W-MSE: {val_mse:.1e} | Gate Sep: {mean_gate_shock-mean_gate_smooth:.3f}")

    # 5. Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': scaler.state_dict(),
        'stencil_size': stencil_size,
        'phys_dim': phys_dim,
        'steps_ahead': steps_ahead
    }, 'kan_model.pth')
    print("Model saved to 'kan_model.pth'")

if __name__ == "__main__":
    train()
