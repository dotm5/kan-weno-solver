import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

from kan import GatedKAN, HybridScaler, PhysicsConsistentLoss


class PDEDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(data_path='kan_train_data.npz', epochs=100, batch_size=4096, lr=1e-3, accumulation_steps=4, device=None):
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 1. Load Data
    data = np.load(data_path)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    stencil_size = int(data['stencil_size'])
    steps_ahead = int(data['steps_ahead'])
    phys_dim = int(X.shape[1] - stencil_size)

    # 2. Split and Scale
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = HybridScaler(stencil_size=stencil_size)
    X_train = scaler.fit(X_train_raw).transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    train_dataset = PDEDataset(X_train, y_train)
    val_dataset = PDEDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda'), num_workers=0)

    # 3. Model & Optimizer
    model = GatedKAN(stencil_size=stencil_size, phys_dim=phys_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    criterion = PhysicsConsistentLoss()

    use_amp = device.type == 'cuda'
    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    shock_feature_idx = stencil_size + (1 if phys_dim > 1 else 0)

    # 4. Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                pred, gate_val = model(batch_X)
                loss, _, _ = criterion(pred, batch_y, gate_val, batch_X[:, shock_feature_idx])
                loss = loss / accumulation_steps

            grad_scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accumulation_steps
            del batch_X, batch_y, pred, gate_val, loss

        # Validation
        model.eval()
        val_mse_sum = 0.0
        gate_smooth_sum = 0.0
        gate_shock_sum = 0.0
        smooth_count = 0.0
        shock_count = 0.0

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_amp):
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                pred, gate = model(batch_X)
                weights = 1.0 + 5.0 * torch.abs(batch_y)
                mse = torch.mean(weights * (pred - batch_y)**2)
                val_mse_sum += mse.item() * batch_X.size(0)

                shock_metric = batch_X[:, shock_feature_idx]
                smooth_mask = (shock_metric < 0.1).float()
                shock_mask = 1.0 - smooth_mask

                gate_smooth_sum += torch.sum(gate * smooth_mask).item()
                smooth_count += torch.sum(smooth_mask).item()
                gate_shock_sum += torch.sum(gate * shock_mask).item()
                shock_count += torch.sum(shock_mask).item()

                del batch_X, batch_y, pred, gate, weights, mse, shock_metric, smooth_mask, shock_mask

        avg_val_mse = val_mse_sum / len(val_dataset)
        mean_gate_smooth = gate_smooth_sum / (smooth_count + 1e-6)
        mean_gate_shock = gate_shock_sum / (shock_count + 1e-6)

        scheduler.step(avg_val_mse)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss/len(train_loader):.1e} | Val W-MSE: {avg_val_mse:.1e} | Gate Sep: {mean_gate_shock-mean_gate_smooth:.3f}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

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
