import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 

from kan import GatedKAN, HybridScaler, PhysicsConsistentLoss

# ==========================================
# 4. 主训练流程 (v4.1 Final)
# ==========================================
def main():
    DATA_PATH = 'kan_v3.1_data.npz' 
    BATCH_SIZE = 1024
    EPOCHS = 200
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using Device: {DEVICE}")

    # 1. Load Data
    try:
        data = np.load(DATA_PATH)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.float32)
        
        STENCIL_SIZE = int(data['stencil_size'])
        # [Fix 1] 显式读取 STEPS_AHEAD，修复保存时的 NameError
        STEPS_AHEAD = int(data['steps_ahead']) 
        PHYS_DIM = 3
        
        print(f"Data Loaded. Shape: {X.shape}. Steps Ahead: {STEPS_AHEAD}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Split FIRST (Anti-Leakage)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Hybrid Normalization (Fit on Train ONLY)
    # [Upgrade] 内部使用了 Percentile Scaling
    scaler = HybridScaler(stencil_size=STENCIL_SIZE)
    
    X_train = scaler.fit(X_train_raw).transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    
    # 验证集物理特征 (|u_x|)
    val_shock_metric = X_val_t[:, STENCIL_SIZE + 1]
    # 基于 Percentile 后的值，0.1 意味着相对较大的梯度
    val_smooth_mask = (val_shock_metric < 0.1).float()
    val_shock_mask = 1.0 - val_smooth_mask

    # 4. Model Init
    model = GatedKAN(stencil_size=STENCIL_SIZE, phys_dim=PHYS_DIM).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    criterion = PhysicsConsistentLoss(shock_weight=5.0, gate_sparsity=5e-4, smooth_threshold=0.1)

    # 5. Training Loop
    print("Starting v4.1 Training (Robust & Monitored)...")
    
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, X_train_t.size()[0], BATCH_SIZE):
            optimizer.zero_grad()
            indices = permutation[i:i+BATCH_SIZE]
            
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]
            batch_shock_metric = batch_X[:, STENCIL_SIZE + 1]
            
            pred, gate_val = model(batch_X)
            
            loss, mse, sparsity = criterion(pred, batch_y, gate_val, batch_shock_metric)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_gate = model(X_val_t)
            
            weights = 1.0 + 5.0 * torch.abs(y_val_t)
            val_weighted_mse = torch.mean(weights * (val_pred - y_val_t)**2)
            
            gate_std = torch.std(val_gate).item()
            
            mean_gate_smooth = torch.sum(val_gate.flatten() * val_smooth_mask) / (torch.sum(val_smooth_mask) + 1e-6)
            mean_gate_shock = torch.sum(val_gate.flatten() * val_shock_mask) / (torch.sum(val_shock_mask) + 1e-6)
            
            # [Upgrade] Gate Separation Monitor
            gate_sep = mean_gate_shock - mean_gate_smooth
            
        scheduler.step(val_weighted_mse)
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:3d} | Loss: {avg_loss:.1e} | Val W-MSE: {val_weighted_mse:.1e} | "
                  f"Gate Sep: {gate_sep:.3f} (Sh:{mean_gate_shock:.2f} - Sm:{mean_gate_smooth:.2f})")

    # 6. Save Model
    scaler_state = scaler.state_dict()
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': scaler_state, 
        'stencil_size': STENCIL_SIZE,
        'phys_dim': PHYS_DIM,
        'steps_ahead': STEPS_AHEAD
    }, 'kan_model_v4.1.pth')
    
    print("Model v4.1 Saved. (Ready for Paper)")

if __name__ == '__main__':
    main()