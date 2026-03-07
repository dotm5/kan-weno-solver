import torch
from .config import DataConfig, ModelConfig, PhysicsConfig, TrainingConfig
from .dataset import PDEDataModule
from .solvers import KANSolver
from .trainer import Trainer
from .callbacks import LoggerCallback, CheckpointCallback

def main():
    # 1. Configuration
    # In a real app, these could be loaded from CLI args or YAML
    data_cfg = DataConfig(
        data_path='kan_train_data.npz',
        stencil_size=9,
        phys_dim=3,
        batch_size=16
    )
    model_cfg = ModelConfig(
        hidden_dim=32,
        grid_size=10,
        spline_order=3
    )
    physics_cfg = PhysicsConfig(
        shock_weight=5.0,
        gate_sparsity=5e-4
    )
    train_cfg = TrainingConfig(
        epochs=100,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints"
    )

    print(f"Configuration Loaded. Device: {train_cfg.device}")

    # 2. Data Preparation
    data_module = PDEDataModule(data_cfg)
    try:
        train_loader, val_loader, scaler = data_module.prepare_data()
    except FileNotFoundError:
        print(f"Error: Data file '{data_cfg.data_path}' not found.")
        print("Please run 'python data/generate.py' first.")
        return

    # 3. Solver Initialization (Model + Physics + Optimization)
    solver = KANSolver(
        model_config=model_cfg,
        physics_config=physics_cfg,
        training_config=train_cfg,
        stencil_size=data_cfg.stencil_size,
        phys_dim=data_cfg.phys_dim
    )

    # 4. Callbacks
    callbacks = [
        LoggerCallback(),
        CheckpointCallback(save_dir=train_cfg.save_dir, monitor='val_loss', mode='min')
    ]

    # 5. Trainer Initialization
    trainer = Trainer(
        solver=solver,
        config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks
    )

    # 6. Start Training
    trainer.fit()
    
    # 7. Save Scaler State (for inference)
    import pickle
    # Or integrate into CheckpointCallback, but simple here:
    # Actually, scaler state should be part of the checkpoint or saved alongside
    # The existing code saved it in 'kan_model.pth'.
    # I'll modify KANSolver to accept scaler or handle it, 
    # but for now, I'll just save it manually here to complete the loop.
    # Ideally, the `data_module` should handle saving its state.
    torch.save(scaler.state_dict(), f"{train_cfg.save_dir}/scaler_state.pth")
    print("Training Complete.")

if __name__ == "__main__":
    main()
