import torch
from data.generate import generate_dataset
from train import train
from evaluate import run_evaluation
import os
import matplotlib
# Use non-interactive backend for matplotlib in CI
matplotlib.use('Agg')

def run_ci_pipeline():
    print("=== CI Pipeline: Dataset Generation ===")
    # Small dataset for CI
    generate_dataset(num_samples=50, N_coarse=64, N_fine=256, steps_ahead=5)
    
    print("=== CI Pipeline: Training ===")
    # Small training for CI
    train(data_path='kan_train_data.npz', epochs=2, batch_size=20, device=torch.device('cpu'))
    
    print("=== CI Pipeline: Evaluation ===")
    # Evaluation with the trained model
    if os.path.exists('kan_model.pth'):
        run_evaluation(model_path='kan_model.pth')
    else:
        print("Error: kan_model.pth not found!")
        exit(1)

if __name__ == "__main__":
    run_ci_pipeline()
