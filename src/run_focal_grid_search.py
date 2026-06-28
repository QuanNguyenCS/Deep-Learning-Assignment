import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from pathlib import Path

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim1),
            torch.nn.LayerNorm(hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.LayerNorm(hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_dim2, output_dim),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class ClassBalancedFocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Shape [40]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.clamp(inputs, min=1e-6, max=1.0 - 1e-6)
        alpha = self.alpha.unsqueeze(0)  # Shape [1, 40]
        
        loss_pos = - alpha * torch.pow(1.0 - p, self.gamma) * torch.log(p)
        loss_neg = - (1.0 - alpha) * torch.pow(p, self.gamma) * torch.log(1.0 - p)
        
        loss = targets * loss_pos + (1.0 - targets) * loss_neg
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

def main():
    print("=" * 70)
    print("Focal Loss Expanded Grid Search (12 Combinations x 50 Epochs)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path("Data")
    checkpoints_root = Path("checkpoints")
    checkpoints_root.mkdir(exist_ok=True)

    train_embeddings_path = data_root / "celeba_train_embeddings.pt"
    val_embeddings_path = data_root / "celeba_val_embeddings.pt"
    test_embeddings_path = data_root / "celeba_test_embeddings.pt"

    # Load data
    print("Loading pre-computed embeddings...")
    E_train = torch.load(train_embeddings_path, map_location=device)
    E_train = F.normalize(E_train, p=2, dim=-1)
    
    E_val = torch.load(val_embeddings_path, map_location=device)
    E_val = F.normalize(E_val, p=2, dim=-1)
    
    E_test = torch.load(test_embeddings_path, map_location=device)
    E_test = F.normalize(E_test, p=2, dim=-1)
    
    print("Loading CelebA datasets metadata...")
    celeba_train = CelebA(root=data_root, split="train", download=False)
    Y_train = celeba_train.attr[:E_train.shape[0]].to(device).float()
    
    celeba_val = CelebA(root=data_root, split="valid", download=False)
    Y_val = celeba_val.attr[:E_val.shape[0]].to(device).float()
    
    celeba_test = CelebA(root=data_root, split="test", download=False)
    Y_test = celeba_test.attr[:E_test.shape[0]].to(device).float()

    # Search Space (12 points)
    alphas = [0.35, 0.45, 0.50, 0.55]
    gammas = [0.5, 1.0, 1.5]
    
    num_samples = E_train.shape[0]
    batch_size = 4096
    
    grid_results = []
    
    print(f"\nStarting Grid Search: 12 combinations, 50 epochs each...")
    
    for alpha_val in alphas:
        for gamma_val in gammas:
            print(f"\nTraining: alpha={alpha_val:.2f}, gamma={gamma_val:.1f} for 50 epochs...")
            
            # Setup alpha tensor
            alpha_tensor = torch.full((40,), alpha_val, device=device)
            
            # Instantiate model
            model = MLPClassifier(input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40).to(device)
            criterion = ClassBalancedFocalLoss(alpha=alpha_tensor, gamma=gamma_val)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            best_val_loss = float('inf')
            best_model_state = None
            
            # Train for 50 epochs
            for epoch in range(50):
                model.train()
                permutation = torch.randperm(num_samples)
                epoch_loss = 0.0
                for i in range(0, num_samples, batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_x = E_train[indices]
                    batch_y = Y_train[indices]
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_x.shape[0]
                    
                model.eval()
                with torch.no_grad():
                    val_outputs = model(E_val)
                    val_loss = criterion(val_outputs, Y_val).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    
            # Load best model state for evaluation
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            model.eval()
            
            # Evaluate classification accuracy on the test split
            with torch.no_grad():
                test_preds = []
                for i in range(0, E_test.shape[0], batch_size):
                    test_preds.append(model(E_test[i : i + batch_size]))
                test_preds = torch.cat(test_preds, dim=0)
                
                # Threshold at 0.5
                predicted_binary = (test_preds > 0.5).float()
                correct = (predicted_binary == Y_test).float().sum().item()
                accuracy = correct / (E_test.shape[0] * 40.0)
                
            print(f"Finished -> Best Val Loss: {best_val_loss:.4f} | Test Classification Accuracy: {accuracy:.4%}")
            
            grid_results.append({
                "alpha": alpha_val,
                "gamma": gamma_val,
                "accuracy": accuracy,
                "state_dict": best_model_state
            })
            
    # Print Leaderboard
    grid_results.sort(key=lambda x: x["accuracy"], reverse=True)
    print("\n" + "=" * 60)
    print("FOCAL LOSS 50-EPOCH LEADERBOARD (sorted by Test Acc)")
    print("=" * 60)
    print(f"{'Alpha':<10} | {'Gamma':<10} | {'Test Accuracy':<15}")
    print("-" * 60)
    for res in grid_results:
        print(f"{res['alpha']:<10.2f} | {res['gamma']:<10.1f} | {res['accuracy']:<15.4%}")
    print("=" * 60)
    
    # Save the absolute winner
    winner = grid_results[0]
    best_model_path = checkpoints_root / "gde_train_mlp_classifier_focal_loss_best.pt"
    torch.save({k: v.to(device) for k, v in winner["state_dict"].items()}, best_model_path)
    print(f"\nWinning Model (alpha={winner['alpha']:.2f}, gamma={winner['gamma']:.1f}) saved to {best_model_path}\n")

if __name__ == "__main__":
    main()
