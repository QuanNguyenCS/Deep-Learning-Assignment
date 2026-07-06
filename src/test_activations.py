import os
import sys
import time
import torch
import torch.nn.functional as F
from torchvision.datasets import CelebA
from pathlib import Path

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40, activation='relu'):
        super().__init__()
        
        if activation == 'relu':
            act1 = torch.nn.ReLU()
            act2 = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            act1 = torch.nn.LeakyReLU(0.01)
            act2 = torch.nn.LeakyReLU(0.01)
        elif activation == 'elu':
            act1 = torch.nn.ELU()
            act2 = torch.nn.ELU()
        elif activation == 'swish':
            act1 = torch.nn.SiLU()
            act2 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
            
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim1),
            torch.nn.LayerNorm(hidden_dim1),
            act1,
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.LayerNorm(hidden_dim2),
            act2,
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
    print("=" * 80)
    print("Activation Function Comparison: ReLU vs Leaky ReLU vs ELU vs Swish")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_root = Path("Data")
    checkpoints_root = Path("checkpoints")
    act_checkpoints_root = checkpoints_root / "activations"
    act_checkpoints_root.mkdir(parents=True, exist_ok=True)

    train_embeddings_path = data_root / "celeba_train_embeddings.pt"
    val_embeddings_path = data_root / "celeba_val_embeddings.pt"
    test_embeddings_path = data_root / "celeba_test_embeddings.pt"

    # Verify embeddings exist
    for p in [train_embeddings_path, val_embeddings_path, test_embeddings_path]:
        if not p.exists():
            print(f"ERROR: Embeddings file not found: {p}")
            sys.exit(1)

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

    print(f"Dataset summary:")
    print(f"  Train: {E_train.shape[0]} samples")
    print(f"  Val:   {E_val.shape[0]} samples")
    print(f"  Test:  {E_test.shape[0]} samples")

    loss_configs = [
        {"name": "Standard BCE Loss", "gamma": 0.0, "alpha": 0.50},
        {"name": "Optimal Focal Loss", "gamma": 0.5, "alpha": 0.50}
    ]
    
    activations = ["relu", "leaky_relu", "elu", "swish"]
    
    num_samples = E_train.shape[0]
    batch_size = 4096
    num_epochs = 50
    
    results = []

    for loss_cfg in loss_configs:
        loss_name = loss_cfg["name"]
        gamma_val = loss_cfg["gamma"]
        alpha_val = loss_cfg["alpha"]
        
        print(f"\n--- Running experiments for {loss_name} (alpha={alpha_val}, gamma={gamma_val}) ---")
        
        # Setup alpha tensor
        alpha_tensor = torch.full((40,), alpha_val, device=device)
        
        for act in activations:
            print(f"Training activation: {act}...")
            start_time = time.time()
            
            # Instantiate model, loss, optimizer
            model = MLPClassifier(input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40, activation=act).to(device)
            criterion = ClassBalancedFocalLoss(alpha=alpha_tensor, gamma=gamma_val)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(num_epochs):
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
            
            train_time = time.time() - start_time
            
            # Load best weights
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            model.eval()
            
            # Evaluate test accuracy
            with torch.no_grad():
                test_preds = []
                for i in range(0, E_test.shape[0], batch_size):
                    test_preds.append(model(E_test[i:i+batch_size]))
                test_preds = torch.cat(test_preds, dim=0)
                
                # Threshold at 0.5
                predicted_binary = (test_preds > 0.5).float()
                correct = (predicted_binary == Y_test).float().sum().item()
                test_accuracy = correct / (E_test.shape[0] * 40.0)
            
            # Save checkpoint
            save_filename = f"{loss_name.lower().replace(' ', '_')}_{act}.pt"
            save_path = act_checkpoints_root / save_filename
            torch.save(best_model_state, save_path)
            
            print(f"  -> Best Val Loss: {best_val_loss:.4f} | Test Acc: {test_accuracy:.4%} | Time: {train_time:.1f}s")
            
            results.append({
                "loss_name": loss_name,
                "activation": act,
                "val_loss": best_val_loss,
                "test_acc": test_accuracy,
                "time": train_time,
                "checkpoint": str(save_path)
            })

    # Sort results by test accuracy within each loss type
    results.sort(key=lambda x: (x["loss_name"], -x["test_acc"]))
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("LEADERBOARD SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Loss Function':<25} | {'Activation':<15} | {'Best Val Loss':<15} | {'Test Accuracy':<15} | {'Train Time':<10}")
    print("-" * 90)
    for res in results:
        print(f"{res['loss_name']:<25} | {res['activation']:<15} | {res['val_loss']:<15.4f} | {res['test_acc']:<15.4%} | {res['time']:<10.1f}s")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
