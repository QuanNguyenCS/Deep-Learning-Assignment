import os
import sys
import time
import torch
import torch.nn.functional as F
from torchvision.datasets import CelebA
from pathlib import Path

# Feature Attention Block (Squeeze-and-Excitation style)
class FeatureAttention(torch.nn.Module):
    def __init__(self, dim=256, reduction=8):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(dim, dim // reduction),
            torch.nn.ReLU(),
            torch.nn.Linear(dim // reduction, dim),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

# Attentional Head Module
class AttentionalHead(torch.nn.Module):
    def __init__(self, shared_dim=256, hidden_dim1=128, hidden_dim2=64, reduction=8):
        super().__init__()
        self.attention = FeatureAttention(dim=shared_dim, reduction=reduction)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(shared_dim, hidden_dim1),
            torch.nn.BatchNorm1d(hidden_dim1),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.BatchNorm1d(hidden_dim2),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_dim2, 1)
        )
    def forward(self, x):
        # Apply task-specific attention before head layers
        gated = self.attention(x)
        return self.net(gated)

# Multi-Task MLP with Task-Specific Attention
class MultiTaskAttentionMLP(torch.nn.Module):
    def __init__(self, input_dim=512, shared_dim=256):
        super().__init__()
        # Shared layers: 512 -> 512 -> 256
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(512, shared_dim),
            torch.nn.BatchNorm1d(shared_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2)
        )
        
        # 40 separate attentional heads
        self.heads = torch.nn.ModuleList([
            AttentionalHead(shared_dim=shared_dim) for _ in range(40)
        ])
        
    def forward(self, x):
        shared_feats = self.shared(x)
        outputs = [head(shared_feats) for head in self.heads]
        outputs = torch.cat(outputs, dim=-1)
        return torch.sigmoid(outputs)

# Baseline Multi-Task MLP (no attention)
class DeepMultiTaskMLP(torch.nn.Module):
    def __init__(self, input_dim=512, shared_dim=256):
        super().__init__()
        # Shared layers: 512 -> 512 -> 256
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(512, shared_dim),
            torch.nn.BatchNorm1d(shared_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2)
        )
        
        # 40 separate standard heads
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(shared_dim, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(128, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(64, 1)
            ) for _ in range(40)
        ])
        
    def forward(self, x):
        shared_feats = self.shared(x)
        outputs = [head(shared_feats) for head in self.heads]
        outputs = torch.cat(outputs, dim=-1)
        return torch.sigmoid(outputs)

def main():
    print("=" * 80)
    print("Task-Specific Attention Evaluation: Multi-Task Baseline vs Attention Multi-Task")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_root = Path("Data")
    checkpoints_root = Path("checkpoints")
    attn_multitask_checkpoints_root = checkpoints_root / "multitask_attention"
    attn_multitask_checkpoints_root.mkdir(parents=True, exist_ok=True)

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
    print(f"  Test:  {E_test.shape[0]} samples\n")

    # Standard unweighted BCE loss configuration
    criterion = torch.nn.BCELoss()

    # Models to compare
    models_to_test = [
        {"name": "Multi-Task Baseline (No Attention)", "class": DeepMultiTaskMLP},
        {"name": "Multi-Task Attention (Task-Specific)", "class": MultiTaskAttentionMLP}
    ]
    
    num_samples = E_train.shape[0]
    batch_size = 4096
    num_epochs = 50
    results = []

    for model_cfg in models_to_test:
        model_name = model_cfg["name"]
        model_class = model_cfg["class"]
        
        print(f"Training {model_name}...")
        start_time = time.time()
        
        model = model_class().to(device)
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
                # Clamp to avoid numerical instability
                outputs = torch.clamp(outputs, min=1e-6, max=1.0 - 1e-6)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.shape[0]
                
            model.eval()
            with torch.no_grad():
                val_outputs = model(E_val)
                val_outputs = torch.clamp(val_outputs, min=1e-6, max=1.0 - 1e-6)
                val_loss = criterion(val_outputs, Y_val).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        train_time = time.time() - start_time
        
        # Load best weights for test evaluation
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
        save_name = f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pt"
        save_path = attn_multitask_checkpoints_root / save_name
        torch.save(best_model_state, save_path)
        
        print(f"  -> Best Val Loss: {best_val_loss:.4f} | Test Acc: {test_accuracy:.4%} | Time: {train_time:.1f}s")
        
        results.append({
            "name": model_name,
            "val_loss": best_val_loss,
            "test_acc": test_accuracy,
            "time": train_time
        })

    # Sort results by test accuracy
    results.sort(key=lambda x: -x["test_acc"])
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("TASK-SPECIFIC ATTENTION LEADERBOARD (sorted by Test Acc)")
    print("=" * 90)
    print(f"{'Model Configuration':<40} | {'Best Val Loss':<15} | {'Test Accuracy':<15} | {'Train Time':<10}")
    print("-" * 90)
    for res in results:
        print(f"{res['name']:<40} | {res['val_loss']:<15.4f} | {res['test_acc']:<15.4%} | {res['time']:<10.1f}s")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
