import os
import sys
import time
import torch
import torch.nn.functional as F
from torchvision.datasets import CelebA
from pathlib import Path

# Baseline 3-Layer Single-Head MLP (matching run_strategy7.py)
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim1),
            torch.nn.LayerNorm(hidden_dim1),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.LayerNorm(hidden_dim2),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_dim2, output_dim),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def calculate_metrics(preds, targets):
    tp = ((preds == 1) & (targets == 1)).float().sum().item()
    fp = ((preds == 1) & (targets == 0)).float().sum().item()
    fn = ((preds == 0) & (targets == 1)).float().sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def main():
    print("=" * 80)
    print("CelebA Attribute-Specific Accuracy & F1-Score Analysis")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_root = Path("Data")
    checkpoints_root = Path("checkpoints")
    checkpoints_root.mkdir(exist_ok=True)
    save_path = checkpoints_root / "attribute_analysis_mlp.pt"

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
    
    attr_names = [name for name in celeba_test.attr_names if name]

    print(f"Dataset summary:")
    print(f"  Train: {E_train.shape[0]} samples")
    print(f"  Val:   {E_val.shape[0]} samples")
    print(f"  Test:  {E_test.shape[0]} samples\n")

    # Train model
    print("Training MLPClassifier for 50 epochs...")
    model = MLPClassifier().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    num_samples = E_train.shape[0]
    batch_size = 4096
    num_epochs = 50
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

    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}")
    torch.save(best_model_state, save_path)
    
    # Load best weights
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    
    # Evaluate test split
    print("Evaluating model predictions on test split...")
    with torch.no_grad():
        test_preds = []
        for i in range(0, E_test.shape[0], batch_size):
            test_preds.append(model(E_test[i:i+batch_size]))
        test_preds = torch.cat(test_preds, dim=0)
        
    predicted_binary = (test_preds > 0.5).float()
    
    # Calculate metrics for each attribute
    attribute_metrics = []
    
    for a in range(40):
        attr_name = attr_names[a]
        
        pred_a = predicted_binary[:, a]
        gt_a = Y_test[:, a]
        
        # Accuracy
        correct = (pred_a == gt_a).float().sum().item()
        accuracy = correct / len(gt_a)
        
        # Precision, Recall, F1
        precision, recall, f1 = calculate_metrics(pred_a, gt_a)
        
        # Pos Ratio
        pos_count = int(gt_a.sum().item())
        pos_ratio = pos_count / len(gt_a)
        
        attribute_metrics.append({
            "name": attr_name,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "pos_ratio": pos_ratio
        })
        
    # Sort strictly by F1-Score descending
    attribute_metrics.sort(key=lambda x: -x["f1"])
    
    # Calculate and print global metrics
    global_correct = (predicted_binary == Y_test).float().sum().item()
    global_accuracy = global_correct / (E_test.shape[0] * 40.0)
    print(f"\nGlobal Average Test Accuracy: {global_accuracy:.4%}\n")
    
    # Print leaderboard table
    print("=" * 115)
    print("ATTRIBUTE LEADERBOARD (Sorted strictly by F1-Score Descending)")
    print("=" * 115)
    print(f"{'Index':<5} | {'Attribute Name':<25} | {'Accuracy':<12} | {'F1-Score':<10} | {'Precision':<12} | {'Recall':<12} | {'Pos Ratio':<10}")
    print("-" * 115)
    for idx, metric in enumerate(attribute_metrics):
        print(f"{idx+1:<5} | {metric['name']:<25} | {metric['accuracy']:<12.4%} | {metric['f1']:<10.4f} | {metric['precision']:<12.2%} | {metric['recall']:<12.2%} | {metric['pos_ratio']:<10.2%}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
