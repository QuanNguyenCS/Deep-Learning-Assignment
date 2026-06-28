import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.datasets import CelebA

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
        self.alpha = alpha  # Shape [40] class weights
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

def train_focal_loss_model(device, data_root, mlp_focal_path, use_balanced_alpha=True):
    print("\n" + "=" * 80)
    loss_type = "Class-Balanced Focal Loss" if use_balanced_alpha else "Standard Focal Loss (alpha=0.25)"
    print(f"Training new MLP model with {loss_type}...")
    print("=" * 80)
    
    train_embeddings_path = data_root / "celeba_train_embeddings.pt"
    val_embeddings_path = data_root / "celeba_val_embeddings.pt"
    
    if not train_embeddings_path.exists() or not val_embeddings_path.exists():
        print("ERROR: Train or Val embeddings not found!")
        sys.exit(1)
        
    print("Loading pre-computed train embeddings...")
    E_train = torch.load(train_embeddings_path, map_location=device)
    E_train = F.normalize(E_train, p=2, dim=-1)
    
    print("Loading pre-computed val embeddings...")
    E_val = torch.load(val_embeddings_path, map_location=device)
    E_val = F.normalize(E_val, p=2, dim=-1)
    
    print("Loading CelebA train dataset metadata...")
    celeba_train = CelebA(root=data_root, split="train", download=False)
    Y_train = celeba_train.attr[:E_train.shape[0]].to(device).float()
    
    print("Loading CelebA val dataset metadata...")
    celeba_val = CelebA(root=data_root, split="valid", download=False)
    Y_val = celeba_val.attr[:E_val.shape[0]].to(device).float()
    
    # Define alpha
    if use_balanced_alpha:
        frequencies = Y_train.mean(dim=0)
        alpha = 1.0 - frequencies
        print(f"Using class-balanced alpha. Rare class examples: Bald={frequencies[4].item():.4f}, Mustache={frequencies[22].item():.4f}")
    else:
        alpha = torch.full((40,), 0.25, device=device)
        print("Using standard constant alpha = 0.25 across all 40 attributes.")
        
    # Instantiate model and loss
    model = MLPClassifier(input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40).to(device)
    criterion = ClassBalancedFocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    num_epochs = 50
    batch_size = 4096
    num_samples = E_train.shape[0]
    best_val_loss = float('inf')
    
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
            
        train_loss = epoch_loss / num_samples
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), mlp_focal_path)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")
            
    print(f"MLP training completed. Best Val Loss: {best_val_loss:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Raw CLIP Attribute Classification Debug Tool (3-Way)")
    parser.add_argument("--indices", "-i", type=int, nargs="+", default=[13], help="List of image indices in CelebA test set to debug (default: [13])")
    args = parser.parse_args()

    print("=" * 118)
    print("ATTRIBUTE EXTRACTION COMPARISON: STANDARD BCE VS CLASS-BALANCED FOCAL LOSS VS OPTIMAL FOCAL LOSS (alpha=0.50, gamma=0.5)")
    print("=" * 118)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path("Data")
    checkpoints_root = Path("checkpoints")
    checkpoints_root.mkdir(exist_ok=True)

    test_embeddings_path = data_root / "celeba_test_embeddings.pt"
    mlp_std_path = checkpoints_root / "gde_train_mlp_classifier_raw.pt"
    mlp_focal_path = checkpoints_root / "gde_train_mlp_classifier_focal_loss.pt"
    mlp_focal_opt_path = checkpoints_root / "gde_train_mlp_classifier_focal_loss_best.pt"

    # Verify embeddings exist
    if not test_embeddings_path.exists():
        print(f"ERROR: Test embeddings not found at {test_embeddings_path}")
        return

    # Verify standard model exists
    if not mlp_std_path.exists():
        print(f"ERROR: Standard model not found at {mlp_std_path}.")
        return

    # Train class-balanced focal model if missing
    if not mlp_focal_path.exists():
        train_focal_loss_model(device, data_root, mlp_focal_path, use_balanced_alpha=True)

    # Verify optimal focal model exists
    if not mlp_focal_opt_path.exists():
        print(f"ERROR: Optimal Focal Loss model not found at {mlp_focal_opt_path}. Please run run_focal_grid_search.py first.")
        return

    # Load Model 1: Standard BCE
    print("Loading Standard BCE Model...")
    model_std = MLPClassifier(input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40).to(device)
    model_std.load_state_dict(torch.load(mlp_std_path, map_location=device))
    model_std.eval()

    # Load Model 2: Class-Balanced Focal Loss
    print("Loading Class-Balanced Focal Loss Model...")
    model_focal_cb = MLPClassifier(input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40).to(device)
    model_focal_cb.load_state_dict(torch.load(mlp_focal_path, map_location=device))
    model_focal_cb.eval()

    # Load Model 3: Optimal Focal Loss (alpha=0.50, gamma=0.5)
    print("Loading Optimal Focal Loss Model (alpha=0.50, gamma=0.5)...")
    model_focal_opt = MLPClassifier(input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40).to(device)
    model_focal_opt.load_state_dict(torch.load(mlp_focal_opt_path, map_location=device))
    model_focal_opt.eval()

    # Load test embeddings
    print("Loading pre-computed test embeddings...")
    E_test = torch.load(test_embeddings_path, map_location=device)
    E_test = F.normalize(E_test, p=2, dim=-1)

    # Load test dataset metadata
    print("Loading CelebA test metadata...")
    celeba_test = CelebA(root=data_root, split="test", download=False)
    attr_names = [name for name in celeba_test.attr_names if name]
    test_attrs = celeba_test.attr

    total_images = len(args.indices)
    std_correct_total = 0
    focal_cb_correct_total = 0
    focal_opt_correct_total = 0

    for idx in args.indices:
        print("\n" + "-" * 118)
        print(f"IMAGE INDEX: {idx}")
        print("-" * 118)
        
        # Get target CLIP feature
        x = E_test[idx].unsqueeze(0).to(device)
        
        # Get ground truth attributes
        gt = test_attrs[idx].to(device)
        
        # Model predictions
        with torch.no_grad():
            pred_std = model_std(x).squeeze(0)
            pred_focal_cb = model_focal_cb(x).squeeze(0)
            pred_focal_opt = model_focal_opt(x).squeeze(0)
            
        print(f"{'Attribute':<25} | {'GT':<5} | {'Model 1 (BCE)':<18} | {'Model 2 (CB Focal)':<20} | {'Model 3 (Opt Focal)':<20}")
        print("-" * 118)
        
        std_correct = 0
        focal_cb_correct = 0
        focal_opt_correct = 0
        
        for a in range(40):
            attr_name = attr_names[a]
            gt_val = int(gt[a].item())
            
            p_std = pred_std[a].item()
            p_focal_cb = pred_focal_cb[a].item()
            p_focal_opt = pred_focal_opt[a].item()
            
            bin_std = 1 if p_std > 0.5 else 0
            bin_focal_cb = 1 if p_focal_cb > 0.5 else 0
            bin_focal_opt = 1 if p_focal_opt > 0.5 else 0
            
            if bin_std == gt_val: std_correct += 1
            if bin_focal_cb == gt_val: focal_cb_correct += 1
            if bin_focal_opt == gt_val: focal_opt_correct += 1
            
            gt_str = "YES" if gt_val == 1 else "NO"
            pred_std_str = f"{p_std*100:5.1f}% ({'YES' if bin_std==1 else 'NO'})"
            pred_focal_cb_str = f"{p_focal_cb*100:5.1f}% ({'YES' if bin_focal_cb==1 else 'NO'})"
            pred_focal_opt_str = f"{p_focal_opt*100:5.1f}% ({'YES' if bin_focal_opt==1 else 'NO'})"
            
            # Highlight differences or correct predictions
            print(f"{attr_name:<25} | {gt_str:<5} | {pred_std_str:<18} | {pred_focal_cb_str:<20} | {pred_focal_opt_str:<20}")
            
        std_correct_total += std_correct
        focal_cb_correct_total += focal_cb_correct
        focal_opt_correct_total += focal_opt_correct
        
        print("-" * 118)
        print(f"Accuracy for Image {idx:<6} | BCE: {std_correct}/40 ({std_correct/40.0:.1%}) | CB Focal: {focal_cb_correct}/40 ({focal_cb_correct/40.0:.1%}) | Opt Focal: {focal_opt_correct}/40 ({focal_opt_correct/40.0:.1%})")

    print("\n" + "=" * 118)
    print(f"OVERALL SUMMARY ({total_images} images, {total_images*40} total attributes):")
    print(f"  - Model 1 (Standard BCE) Accuracy            : {std_correct_total}/{total_images*40} ({std_correct_total/(total_images*40.0):.2%})")
    print(f"  - Model 2 (Class-Balanced Focal Loss) Accuracy: {focal_cb_correct_total}/{total_images*40} ({focal_cb_correct_total/(total_images*40.0):.2%})")
    print(f"  - Model 3 (Optimal Focal Loss: 0.50, 0.5) Accuracy: {focal_opt_correct_total}/{total_images*40} ({focal_opt_correct_total/(total_images*40.0):.2%})")
    print("=" * 118 + "\n")

if __name__ == "__main__":
    main()
