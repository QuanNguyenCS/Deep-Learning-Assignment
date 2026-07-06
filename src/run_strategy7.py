import os
import re
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from pathlib import Path
from tqdm import tqdm

from utils import (
    parse_query,
    get_test_subset,
    evaluate_all_queries,
    visualize_all_results_grid,
    print_evaluation_summary,
    evaluate_retrieval
)

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

def main():
    print("=" * 75)
    print("Strategy 7: Global Weighted BCE Loss Retrieval (omega=4.0)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path("Data")
    checkpoints_root = Path("checkpoints")
    output_root = Path("output")

    # Create output root if not exists
    output_root.mkdir(exist_ok=True)

    test_embeddings_path = data_root / "celeba_test_embeddings.pt"
    mlp_classifier_path = checkpoints_root / "gde_train_mlp_classifier_raw.pt"
    annotations_path = data_root / "celeba_evaluation.json"

    # Verify caches exist
    for path in [test_embeddings_path, mlp_classifier_path, annotations_path]:
        if not path.exists():
            print(f"ERROR: Required file not found: {path}")
            sys.exit(1)

    # 1. Load data
    print("Loading CelebA test metadata...")
    celeba_test = CelebA(root=data_root, split="test", download=False)
    attr_names = [name for name in celeba_test.attr_names if name]
    name_to_idx = {name.lower().replace(' ', '_'): i for i, name in enumerate(attr_names)}

    print("Loading pre-computed test embeddings...")
    E_test = torch.load(test_embeddings_path, map_location=device)
    E_test = F.normalize(E_test, p=2, dim=-1)
    print(f"Test embeddings shape: {E_test.shape}")

    # Load trained MLP
    print("Loading trained MLP classifier (Raw CLIP)...")
    mlp_model = MLPClassifier(input_dim=E_test.shape[1], hidden_dim1=256, hidden_dim2=128, output_dim=40).to(device)
    mlp_model.load_state_dict(torch.load(mlp_classifier_path, map_location=device))
    mlp_model.eval()

    # Pre-compute probabilities of ALL candidates globally
    print("Pre-computing attribute probabilities for all 19,962 test candidates (Raw CLIP)...")
    with torch.no_grad():
        P_candidates = []
        batch_size = 4096
        for i in range(0, E_test.shape[0], batch_size):
            batch_x = E_test[i : i + batch_size]
            probs_batch = mlp_model(batch_x)
            P_candidates.append(probs_batch)
        P_candidates = torch.cat(P_candidates, dim=0)
    
    # Clamp candidate probabilities to avoid log(0) and log(1)
    P_candidates_clamped = torch.clamp(P_candidates, min=1e-6, max=1.0 - 1e-6)
    print(f"Candidate probabilities matrix cached: {P_candidates_clamped.shape}")

    # Load annotations
    print("Loading evaluation annotations...")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # =========================================================
    # Strategy 7: Global Weighted BCE Loss Minimization (omega=4.0)
    # =========================================================
    def strategy7_global_bce(source_idx, query_str, k):
        # 1. Parse positive and negative modification attributes
        pos_attrs, neg_attrs = parse_query(query_str)
        pos_idx = [name_to_idx[a.strip().lower().replace(' ', '_')]
                   for a in pos_attrs if a.strip().lower().replace(' ', '_') in name_to_idx]
        neg_idx = [name_to_idx[a.strip().lower().replace(' ', '_')]
                   for a in neg_attrs if a.strip().lower().replace(' ', '_') in name_to_idx]

        # 2. Get baseline probabilities of source image
        probs = P_candidates_clamped[source_idx]  # shape [40]

        # 3. Build continuous target profile Y*
        Y_target = probs.clone()
        for a in pos_idx:
            Y_target[a] = 1.0
        for a in neg_idx:
            Y_target[a] = 0.0

        # 4. Compute Weighted Binary Cross Entropy globally against all candidates
        y = Y_target.unsqueeze(0)  # shape [1, 40]
        
        # Setup weights W_a (omega=4.0 for query attributes, 1.0 otherwise)
        w = torch.ones(40, device=device)
        for a in pos_idx + neg_idx:
            w[a] = 4.0
        w = w.unsqueeze(0)  # shape [1, 40]

        term1 = y * torch.log(P_candidates_clamped)
        term2 = (1.0 - y) * torch.log(1.0 - P_candidates_clamped)
        
        # Weighted BCE loss mean
        weighted_bce = - torch.sum(w * (term1 + term2), dim=-1) / w.sum()  # shape [19962]

        # 5. Retrieve top K candidates with the SMALLEST BCE loss
        _, top_k = torch.topk(weighted_bce, k=k, largest=False)

        return top_k.tolist()

    # 4. Evaluate Strategy 7
    test_annotations = get_test_subset(annotations, num_queries=None, cases_per_query=500)
    print(f"Filtered to full test subset: {len(test_annotations)} queries, 500 cases each.")

    k_values = [1, 5, 10]

    print(f"\n{'='*60}")
    print("EVALUATING STRATEGY 7: Global Weighted BCE Loss Retrieval (omega=4.0)")
    print(f"{'='*60}")
    results = evaluate_all_queries(test_annotations, strategy7_global_bce, k_values=k_values)
    print_evaluation_summary(results, k_values, title="STRATEGY 7 EVALUATION — Global Weighted BCE Loss Retrieval (omega=4.0)")

    # 5. Visual Grid
    print(f"\nGenerating grid visualization for Strategy 7...")
    vis_annotations = get_test_subset(annotations, num_queries=None, cases_per_query=2)
    visualize_all_results_grid(vis_annotations, strategy7_global_bce, celeba_test,
                               save_path=output_root / "gde_strategy7_results.png")

    print("\n" + "=" * 60)
    print("STRATEGY 7 EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
