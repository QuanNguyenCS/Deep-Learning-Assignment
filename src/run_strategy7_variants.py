import os
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
    print_evaluation_summary,
    evaluate_retrieval
)

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

def main():
    print("=" * 75)
    print("Query-Aware BCE Weighting Parameter Sweep (omega)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path("Data")
    checkpoints_root = Path("checkpoints")

    test_embeddings_path = data_root / "celeba_test_embeddings.pt"
    annotations_path = data_root / "celeba_evaluation.json"
    mlp_raw_path = checkpoints_root / "gde_train_mlp_classifier_raw.pt"

    # Verify embeddings exist
    if not test_embeddings_path.exists():
        print(f"ERROR: Test embeddings not found at {test_embeddings_path}")
        return

    # Verify model exists
    if not mlp_raw_path.exists():
        print(f"ERROR: Model not found at {mlp_raw_path}.")
        return

    # 1. Load CelebA test metadata & embeddings
    print("Loading CelebA test metadata...")
    celeba_test = CelebA(root=data_root, split="test", download=False)
    attr_names = [name for name in celeba_test.attr_names if name]
    name_to_idx = {name.lower().replace(' ', '_'): i for i, name in enumerate(attr_names)}

    print("Loading pre-computed test embeddings...")
    E_test = torch.load(test_embeddings_path, map_location=device)
    E_test = F.normalize(E_test, p=2, dim=-1)
    print(f"Test embeddings shape: {E_test.shape}")

    # 2. Load trained MLP (Raw CLIP BCE Backbone)
    print("Loading trained MLP classifier (Raw CLIP BCE)...")
    mlp_raw = MLPClassifier(input_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=40).to(device)
    mlp_raw.load_state_dict(torch.load(mlp_raw_path, map_location=device))
    mlp_raw.eval()

    # Pre-compute test candidate probabilities
    print("Pre-computing predictions on the test set...")
    with torch.no_grad():
        P_raw = []
        batch_size = 4096
        for i in range(0, E_test.shape[0], batch_size):
            P_raw.append(mlp_raw(E_test[i : i + batch_size]))
        P_raw = torch.cat(P_raw, dim=0)
        P_raw_clamped = torch.clamp(P_raw, min=1e-6, max=1.0 - 1e-6)

    # Loading test queries
    print("Loading test queries...")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    test_annotations = get_test_subset(annotations, num_queries=None, cases_per_query=500)

    # Generalized Strategy 7 Evaluator
    def run_eval(omega):
        """
        Runs global retrieval using weighted BCE loss.
        """
        def strategy_fn(source_idx, query_str, k):
            pos_attrs, neg_attrs = parse_query(query_str)
            pos_idx = [name_to_idx[a.strip().lower().replace(' ', '_')]
                       for a in pos_attrs if a.strip().lower().replace(' ', '_') in name_to_idx]
            neg_idx = [name_to_idx[a.strip().lower().replace(' ', '_')]
                       for a in neg_attrs if a.strip().lower().replace(' ', '_') in name_to_idx]

            # Get source predictions
            probs = P_raw_clamped[source_idx]

            # Construct target profile Y*
            Y_target = probs.clone()
            for a in pos_idx:
                Y_target[a] = 1.0
            for a in neg_idx:
                Y_target[a] = 0.0

            # Compute weighted BCE Loss globally
            y = Y_target.unsqueeze(0)  # [1, 40]
            
            # Setup weights W_a
            w = torch.ones(40, device=device)
            for a in pos_idx + neg_idx:
                w[a] = omega
            w = w.unsqueeze(0)  # [1, 40]

            term1 = y * torch.log(P_raw_clamped)
            term2 = (1.0 - y) * torch.log(1.0 - P_raw_clamped)
            
            # Weighted BCE loss per candidate
            weighted_bce = - torch.sum(w * (term1 + term2), dim=-1) / w.sum() # [19962]

            _, top_k = torch.topk(weighted_bce, k=k, largest=False)
            return top_k.tolist()

        # Run evaluation silently
        import io
        sys.stdout = io.StringIO()
        try:
            results = evaluate_all_queries(test_annotations, strategy_fn, k_values=[1, 5, 10])
        finally:
            sys.stdout = sys.__stdout__

        # Calculate averages
        r1 = sum(m["Recall@1"] for m in results.values()) / len(results)
        r5 = sum(m["Recall@5"] for m in results.values()) / len(results)
        r10 = sum(m["Recall@10"] for m in results.values()) / len(results)
        p1 = sum(m["Precision@1"] for m in results.values()) / len(results)
        p5 = sum(m["Precision@5"] for m in results.values()) / len(results)
        p10 = sum(m["Precision@10"] for m in results.values()) / len(results)

        return r1, r5, r10, p1, p5, p10

    def calc_f1(p, r):
        if p + r == 0:
            return 0.0
        return 2.0 * p * r / (p + r)

    # Query-Aware BCE Weighting Sweep
    omegas = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 15.0, 20.0]
    print(f"\nSweeping Query-Aware penalty parameter (omega) over {omegas}...")
    
    sweep_results = []
    for o_val in omegas:
        print(f"Evaluating omega = {o_val:.1f}...")
        r1, r5, r10, p1, p5, p10 = run_eval(omega=o_val)
        
        f1_1 = calc_f1(p1, r1)
        f1_5 = calc_f1(p5, r5)
        f1_10 = calc_f1(p10, r10)
        comp = (f1_1 + f1_5 + f1_10) / 3.0
        
        sweep_results.append({
            "omega": o_val,
            "metrics": (r1, r5, r10, p1, p5, p10),
            "composite": comp
        })
        print(f"  -> Recall@10: {r10:.4f} | Composite F1 Score: {comp:.4f}")
        
    # Print Leaderboard Table
    print("\n" + "=" * 115)
    print("QUERY-AWARE BCE WEIGHTING LEADERBOARD (sorted by Composite F1 Score)")
    print("=" * 115)
    print(f"{'Configuration':<50} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'P@1':>6} | {'P@5':>6} | {'P@10':>6} | {'Comp F1':>9}")
    print("-" * 115)
    
    # Sort descending by composite score
    sweep_results.sort(key=lambda x: x["composite"], reverse=True)
    for res in sweep_results:
        m = res['metrics']
        desc = f"Raw CLIP | Query-Aware BCE (omega={res['omega']:.1f})"
        if res['omega'] == 1.0:
            desc = "Raw CLIP | Baseline BCE (omega=1.0)"
        print(f"{desc:<50} | {m[0]:>6.4f} | {m[1]:>6.4f} | {m[2]:>6.4f} | {m[3]:>6.4f} | {m[4]:>6.4f} | {m[5]:>6.4f} | {res['composite']:>9.4f}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
