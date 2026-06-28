import os
import re
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_query(query_str):
    """
    Parse query attributes from string separated by comma or &.
    """
    pos_attrs = []
    neg_attrs = []
    parts = re.split(r'[&,]', query_str)
    for part in parts:
        part = part.strip()
        if part.startswith('+'):
            pos_attrs.append(part[1:].strip())
        elif part.startswith('-'):
            neg_attrs.append(part[1:].strip())
    return pos_attrs, neg_attrs

def compose_prompt(query_str):
    """
    Combine positive and negative attributes into a single natural language description.
    """
    pos_attrs, neg_attrs = parse_query(query_str)
    parts = []
    if pos_attrs:
        pos_part = ", ".join(attr.lower().replace("_", " ") for attr in pos_attrs)
        parts.append(pos_part)
    if neg_attrs:
        neg_part = ", ".join(f"not {attr.lower().replace('_', ' ')}" for attr in neg_attrs)
        parts.append(neg_part)
    combined = ", ".join(parts)
    return f"a photo of a person with {combined}"

def get_text_embedding(prompt, model, processor, device):
    """
    Encode prompt text using CLIP and return normalized L2 embedding.
    """
    inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = model.get_text_features(**inputs)
        text_features = F.normalize(text_outputs, p=2, dim=-1)
    return text_features

def retrieve_top_k_cosine(query_embedding, all_embeddings, k=5):
    """
    Calculate cosine similarity and return top K indices.
    """
    query_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=-1)
    similarities = (query_embedding_norm @ all_embeddings_norm.T).squeeze(0)
    top_k_values, top_k_indices = torch.topk(similarities, k=k)
    return top_k_indices.tolist()

def evaluate_retrieval(retrieved_indices, ground_truth_indices, k):
    """
    Calculate Recall@K and Precision@K.
    """
    top_k_retrieved = retrieved_indices[:k]
    hits = set(top_k_retrieved).intersection(set(ground_truth_indices))
    num_hits = len(hits)
    recall_at_k = 1 if num_hits > 0 else 0
    precision_at_k = num_hits / k
    return {
        f"Recall@{k}": recall_at_k,
        f"Precision@{k}": precision_at_k
    }

def get_test_subset(annotations, num_queries=5, cases_per_query=2):
    """
    Returns a copy of the annotations subset with a limited number of queries
    and cases per query.
    """
    import copy
    subset = []
    limit_queries = num_queries if (num_queries is not None and num_queries > 0) else len(annotations)
    for q_data in annotations[:limit_queries]:
        q_copy = copy.deepcopy(q_data)
        if cases_per_query is not None and cases_per_query > 0:
            gt_items = list(q_copy["ground_truth"].items())[:cases_per_query]
            q_copy["ground_truth"] = dict(gt_items)
        subset.append(q_copy)
    return subset

def get_tuning_subset(annotations, cases_per_query=2):
    """
    Returns a copy of the annotations subset with all queries,
    but only keeping the first `cases_per_query` cases for each.
    """
    import copy
    subset = []
    for q_data in annotations:
        q_copy = copy.deepcopy(q_data)
        if cases_per_query is not None and cases_per_query > 0:
            gt_items = list(q_copy["ground_truth"].items())[:cases_per_query]
            q_copy["ground_truth"] = dict(gt_items)
        subset.append(q_copy)
    return subset

def evaluate_all_queries(annotations, retrieval_fn, k_values=[1, 5, 10]):
    """
    Evaluate all queries in the benchmark JSON using a callback retrieval function.
    """
    all_results = {}

    for q_data in annotations:
        query_str = q_data["query"]
        ground_truth_dict = q_data["ground_truth"]

        print(f"\nEvaluating query: '{query_str}' for {len(ground_truth_dict)} source images...")
        query_results = []

        for source_img_id, gt_indices in tqdm(ground_truth_dict.items(), desc=f"Query: '{query_str}'"):
            source_idx = int(source_img_id)
            max_k = max(k_values)

            # Get predictions from retrieval callback function
            predictions = retrieval_fn(source_idx, query_str, max_k)

            metrics = {}
            for k in k_values:
                res = evaluate_retrieval(predictions, gt_indices, k=k)
                metrics[f"Recall@{k}"] = res[f"Recall@{k}"]
                metrics[f"Precision@{k}"] = res[f"Precision@{k}"]
            metrics["gt_count"] = len(gt_indices)
            query_results.append(metrics)

        # Average metrics
        avg_metrics = {}
        for k in k_values:
            valid_results = [res for res in query_results if res["gt_count"] >= k]
            if len(valid_results) > 0:
                avg_metrics[f"Recall@{k}"] = sum(res[f"Recall@{k}"] for res in valid_results) / len(valid_results)
                avg_metrics[f"Precision@{k}"] = sum(res[f"Precision@{k}"] for res in valid_results) / len(valid_results)
            else:
                avg_metrics[f"Recall@{k}"] = 0.0
                avg_metrics[f"Precision@{k}"] = 0.0

        all_results[query_str] = avg_metrics
        print_parts = [f"Recall@{k}: {avg_metrics[f'Recall@{k}']:.4f}" for k in k_values]
        print(f"Result -> " + " | ".join(print_parts))

    return all_results

def visualize_all_results_grid(test_subset, retrieval_fn, dataset, save_path="all_retrieval_results.png"):
    """
    Retrieve top 10 for all test cases and plot them in a single figure grid alongside their ground truth targets.
    """
    all_cases = []
    for q_data in test_subset:
        query_str = q_data["query"]
        for source_img_id, gt_indices in q_data["ground_truth"].items():
            all_cases.append({
                "query": query_str,
                "source_idx": int(source_img_id),
                "gt_indices": gt_indices
            })
    
    num_cases = len(all_cases)
    if num_cases == 0:
        print("No cases to visualize.")
        return
        
    num_rows = 2 * num_cases
    num_cols = 11  # 1 source/text label + 10 images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 2.2 * num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
        
    for case_idx, case in enumerate(all_cases):
        source_idx = case["source_idx"]
        query_str = case["query"]
        gt_indices = case["gt_indices"]
        
        # Get top 10 predictions from retrieval callback function
        retrieved_indices = retrieval_fn(source_idx, query_str, 10)
        
        row_ret = 2 * case_idx
        row_gt = 2 * case_idx + 1
        
        # --- Row 1: Source + Retrieved ---
        source_img = dataset[source_idx][0]
        axes[row_ret, 0].imshow(source_img)
        axes[row_ret, 0].set_title(f"Src: {source_idx}\nQ: {query_str}", fontsize=8, color='blue', fontweight='bold')
        axes[row_ret, 0].set_xticks([])
        axes[row_ret, 0].set_yticks([])
        for spine in axes[row_ret, 0].spines.values():
            spine.set_color('blue')
            spine.set_linewidth(1.5)
            
        for col_idx in range(1, 11):
            ret_idx_in_list = col_idx - 1
            if ret_idx_in_list < len(retrieved_indices):
                idx = retrieved_indices[ret_idx_in_list]
                img = dataset[idx][0]
                axes[row_ret, col_idx].imshow(img)
                
                is_correct = idx in gt_indices
                border_color = 'green' if is_correct else 'red'
                
                axes[row_ret, col_idx].set_title(f"Rank {col_idx}\nID: {idx}", fontsize=7, color=border_color)
                axes[row_ret, col_idx].set_xticks([])
                axes[row_ret, col_idx].set_yticks([])
                for spine in axes[row_ret, col_idx].spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(2)
            else:
                axes[row_ret, col_idx].axis('off')
                
        # --- Row 2: Ground Truth label + Ground Truth images ---
        axes[row_gt, 0].text(0.5, 0.5, f"Ground Truth\nTargets\n(Src {source_idx})", ha='center', va='center', fontsize=9, color='green', fontweight='bold')
        axes[row_gt, 0].axis('off')
        
        for col_idx in range(1, 11):
            gt_idx_in_list = col_idx - 1
            if gt_idx_in_list < len(gt_indices):
                idx = gt_indices[gt_idx_in_list]
                img = dataset[idx][0]
                axes[row_gt, col_idx].imshow(img)
                axes[row_gt, col_idx].set_title(f"GT {col_idx}\nID: {idx}", fontsize=7, color='green')
                axes[row_gt, col_idx].set_xticks([])
                axes[row_gt, col_idx].set_yticks([])
                for spine in axes[row_gt, col_idx].spines.values():
                    spine.set_color('green')
                    spine.set_linewidth(1.5)
            else:
                axes[row_gt, col_idx].axis('off')
                
    plt.suptitle("Compositional Image Retrieval Results (Top 10 vs Ground Truth)", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nGrid visualization saved to {save_path}")

def print_evaluation_summary(results, k_values, title="EVALUATION RESULTS SUMMARY"):
    """
    Print a beautifully aligned markdown table of retrieval results.
    """
    import textwrap
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    # Set a maximum width for the query column
    query_col_width = 25
    max_query_len = max(len(q) for q in results.keys())
    actual_query_width = min(max(max_query_len, len("Query String")), query_col_width)
    
    # Use shorter header names for terminal fitting (R@K, P@K)
    metric_headers = []
    for k in k_values:
        metric_headers.append(f"R@{k}")
    for k in k_values:
        metric_headers.append(f"P@{k}")
        
    header_cols = ["Query String"] + metric_headers
    # Short metrics (0.0000) are 6 chars, headers are 3-5 chars.
    col_widths = [actual_query_width] + [max(len(h), 6) for h in metric_headers]
    
    header_str = " | ".join(f"{col:<{width}}" for col, width in zip(header_cols, col_widths))
    header_str = f"| {header_str} |"
    print(header_str)
    
    divider_str = " | ".join("-" * width for width in col_widths)
    divider_str = f"| {divider_str} |"
    print(divider_str)
    
    for query_str, metrics in results.items():
        # Wrap query string into lines of max length actual_query_width
        query_lines = textwrap.wrap(query_str, width=actual_query_width)
        if not query_lines:
            query_lines = [""]
            
        # First line of the wrapped query contains the metrics
        row_vals = [query_lines[0]]
        for k in k_values:
            row_vals.append(f"{metrics[f'Recall@{k}']:.4f}")
        for k in k_values:
            row_vals.append(f"{metrics[f'Precision@{k}']:.4f}")
        row_str = " | ".join(f"{val:<{width}}" for val, width in zip(row_vals, col_widths))
        print(f"| {row_str} |")
        
        # Subsequent lines contain empty space for metrics
        for line in query_lines[1:]:
            row_vals = [line] + [""] * len(metric_headers)
            row_str = " | ".join(f"{val:<{width}}" for val, width in zip(row_vals, col_widths))
            print(f"| {row_str} |")
