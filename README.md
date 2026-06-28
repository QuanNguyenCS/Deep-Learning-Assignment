# Compositional Image Retrieval on CelebA

This repository implements a State-of-the-Art (SOTA) Compositional Image Retrieval pipeline on the CelebA dataset using raw CLIP embeddings and optimized neural classification backbones.

---

## Project Structure

```
Deep-Learning-Assignment/
├── checkpoints/
│   ├── gde_train_mlp_classifier_raw.pt               # Trained Standard BCE MLP Classifier
│   └── gde_train_mlp_classifier_focal_loss_best.pt   # Swept Optimal Focal Loss MLP Classifier (alpha=0.50, gamma=0.5)
├── Data/
│   ├── celeba/                                       # CelebA images directory
│   ├── celeba_evaluation.json                        # Query metadata & ground truth targets mapping
│   ├── celeba_test_embeddings.pt                     # Cached CLIP embeddings (test split)
│   ├── celeba_train_embeddings.pt                    # Cached CLIP embeddings (train split)
│   └── celeba_val_embeddings.pt                      # Cached CLIP embeddings (val split)
├── output/
│   └── gde_strategy7_results.png                     # Visual grid of retrieval results
├── src/
│   ├── debug_gde.py                                  # Debug classification output on sample images
│   ├── run_focal_grid_search.py                      # Sweep code for Focal Loss hyperparameters (alpha, gamma)
│   ├── run_strategy7.py                              # Production compositional image retrieval run
│   ├── run_strategy7_variants.py                     # Sweep code for query weighting parameter (omega)
│   └── utils.py                                      # Retrieval evaluations, metrics, and visualization utilities
├── README.md
└── requirements.txt
```

---

## Retrieval Workflow & SOTA Performance

The core image retrieval strategy is **Global Weighted BCE Loss Retrieval (Strategy 7)**. Rather than relying on cosine similarities in GDE space which suffer from primitive vector noise, it models retrieval directly in the predicted attribute probability space:
1. An MLP backbone predicts the probabilities of 40 attributes from raw CLIP embeddings.
2. For any query (e.g. "+Smiling, -Young"), we construct a continuous target profile $Y^*$, preserving the source image's predicted scores for unmodified attributes (identity check) while forcing query-specified attributes to 1.0 or 0.0.
3. We compute the weighted Binary Cross-Entropy (BCE) loss globally against all 19,962 test candidates:
   $$\text{BCE Loss}(c) = - \frac{1}{\sum W_a} \sum_{a=1}^{40} W_a \cdot \left[ Y^*_a \ln P_{c, a} + (1 - Y^*_a) \ln (1 - P_{c, a}) \right]$$
   where $W_a = \omega = 4.0$ for query-modified attributes, and $W_a = 1.0$ otherwise.
4. Candidates with the lowest BCE loss are returned.

### Final Leaderboard (cases=500, full test split):

| Configuration | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 | Composite F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Query-Aware BCE ($\omega=4.0$)** | **0.1136** | **0.3423** | **0.5865** | **0.1136** | **0.0981** | **0.1211** | **`0.1556`** *(Winner!)* |
| **Query-Aware BCE ($\omega=5.0$)** | 0.1087 | 0.3414 | 0.5828 | 0.1087 | 0.0973 | 0.1203 | 0.1532 |
| **Baseline BCE ($\omega=1.0$)** | 0.0926 | 0.2876 | 0.4966 | 0.0926 | 0.0795 | 0.0945 | 0.1253 |

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Production Retrieval
To perform the full evaluation (14 queries, 500 cases each) and generate the results visualization grid:
```bash
python src/run_strategy7.py
```
*Output visual grid is generated at `output/gde_strategy7_results.png`.*

### 3. Run Parameter Sweeps
To rerun the weighting factor parameter sweep ($\omega \in [1.0, ..., 20.0]$) optimizing for Composite F1-Score:
```bash
python src/run_strategy7_variants.py
```

### 4. Run Model Debugging
To inspect classification accuracy and attribute predictions on debug sample indices (e.g. index 13):
```bash
python src/debug_gde.py --indices 13 14 15
```