# Experiment 002: Ablation - Contrastive Fine-Tuning Impact

## Date
2025-01-XX (follow-up to Experiment 001)

## Objective

Measure the contribution of contrastive fine-tuning (Step 5 in the training pipeline) by
comparing Fisher LDA-only projection vs. Fisher LDA + contrastive SGD. The review of
Experiment 001 noted the absence of ablation data proving the contrastive pass helps.

## Setup

Two configurations tested on the same data (train.jsonl / test.jsonl):

1. **Fisher LDA only**: Train the full pipeline (feature extraction, normalization, scatter
   matrices, eigenvalue problem) but skip the contrastive SGD loop entirely. The projection
   matrix W is the raw top-256 eigenvectors from the Fisher generalized eigenvalue problem.

2. **Fisher LDA + Contrastive** (full model): After Fisher LDA, run 5 epochs of contrastive
   SGD on 3,000 sampled pairs (target positive sim > 0.95, push negatives below margin 0.1,
   learning rate 0.001 with 0.8x decay per epoch).

Both use identical: feature extraction (320-dim numba-compiled), normalization (zero-mean,
unit-variance), and encoding pipeline (project + L2 norm). Seed=42 throughout.

## Results

| Metric | Target | LDA Only | LDA + Contrastive | Delta |
|--------|--------|----------|-------------------|-------|
| AUC-ROC | >= 0.92 | 0.9894 | **0.9980** | +0.0086 |
| Separation | >= 2.5 | 2.0722 | **3.7704** | +1.6982 |
| Best F1 | >= 0.88 | 0.9449 | **0.9874** | +0.0425 |
| Mean Pos Sim | >= 0.85 | 0.8412 | **0.9465** | +0.1053 |
| Mean Neg Sim | <= 0.35 | -0.0105 | **-0.0368** | -0.0263 |
| Speed (texts/s) | >= 10,000 | 77,473 | **76,202** | -1,271 |
| Vector Bytes | <= 256 | 256 | **256** | 0 |

### Criteria Pass/Fail

| Metric | LDA Only | LDA + Contrastive |
|--------|----------|-------------------|
| AUC-ROC >= 0.92 | PASS | PASS |
| Separation >= 2.5 | **FAIL (2.07)** | PASS |
| Best F1 >= 0.88 | PASS | PASS |
| Mean Pos Sim >= 0.85 | **MARGINAL (0.841)** | PASS |
| Mean Neg Sim <= 0.35 | PASS | PASS |
| Speed >= 10,000/s | PASS | PASS |
| Vector Bytes <= 256 | PASS | PASS |

LDA-only: **5/7 criteria pass** (fails separation, borderline on pos sim)
LDA + Contrastive: **7/7 criteria pass**

## Analysis

### Where Contrastive Helps Most

1. **Separation (+82%)**: This is the largest improvement. Fisher LDA maximizes between-class
   scatter but does not directly optimize the similarity metric used in evaluation. The
   contrastive pass explicitly optimizes pairwise cosine similarity -- pulling positive pairs
   above 0.95 and pushing negatives below 0.1 -- which directly maximizes the numerator and
   minimizes the denominator of the separation formula.

2. **Mean Positive Similarity (+12.5%)**: Fisher LDA separates classes well in a global sense
   but does not ensure same-class vectors are highly aligned. The contrastive pass explicitly
   targets sim > 0.95 for positive pairs, significantly boosting intra-class cohesion from
   0.84 to 0.95.

3. **Best F1 (+4.5%)**: Tighter clusters with more gap between positive/negative distributions
   means the optimal threshold becomes more effective, yielding better precision and recall.

### Where Contrastive Does Not Help

- **Speed**: Slight decrease (~1.6%) due to noise -- both configurations use identical
  inference code. The difference is within run-to-run variance.
- **Mean Negative Similarity**: Already near zero with LDA-only (-0.01). The contrastive pass
  pushes it slightly more negative (-0.04) but the primary work was already done by Fisher LDA.

### Why Fisher LDA Alone Falls Short

Fisher LDA solves for directions that maximize `Sb / Sw` (between-class vs within-class
scatter). This is optimal for class separation in a Euclidean sense, but the evaluation uses
cosine similarity after L2 normalization. The L2 norm projection introduces non-linearity
that Fisher LDA does not account for. Specifically:

- Two vectors can be well-separated in Euclidean distance but have high cosine similarity if
  they point in similar directions
- Fisher's eigenvectors are orthogonal in the original space but become non-orthogonal after
  projection and normalization

The contrastive pass corrects for this mismatch by directly optimizing the cosine similarity
objective, effectively adapting the Fisher projection to the actual evaluation metric.

### Overfitting Concern

The contrastive pass trains on 3,000 pairs sampled from training data and evaluates on held-out
test data. The consistent improvement across all metrics on test data (not just training data)
suggests the contrastive pass captures genuine structural patterns rather than overfitting to
training-specific noise. However, the lack of a validation split means we cannot detect subtle
overfitting. Future work could add a 20% validation split from training pairs.

## Conclusion

The contrastive fine-tuning is **essential** for meeting all 7 criteria. Without it, the encoder
fails on separation (2.07 vs target 2.5) and is borderline on mean positive similarity
(0.841 vs target 0.85). The contrastive pass provides:

- +82% improvement in separation (the hardest criterion to meet)
- +12.5% improvement in positive similarity
- Negligible speed cost (same inference code)

The combination of Fisher LDA (global class separation) + contrastive SGD (pairwise similarity
optimization) is synergistic: Fisher provides a strong initialization that the contrastive pass
refines for the cosine similarity objective.
