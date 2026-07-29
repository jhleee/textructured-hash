# Independent Holdout Validation Protocol

## Motivation

The Fisher Structure Encoder achieves 7/7 success criteria on the **development benchmark**
(synthetic.jsonl, seed=42, 24 categories, 15,300 texts, 9,992 pairs). However, this benchmark
was used iteratively during algorithm design — hyperparameters, feature design, and contrastive
training were all tuned against it. To confirm **out-of-distribution generalization**, we must
evaluate the frozen algorithm on data it has never influenced.

This document defines the protocol for that independent holdout re-validation.

---

## 1. Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Algorithm freeze** | No code, hyperparameter, or threshold changes after holdout generation |
| **Independent generator** | New seed, new templates, new vocabulary — no overlap with dev data |
| **Source-family holdout** | Include at least 2 entirely new category families not in the 24 dev categories |
| **Blind evaluation** | Same metrics pipeline; no per-category inspection until after aggregate pass/fail |
| **Reproducibility** | Fixed seed (holdout_seed=2024), deterministic pair sampling |

---

## 2. Holdout Dataset Specification

### 2.1 Data Independence Guarantees

| Dimension | Development Set | Holdout Set |
|-----------|----------------|-------------|
| Generator seed | 42 | 2024 |
| Template vocabulary | Original (e.g., domains: `example, test, demo…`) | Entirely new vocabulary (e.g., domains: `acme, globex, initech…`) |
| Category coverage | 24 categories | Same 24 + **4 new unseen families** |
| Text count | ~15,300 | ~8,000 (lower count to avoid overpowering noise) |
| Pair count | 9,992 | 6,000 (3,000 pos + 3,000 neg) |
| Pair seed | 42 | 7777 |
| Split | 60/20/20 | 100% test (no train/val — model is frozen) |

### 2.2 New Source Families (Unseen Categories)

These categories were **not present** in the development set and test whether the encoder
generalizes to structural patterns it was never trained on:

1. **`markdown`** — Markdown-formatted text (headings, lists, bold/italic, links)
2. **`log_entry`** — Timestamped log lines (syslog, Apache, JSON-structured logs)
3. **`regex`** — Regular expression patterns of varying complexity
4. **`ini_config`** — INI/TOML configuration file sections

### 2.3 Modified Templates for Existing Categories

For the 24 existing categories, the holdout generator uses:

- **Different vocabulary**: new domain names, person names, variable names, etc.
- **Different format distributions**: shifted probabilities for format variants
- **Different length distributions**: slightly wider/narrower ranges
- **Different language content**: new sentence templates for Korean/English/Chinese/Japanese

This ensures that even for known categories, the encoder cannot rely on memorized token patterns.

---

## 3. Frozen Model Definition

The model under evaluation is **FisherStructureEncoder** with:

- Feature dim: 320 (numba-compiled)
- Output dim: 256
- Training: Fisher LDA + 5 epochs contrastive fine-tuning
- Training data: `data/train.jsonl` (development set, seed=42)
- Normalization: saved `feature_mean`, `feature_std`
- Projection: saved `W` matrix

The model is loaded from a saved `.npz` checkpoint. **No retraining is performed.**

---

## 4. Evaluation Protocol

### 4.1 Steps

```bash
# 1. Freeze: save the trained model (if not already saved)
python scripts/evaluate_holdout.py --phase save_model \
    --train data/train.jsonl \
    --model_path results/holdout_validation/frozen_model.npz

# 2. Generate: create independent holdout data
python scripts/generate_holdout_data.py \
    --output data/holdout/ \
    --seed 2024 \
    --pair_seed 7777 \
    --n_positive 3000 \
    --n_negative 3000

# 3. Evaluate: run frozen model on holdout
python scripts/evaluate_holdout.py --phase evaluate \
    --model_path results/holdout_validation/frozen_model.npz \
    --holdout_pairs data/holdout/pairs.jsonl \
    --output results/holdout_validation/
```

### 4.2 Metrics (identical to dev benchmark)

| Metric | Target | Pass Condition |
|--------|--------|----------------|
| AUC-ROC | ≥ 0.92 | Hard requirement |
| Separation | ≥ 2.5 | Hard requirement |
| Best F1 | ≥ 0.88 | Hard requirement |
| Mean Positive Sim | ≥ 0.85 | Hard requirement |
| Mean Negative Sim | ≤ 0.35 | Hard requirement |
| Encoding Speed | ≥ 10,000/s | Hard requirement |
| Vector Bytes (int8) | ≤ 256 | Hard requirement |

### 4.3 Additional Analyses

After aggregate pass/fail is determined:

1. **Per-category-family breakdown**: How does performance differ between
   "known 24" categories and "new 4" families?
2. **Cross-family pairs**: Accuracy on pairs where one text is from a known
   category and the other is from a new family.
3. **Hardest pairs**: Bottom-10 worst predictions — manual inspection.
4. **Confidence interval**: Bootstrap 95% CI on AUC-ROC (1000 resamples).

---

## 5. Pass/Fail Criteria

| Outcome | Definition |
|---------|------------|
| **FULL PASS** | All 7 metrics meet targets on the holdout set |
| **SOFT PASS** | ≥5/7 metrics pass AND AUC-ROC ≥ 0.92 AND new families AUC ≥ 0.85 |
| **FAIL** | AUC-ROC < 0.92 OR ≥3 metrics fail |

### Interpretation

- **FULL PASS** → Algorithm generalizes; ready for production evaluation.
- **SOFT PASS** → Investigate specific failure modes; may need targeted improvement for new families.
- **FAIL** → Overfitting to development set; requires redesign or regularization.

---

## 6. Reporting Template

```markdown
# Holdout Validation Results

## Summary
- Date: YYYY-MM-DD
- Model: FisherStructureEncoder (frozen from dev training)
- Holdout size: X texts, Y pairs
- Result: [FULL PASS / SOFT PASS / FAIL]

## Aggregate Metrics

| Metric | Target | Holdout | Dev Benchmark | Delta |
|--------|--------|---------|---------------|-------|
| AUC-ROC | ≥0.92 | X.XXXX | 0.9977 | -X.XX% |
| ... | ... | ... | ... | ... |

## Per-Family Breakdown

| Family | Pairs | AUC-ROC | Mean Sim (pos) | Mean Sim (neg) |
|--------|-------|---------|----------------|----------------|
| Known 24 (intra) | ... | ... | ... | ... |
| New families (intra) | ... | ... | ... | ... |
| Cross-family | ... | ... | ... | ... |

## Failure Analysis
[If applicable]

## Conclusion
[Statement on generalization]
```

---

## 7. Timeline & Dependencies

| Step | Dependency | Estimated Time |
|------|------------|----------------|
| Freeze model checkpoint | Existing train pipeline | 1 min |
| Implement holdout generator | None (independent) | New script |
| Generate holdout data | Generator script | ~10 sec |
| Run evaluation | Frozen model + holdout data | ~30 sec |
| Write report | Evaluation results | Manual |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Holdout generator accidentally shares templates | Code review: no imports from `src/data/generator.py` |
| Model checkpoint incompatible | Test save/load round-trip before holdout generation |
| New category features OOD for normalization | Feature normalization is per-dimension; new patterns still produce bounded features |
| Contrastive training overfit to 24-category structure | This is exactly what holdout validation detects |

---

**End of Protocol**
