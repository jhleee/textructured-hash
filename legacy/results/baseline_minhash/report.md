# Experiment Results: minhash

## Configuration

- Model: minhash
- Vector Dimension: 128
- Key Parameters: {'ngram_size': 3}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.5785 |
| Best F1 | 0.6597 |
| Best Threshold | 0.10 |
| Separation | 0.1470 |
| Precision@100 | 0.7100 |
| Precision@1000 | 0.5540 |
| Mean Positive Sim | 0.6071 |
| Mean Negative Sim | 0.5773 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 531 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1310836 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.5785 | ✗ |
| Separation | ≥2.5 | 0.1470 | ✗ |
| Best F1 | ≥0.88 | 0.6597 | ✗ |
| Encoding Speed | ≥10,000/s | 531/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.6071 | ✗ |
| Mean Negative Sim | ≤0.35 | 0.5773 | ✗ |
