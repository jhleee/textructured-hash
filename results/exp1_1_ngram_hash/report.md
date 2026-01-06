# Experiment Results: ngram_hash

## Configuration

- Model: ngram_hash
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.6772 |
| Best F1 | 0.5405 |
| Best Threshold | 0.10 |
| Separation | 0.3888 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.6140 |
| Mean Positive Sim | 0.1222 |
| Mean Negative Sim | 0.0112 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 5431 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1177840 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.6772 | ✗ |
| Separation | ≥2.5 | 0.3888 | ✗ |
| Best F1 | ≥0.88 | 0.5405 | ✗ |
| Encoding Speed | ≥10,000/s | 5431/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.1222 | ✗ |
| Mean Negative Sim | ≤0.35 | 0.0112 | ✓ |
