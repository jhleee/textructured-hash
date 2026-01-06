# Experiment Results: ngram_hash_multiscale

## Configuration

- Model: ngram_hash_multiscale
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7577 |
| Best F1 | 0.6544 |
| Best Threshold | 0.10 |
| Separation | 0.5241 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.6840 |
| Mean Positive Sim | 0.1669 |
| Mean Negative Sim | 0.0155 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 6146 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1380804 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.7577 | ✗ |
| Separation | ≥2.5 | 0.5241 | ✗ |
| Best F1 | ≥0.88 | 0.6544 | ✗ |
| Encoding Speed | ≥10,000/s | 6146/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.1669 | ✗ |
| Mean Negative Sim | ≤0.35 | 0.0155 | ✓ |
