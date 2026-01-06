# Experiment Results: pattern_free

## Configuration

- Model: pattern_free
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9328 |
| Best F1 | 0.8523 |
| Best Threshold | 0.90 |
| Separation | 1.0344 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8490 |
| Mean Positive Sim | 0.9536 |
| Mean Negative Sim | 0.7456 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 2032 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1361405 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9328 | ✓ |
| Separation | ≥2.5 | 1.0344 | ✗ |
| Best F1 | ≥0.88 | 0.8523 | ✗ |
| Encoding Speed | ≥10,000/s | 2032/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.9536 | ✓ |
| Mean Negative Sim | ≤0.35 | 0.7456 | ✗ |
