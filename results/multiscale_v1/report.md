# Experiment Results: multiscale

## Configuration

- Model: multiscale
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9549 |
| Best F1 | 0.8775 |
| Best Threshold | 0.85 |
| Separation | 1.2141 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8710 |
| Mean Positive Sim | 0.9475 |
| Mean Negative Sim | 0.6432 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 9340 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1218827 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9549 | ✓ |
| Separation | ≥2.5 | 1.2141 | ✗ |
| Best F1 | ≥0.88 | 0.8775 | ✗ |
| Encoding Speed | ≥10,000/s | 9340/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.9475 | ✓ |
| Mean Negative Sim | ≤0.35 | 0.6432 | ✗ |
