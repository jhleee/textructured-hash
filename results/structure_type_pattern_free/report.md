# Experiment Results: structure_type

## Configuration

- Model: structure_type
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9589 |
| Best F1 | 0.8796 |
| Best Threshold | 0.90 |
| Separation | 1.0748 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8820 |
| Mean Positive Sim | 0.9637 |
| Mean Negative Sim | 0.6170 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 6733 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1332732 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9589 | ✓ |
| Separation | ≥2.5 | 1.0748 | ✗ |
| Best F1 | ≥0.88 | 0.8796 | ✗ |
| Encoding Speed | ≥10,000/s | 6733/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.9637 | ✓ |
| Mean Negative Sim | ≤0.35 | 0.6170 | ✗ |
