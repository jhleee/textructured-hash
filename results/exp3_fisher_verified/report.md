# Experiment Results: fisher

## Configuration

- Model: fisher
- Vector Dimension: 256
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9977 |
| Best F1 | 0.9874 |
| Best Threshold | 0.55 |
| Separation | 3.6847 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.9790 |
| Mean Positive Sim | 0.9458 |
| Mean Negative Sim | -0.0334 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 73019 texts/sec |
| Vector Bytes | 256 bytes |
| Vector Dimensions | 256 |
| Comparison Speed | 1652781 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9977 | ✓ |
| Separation | ≥2.5 | 3.6847 | ✓ |
| Best F1 | ≥0.88 | 0.9874 | ✓ |
| Encoding Speed | ≥10,000/s | 73019/s | ✓ |
| Vector Bytes | ≤256 | 256 | ✓ |
| Mean Positive Sim | ≥0.85 | 0.9458 | ✓ |
| Mean Negative Sim | ≤0.35 | -0.0334 | ✓ |
