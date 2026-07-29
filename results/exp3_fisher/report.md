# Experiment Results: fisher

## Configuration

- Model: fisher
- Vector Dimension: 256
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9979 |
| Best F1 | 0.9874 |
| Best Threshold | 0.50 |
| Separation | 3.7448 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.9790 |
| Mean Positive Sim | 0.9472 |
| Mean Negative Sim | -0.0365 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 73570 texts/sec |
| Vector Bytes | 256 bytes |
| Vector Dimensions | 256 |
| Comparison Speed | 1676337 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9979 | ✓ |
| Separation | ≥2.5 | 3.7448 | ✓ |
| Best F1 | ≥0.88 | 0.9874 | ✓ |
| Encoding Speed | ≥10,000/s | 73570/s | ✓ |
| Vector Bytes | ≤256 | 256 | ✓ |
| Mean Positive Sim | ≥0.85 | 0.9472 | ✓ |
| Mean Negative Sim | ≤0.35 | -0.0365 | ✓ |
