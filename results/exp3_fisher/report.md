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
| Best Threshold | 0.60 |
| Separation | 3.7027 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.9790 |
| Mean Positive Sim | 0.9469 |
| Mean Negative Sim | -0.0377 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 74495 texts/sec |
| Vector Bytes | 256 bytes |
| Vector Dimensions | 256 |
| Comparison Speed | 1644185 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9977 | ✓ |
| Separation | ≥2.5 | 3.7027 | ✓ |
| Best F1 | ≥0.88 | 0.9874 | ✓ |
| Encoding Speed | ≥10,000/s | 74495/s | ✓ |
| Vector Bytes | ≤256 | 256 | ✓ |
| Mean Positive Sim | ≥0.85 | 0.9469 | ✓ |
| Mean Negative Sim | ≤0.35 | -0.0377 | ✓ |
