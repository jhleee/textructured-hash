# Experiment Results: structure_type_fast

## Configuration

- Model: structure_type_fast
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9447 |
| Best F1 | 0.8721 |
| Best Threshold | 0.85 |
| Separation | 1.2609 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8500 |
| Mean Positive Sim | 0.8871 |
| Mean Negative Sim | 0.3600 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 8857 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1365311 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9447 | ✓ |
| Separation | ≥2.5 | 1.2609 | ✗ |
| Best F1 | ≥0.88 | 0.8721 | ✗ |
| Encoding Speed | ≥10,000/s | 8857/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.8871 | ✓ |
| Mean Negative Sim | ≤0.35 | 0.3600 | ✗ |
