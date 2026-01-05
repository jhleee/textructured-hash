# Experiment Results: structure_type_quantized

## Configuration

- Model: structure_type_quantized
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9679 |
| Best F1 | 0.8999 |
| Best Threshold | 0.70 |
| Separation | 1.3310 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8940 |
| Mean Positive Sim | 0.9247 |
| Mean Negative Sim | 0.4488 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 6463 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1308958 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9679 | ✓ |
| Separation | ≥2.5 | 1.3310 | ✗ |
| Best F1 | ≥0.88 | 0.8999 | ✓ |
| Encoding Speed | ≥10,000/s | 6463/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.9247 | ✓ |
| Mean Negative Sim | ≤0.35 | 0.4488 | ✗ |
