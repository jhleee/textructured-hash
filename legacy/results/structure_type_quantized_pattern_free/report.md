# Experiment Results: structure_type_quantized

## Configuration

- Model: structure_type_quantized
- Vector Dimension: 128
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9590 |
| Best F1 | 0.8800 |
| Best Threshold | 0.90 |
| Separation | 1.0752 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8810 |
| Mean Positive Sim | 0.9634 |
| Mean Negative Sim | 0.6170 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 5896 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1367061 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9590 | ✓ |
| Separation | ≥2.5 | 1.0752 | ✗ |
| Best F1 | ≥0.88 | 0.8800 | ✓ |
| Encoding Speed | ≥10,000/s | 5896/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.9634 | ✓ |
| Mean Negative Sim | ≤0.35 | 0.6170 | ✗ |
