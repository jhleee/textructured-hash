# Experiment Results: structure_type_quantized_256

## Configuration

- Model: structure_type_quantized_256
- Vector Dimension: 256
- Key Parameters: {}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9687 |
| Best F1 | 0.9048 |
| Best Threshold | 0.75 |
| Separation | 1.3749 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8770 |
| Mean Positive Sim | 0.9242 |
| Mean Negative Sim | 0.4660 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 7597 texts/sec |
| Vector Bytes | 1024 bytes |
| Vector Dimensions | 256 |
| Comparison Speed | 1225011 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9687 | ✓ |
| Separation | ≥2.5 | 1.3749 | ✗ |
| Best F1 | ≥0.88 | 0.9048 | ✓ |
| Encoding Speed | ≥10,000/s | 7597/s | ✗ |
| Vector Bytes | ≤256 | 1024 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.9242 | ✓ |
| Mean Negative Sim | ≤0.35 | 0.4660 | ✗ |
