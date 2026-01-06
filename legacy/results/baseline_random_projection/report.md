# Experiment Results: random_projection

## Configuration

- Model: random_projection
- Vector Dimension: 128
- Key Parameters: {'input_dim': 1024, 'seed': 42}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8994 |
| Best F1 | 0.8243 |
| Best Threshold | 0.30 |
| Separation | 0.9681 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8170 |
| Mean Positive Sim | 0.5762 |
| Mean Negative Sim | 0.1639 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 42894 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1252585 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.8994 | ✗ |
| Separation | ≥2.5 | 0.9681 | ✗ |
| Best F1 | ≥0.88 | 0.8243 | ✗ |
| Encoding Speed | ≥10,000/s | 42894/s | ✓ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.5762 | ✗ |
| Mean Negative Sim | ≤0.35 | 0.1639 | ✓ |
