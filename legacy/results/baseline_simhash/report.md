# Experiment Results: simhash

## Configuration

- Model: simhash
- Vector Dimension: 128
- Key Parameters: {'ngram_size': 3}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.5847 |
| Best F1 | 0.5137 |
| Best Threshold | 0.10 |
| Separation | 0.0775 |
| Precision@100 | 0.9800 |
| Precision@1000 | 0.5140 |
| Mean Positive Sim | 0.0639 |
| Mean Negative Sim | -0.0163 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 1132 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1271009 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.5847 | ✗ |
| Separation | ≥2.5 | 0.0775 | ✗ |
| Best F1 | ≥0.88 | 0.5137 | ✗ |
| Encoding Speed | ≥10,000/s | 1132/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.0639 | ✗ |
| Mean Negative Sim | ≤0.35 | -0.0163 | ✓ |
