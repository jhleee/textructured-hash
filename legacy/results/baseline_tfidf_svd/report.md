# Experiment Results: tfidf_svd

## Configuration

- Model: tfidf_svd
- Vector Dimension: 128
- Key Parameters: {'max_features': 10000, 'ngram_range': (2, 4)}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9203 |
| Best F1 | 0.8456 |
| Best Threshold | 0.10 |
| Separation | 0.9730 |
| Precision@100 | 1.0000 |
| Precision@1000 | 0.8520 |
| Mean Positive Sim | 0.4241 |
| Mean Negative Sim | 0.0323 |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | 1252 texts/sec |
| Vector Bytes | 512 bytes |
| Vector Dimensions | 128 |
| Comparison Speed | 1257739 comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | 0.9203 | ✓ |
| Separation | ≥2.5 | 0.9730 | ✗ |
| Best F1 | ≥0.88 | 0.8456 | ✗ |
| Encoding Speed | ≥10,000/s | 1252/s | ✗ |
| Vector Bytes | ≤256 | 512 | ✗ |
| Mean Positive Sim | ≥0.85 | 0.4241 | ✗ |
| Mean Negative Sim | ≤0.35 | 0.0323 | ✓ |
