# Experiment: Dimension Sweep for Int8 Quantization

## Objective

Test how int8 quantized StructureTypeEncoder performance varies with different dimensions.

## Methodology

- **Model**: QuantizedStructureTypeCompactEncoder
- **Dimensions tested**: 32, 64, 128, 256, 512, 1024
- **Type dimension**: Scaled proportionally (dim // 8, minimum 8)
- **Quantization**: Float32 → Int8 (127 levels)
- **Seed**: 42 (reproducible)

---

## Results Summary

### Quality Metrics Comparison

| Dimension | AUC-ROC | Best F1 | Separation | Mean Pos Sim | Mean Neg Sim | Speed (texts/s) | Int8 Bytes |
|-----------|---------|---------|------------|--------------|--------------|-----------------|------------|
|   64 | 0.9570 | 0.8812 | 1.2541 | 0.9256 | 0.4956 |    7565 |  64 |
|  128 | 0.9679 | 0.8999 | 1.3310 | 0.9247 | 0.4488 |    7164 | 128 |
|  256 | 0.9687 | 0.9048 | 1.3749 | 0.9242 | 0.4660 |    7109 | 256 |
|  512 | 0.9710 | 0.9042 | 1.4009 | 0.9261 | 0.4777 |    7108 | 512 |
| 1024 | 0.9715 | 0.9034 | 1.4207 | 0.9225 | 0.4665 |    6567 | 1024 |

### Success Criteria Achievement

| Dimension | AUC ≥0.92 | F1 ≥0.88 | Sep ≥2.5 | Pos ≥0.85 | Neg ≤0.35 | Speed ≥10k | Bytes ≤256 | Total |
|-----------|-----------|----------|----------|-----------|-----------|------------|------------|-------|
|   64 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | 4/7 |
|  128 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | 4/7 |
|  256 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | 4/7 |
|  512 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | 3/7 |
| 1024 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | 3/7 |

---

## Detailed Analysis

### Dimension: 64

**Configuration**:
- Total dimension: 64
- Type dimension: 8
- Int8 storage: 64 bytes
- Float32 storage (eval): 256 bytes

**Quality Metrics**:
- AUC-ROC: 0.9570
- Best F1: 0.8812 (threshold: 0.75)
- Separation: 1.2541
- Mean Positive Similarity: 0.9256
- Mean Negative Similarity: 0.4956
- Precision@100: 1.0000
- Precision@1000: 0.8610

**Efficiency Metrics**:
- Encoding Speed: 7565 texts/sec
- Comparison Speed: 885852 comparisons/sec

---

### Dimension: 128

**Configuration**:
- Total dimension: 128
- Type dimension: 16
- Int8 storage: 128 bytes
- Float32 storage (eval): 512 bytes

**Quality Metrics**:
- AUC-ROC: 0.9679
- Best F1: 0.8999 (threshold: 0.70)
- Separation: 1.3310
- Mean Positive Similarity: 0.9247
- Mean Negative Similarity: 0.4488
- Precision@100: 1.0000
- Precision@1000: 0.8940

**Efficiency Metrics**:
- Encoding Speed: 7164 texts/sec
- Comparison Speed: 1267599 comparisons/sec

---

### Dimension: 256

**Configuration**:
- Total dimension: 256
- Type dimension: 32
- Int8 storage: 256 bytes
- Float32 storage (eval): 1024 bytes

**Quality Metrics**:
- AUC-ROC: 0.9687
- Best F1: 0.9048 (threshold: 0.75)
- Separation: 1.3749
- Mean Positive Similarity: 0.9242
- Mean Negative Similarity: 0.4660
- Precision@100: 1.0000
- Precision@1000: 0.8770

**Efficiency Metrics**:
- Encoding Speed: 7109 texts/sec
- Comparison Speed: 1231314 comparisons/sec

---

### Dimension: 512

**Configuration**:
- Total dimension: 512
- Type dimension: 64
- Int8 storage: 512 bytes
- Float32 storage (eval): 2048 bytes

**Quality Metrics**:
- AUC-ROC: 0.9710
- Best F1: 0.9042 (threshold: 0.75)
- Separation: 1.4009
- Mean Positive Similarity: 0.9261
- Mean Negative Similarity: 0.4777
- Precision@100: 1.0000
- Precision@1000: 0.8920

**Efficiency Metrics**:
- Encoding Speed: 7108 texts/sec
- Comparison Speed: 1143841 comparisons/sec

---

### Dimension: 1024

**Configuration**:
- Total dimension: 1024
- Type dimension: 128
- Int8 storage: 1024 bytes
- Float32 storage (eval): 4096 bytes

**Quality Metrics**:
- AUC-ROC: 0.9715
- Best F1: 0.9034 (threshold: 0.75)
- Separation: 1.4207
- Mean Positive Similarity: 0.9225
- Mean Negative Similarity: 0.4665
- Precision@100: 1.0000
- Precision@1000: 0.8930

**Efficiency Metrics**:
- Encoding Speed: 6567 texts/sec
- Comparison Speed: 1157248 comparisons/sec

---

## Recommendations

### Best Quality (F1 Score)
- **Dimension 256**: F1 = 0.9048, AUC = 0.9687
- Storage: 256 bytes (int8)

### Best Speed
- **Dimension 64**: 7565 texts/sec
- Quality: F1 = 0.8812, AUC = 0.9570

### Best Memory Efficiency
- **Dimension 64**: 64 bytes (int8)
- Quality: F1 = 0.8812, AUC = 0.9570

### Best Separation
- **Dimension 1024**: 1.4207
- Quality: F1 = 0.9034, AUC = 0.9715

### Balanced Recommendation

Based on meeting targets and trade-offs:

**Recommended: Dimension 256**
- Criteria met: 4/7
- F1 Score: 0.9048
- AUC-ROC: 0.9687
- Storage: 256 bytes (int8)
- Speed: 7109 texts/sec

