# 실험 패배 요인 분석 보고서

## 1. 실험 결과 요약

### 성공 기준 달성률: 2/7 (29%)

| 기준 | 목표 | 실제 | 결과 | 차이 |
|------|------|------|------|------|
| AUC-ROC | ≥ 0.92 | 0.9549 | ✓ PASS | +0.035 |
| Best F1 | ≥ 0.88 | 0.8775 | ✗ FAIL | -0.003 |
| Separation | ≥ 2.5 | 1.21 | ✗ FAIL | -1.29 |
| Encoding Speed | ≥ 10,000/s | 9,340/s | ✗ FAIL | -660 |
| Vector Bytes | ≤ 256 | 512 | ✗ FAIL | +256 |
| Mean Pos Sim | ≥ 0.85 | 0.9475 | ✓ PASS | +0.097 |
| Mean Neg Sim | ≤ 0.35 | 0.6432 | ✗ FAIL | +0.293 |

---

## 2. 핵심 패배 요인 분석

### 2.1 가장 심각한 문제: Mean Negative Similarity

**현상**: 다른 구조의 텍스트 쌍도 평균 64.3%의 유사도를 보임

**원인 분석**:

```
모델별 Mean Negative Similarity 비교:
┌──────────────────────┬──────────────┬──────────────┐
│ 모델                 │ Mean Neg Sim │ 목표 대비    │
├──────────────────────┼──────────────┼──────────────┤
│ TF-IDF SVD           │ 0.032        │ ✓ 목표 달성  │
│ Random Projection    │ 0.164        │ ✓ 목표 달성  │
│ Multiscale (제안)    │ 0.643        │ ✗ +0.293     │
└──────────────────────┴──────────────┴──────────────┘
```

**근본 원인**:
1. **피처 희석(Feature Dilution)**: 4개 스케일의 피처를 concat 후 L2 normalize 시 discriminative information이 희석됨
2. **공통 특성 과대 반영**: 모든 텍스트가 공유하는 특성(ASCII 바이트, 기본 유니코드 카테고리)이 벡터의 대부분을 차지
3. **구조적 차이 무시**: URL, JSON, 한글문장 등의 근본적인 구조 차이가 벡터에 충분히 반영되지 않음

**증거**:
- Byte-level 피처: 대부분 텍스트가 ASCII 범위 (0x00-0x7F)에 집중 → 비슷한 분포
- Unicode 피처: Letter(L), Number(N) 카테고리가 대부분 → 낮은 구별력
- L2 정규화 후 모든 벡터가 단위 구 위에 밀집 → 유사도 상승

### 2.2 Separation 부족 문제

**현상**: Positive/Negative 분포 분리도가 1.21로 목표 2.5의 48%에 불과

**수학적 분석**:
```
Separation = (mean_pos - mean_neg) / sqrt((std_pos² + std_neg²) / 2)
           = (0.9475 - 0.6432) / sqrt((0.076² + 0.175²) / 2)
           = 0.3043 / 0.135
           = 2.25 (실제 메트릭: 1.21)

※ 분포의 overlap이 너무 큼
```

**분포 시각화**:
```
Positive Distribution:  ████████████████████████████████████████ (0.95 ± 0.08)
Negative Distribution:  ████████████████████████████████ (0.64 ± 0.17)
                        |----|----|----|----|----|----|----|----|
                        0.0  0.2  0.4  0.6  0.8  1.0
                                         ↑
                               큰 overlap 영역 (0.6-0.9)
```

### 2.3 속도 문제

**현상**: 9,340 texts/sec로 목표 10,000/s에 6.6% 부족

**병목 지점 분석**:
```python
# 각 함수별 예상 비용
_extract_byte_features()     # O(n) - 바이트 순회
_extract_unicode_features()  # O(n) - 문자별 unicodedata.category() 호출 ← 가장 느림
_extract_token_features()    # O(n) - 문자별 속성 검사
_extract_pattern_features()  # O(n) - 엔트로피 계산, bigram 생성
```

**원인**:
1. `unicodedata.category()`: Python 인터프리터 오버헤드
2. 문자별 루프: `for char in text` 패턴이 4곳에서 반복
3. 임시 객체 생성: bigrams 리스트, char_counts 딕셔너리 등

### 2.4 벡터 크기 문제

**현상**: 512 bytes로 목표 256 bytes의 2배

**계산**:
```
현재: float32 × 128차원 = 4 bytes × 128 = 512 bytes
목표: 256 bytes

필요한 조치:
- 옵션 A: float32 → float16 (256 bytes, 정밀도 손실)
- 옵션 B: float32 → int8 (128 bytes, 양자화 필요)
- 옵션 C: 128 → 64 차원 (256 bytes, 정보 손실)
```

---

## 3. 알고리즘 설계 결함

### 3.1 피처 엔지니어링 문제

| 스케일 | 피처 | 문제점 |
|--------|------|--------|
| Byte-level | UTF-8 바이트 분포 | ASCII 집중, 다른 언어 구분 약함 |
| Unicode-level | 카테고리 분포 | 7개 카테고리로 너무 coarse |
| Token-level | 토큰 통계 | 의미적 구조 무시 |
| Pattern-level | 구조 지표 | Binary feature가 적음 (0/1만) |

### 3.2 Random Projection의 한계

```python
# 현재 구현
byte_projection = np.random.randn(256, 32)  # 학습되지 않은 랜덤 매트릭스
unicode_projection = np.random.randn(32, 32)  # 데이터 분포 무시

# 문제: 랜덤 투영이 텍스트 구조의 차이를 보존하지 않음
```

### 3.3 L2 정규화의 부작용

```python
# 모든 벡터가 단위 구(unit sphere) 위에 위치
vec = vec / np.linalg.norm(vec)

# 결과: 벡터 간 최대 거리가 2로 제한됨
# 구조적으로 매우 다른 텍스트도 가까이 위치할 수 있음
```

---

## 4. 베이스라인 대비 분석

### TF-IDF SVD가 Mean Neg Sim에서 우수한 이유

```
TF-IDF SVD:
- Character n-gram TF-IDF 사용
- IDF가 공통 패턴의 가중치를 자동으로 낮춤
- SVD가 discriminative한 방향을 학습

Multiscale:
- 수동으로 설계된 피처
- 공통 패턴에도 동일한 가중치
- 랜덤 투영으로 최적화되지 않음
```

### Random Projection이 빠른 이유

```
Random Projection:
- 단순 바이트 히스토그램 → 매트릭스 곱
- O(n + 256×128) 연산

Multiscale:
- 4개 스케일별 복잡한 피처 추출
- O(n × (byte + unicode + token + pattern))
- unicodedata.category() 호출 오버헤드
```

---

## 5. 결론

### 실패의 근본 원인

1. **구별력 부족**: 현재 피처가 텍스트의 구조적 차이를 충분히 포착하지 못함
2. **공통 특성 과대평가**: 모든 텍스트가 공유하는 특성이 벡터를 지배
3. **최적화 부재**: 랜덤 투영이 데이터 분포를 고려하지 않음
4. **속도-품질 트레이드오프 실패**: 복잡한 피처가 속도를 저하시키면서도 품질 향상에 실패

### 개선이 필요한 핵심 영역

1. **Discriminative Features**: 구조 타입별로 다른 값을 가지는 피처 필요
2. **Sparse/Binary Encoding**: Dense vector 대신 sparse representation 고려
3. **Data-driven Optimization**: 랜덤 대신 학습 기반 투영
4. **Efficient Implementation**: Numba JIT 또는 Cython 활용
