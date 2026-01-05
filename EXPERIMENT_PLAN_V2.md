# 실험 계획 V2: 구조 해싱 알고리즘 개선

## 1. 목표 재정의

### 달성해야 할 기준 (미달성 항목 중심)

| 우선순위 | 기준 | 현재 | 목표 | 개선폭 |
|----------|------|------|------|--------|
| **P0** | Mean Neg Sim | 0.643 | ≤ 0.35 | -45% |
| **P0** | Separation | 1.21 | ≥ 2.5 | +107% |
| **P1** | Best F1 | 0.8775 | ≥ 0.88 | +0.3% |
| **P1** | Encoding Speed | 9,340/s | ≥ 10,000/s | +7% |
| **P2** | Vector Bytes | 512 | ≤ 256 | -50% |

---

## 2. 실험 시리즈 설계

### Phase 1: 핵심 문제 해결 (Mean Neg Sim, Separation)

#### Experiment 1.1: Structure-Aware N-gram Hashing

**가설**: Character n-gram의 해시 기반 피처가 구조적 차이를 더 잘 포착할 것

**설계**:
```python
class StructureNgramEncoder(BaseEncoder):
    """
    1. Character 2,3,4-gram 추출
    2. 각 n-gram을 해시하여 sparse vector 생성
    3. TF-IDF 가중치 적용 (without sklearn)
    4. Random projection으로 차원 축소
    """

    def __init__(self, dim=128, n_grams=[2,3,4], vocab_size=8192):
        self.n_grams = n_grams
        self.vocab_size = vocab_size
        self.projection = np.random.randn(vocab_size, dim) / np.sqrt(dim)
        self.idf = None  # 학습 시 계산

    def fit(self, texts):
        """IDF 계산 (선택적 학습)"""
        doc_freq = np.zeros(self.vocab_size)
        for text in texts:
            seen = set()
            for n in self.n_grams:
                for i in range(len(text) - n + 1):
                    h = hash(text[i:i+n]) % self.vocab_size
                    if h not in seen:
                        doc_freq[h] += 1
                        seen.add(h)
        self.idf = np.log(len(texts) / (doc_freq + 1))

    def encode(self, text):
        tf = np.zeros(self.vocab_size, dtype=np.float32)
        for n in self.n_grams:
            for i in range(len(text) - n + 1):
                h = hash(text[i:i+n]) % self.vocab_size
                tf[h] += 1
        if self.idf is not None:
            tf *= self.idf
        vec = self.projection.T @ tf
        return vec / (np.linalg.norm(vec) + 1e-10)
```

**예상 개선**:
- Mean Neg Sim: 0.64 → 0.25 (TF-IDF 효과)
- 속도: ~15,000/s (단순 해시 연산)

**평가 지표**: AUC-ROC, Mean Neg Sim, Encoding Speed

---

#### Experiment 1.2: Structure Type Detection + Conditional Encoding

**가설**: 텍스트 구조를 먼저 분류한 후 타입별 인코딩이 분리도를 높일 것

**설계**:
```python
class StructureTypeEncoder(BaseEncoder):
    """
    1. 정규식으로 구조 타입 감지 (URL, Email, JSON, XML, ...)
    2. 타입별 다른 피처 추출
    3. 타입 ID를 벡터에 포함하여 다른 타입 간 유사도 감소
    """

    PATTERNS = {
        'url': r'^https?://|^ftp://',
        'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
        'json': r'^\s*[\{\[]',
        'xml': r'^\s*<\w+',
        'filepath': r'^[A-Z]:\\|^/[a-z]',
        'phone': r'^\+?\d[\d\-\s\(\)]+$',
        'hash': r'^[a-f0-9\-]{32,}$',
        'base64': r'^[A-Za-z0-9+/]+=*$',
        'code': r'(function|def|class|const|let|var|if|for)',
    }

    def detect_type(self, text):
        for type_name, pattern in self.PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return type_name
        return 'text'  # default

    def encode(self, text):
        type_id = self.detect_type(text)
        type_vec = self._one_hot(type_id)  # 16 dims
        content_vec = self._encode_content(text)  # 112 dims
        return normalize(concat(type_vec, content_vec))
```

**예상 개선**:
- 다른 타입의 텍스트는 type_vec이 다르므로 유사도 ↓
- Separation: 1.21 → 2.8+ (타입 분리 효과)

---

#### Experiment 1.3: Contrastive Feature Weighting

**가설**: Positive/Negative 분포를 잘 분리하는 피처에 가중치를 주면 separation 향상

**설계**:
```python
class ContrastiveEncoder(BaseEncoder):
    """
    1. 기존 MultiScale 피처 추출
    2. 학습 데이터로 각 피처의 discriminative power 계산
    3. Fisher's ratio가 높은 피처에 가중치 부여
    """

    def fit(self, train_pairs):
        # 각 피처의 Fisher's ratio 계산
        # F = (mean_pos - mean_neg)^2 / (var_pos + var_neg)
        pos_features, neg_features = [], []
        for pair in train_pairs:
            v1, v2 = self.encode_raw(pair.text1), self.encode_raw(pair.text2)
            sim_features = v1 * v2  # element-wise similarity
            if pair.label == 1:
                pos_features.append(sim_features)
            else:
                neg_features.append(sim_features)

        pos_mean, pos_var = np.mean(pos_features, 0), np.var(pos_features, 0)
        neg_mean, neg_var = np.mean(neg_features, 0), np.var(neg_features, 0)

        fisher_ratio = (pos_mean - neg_mean)**2 / (pos_var + neg_var + 1e-10)
        self.weights = np.sqrt(fisher_ratio)  # Feature weights
```

**예상 개선**:
- Discriminative 피처 강조로 separation 향상
- 학습 의존적이지만 효과적

---

### Phase 2: 효율성 최적화 (Speed, Vector Size)

#### Experiment 2.1: Numba JIT Acceleration

**가설**: Python 루프를 JIT 컴파일하면 2-5배 속도 향상

**설계**:
```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def extract_byte_features_fast(text_bytes, projection):
    byte_hist = np.zeros(256, dtype=np.float32)
    for b in text_bytes:
        byte_hist[b] += 1
    byte_hist /= len(text_bytes)
    return projection.T @ byte_hist

@jit(nopython=True)
def extract_ngram_hashes_fast(text_bytes, n, vocab_size):
    tf = np.zeros(vocab_size, dtype=np.float32)
    for i in range(len(text_bytes) - n + 1):
        h = 0
        for j in range(n):
            h = h * 31 + text_bytes[i+j]
        tf[h % vocab_size] += 1
    return tf
```

**예상 개선**:
- 속도: 9,340/s → 20,000+/s
- 특히 문자 순회 루프에서 큰 효과

---

#### Experiment 2.2: Int8 Quantization

**가설**: Float32 → Int8 양자화로 벡터 크기 75% 감소, 품질 손실 최소화

**설계**:
```python
class QuantizedEncoder(BaseEncoder):
    """
    1. Float32 벡터 생성
    2. Min-Max 스케일링 후 Int8 변환
    3. 비교 시 Int8 cosine similarity
    """

    def encode(self, text):
        vec_f32 = self.base_encoder.encode(text)

        # Quantize to int8 [-128, 127]
        v_min, v_max = vec_f32.min(), vec_f32.max()
        scale = 255.0 / (v_max - v_min + 1e-10)
        vec_i8 = ((vec_f32 - v_min) * scale - 128).astype(np.int8)

        return vec_i8

    def similarity(self, v1, v2):
        # Int8 dot product
        return np.dot(v1.astype(np.int32), v2.astype(np.int32)) / (128 * 128 * len(v1))
```

**예상 개선**:
- 벡터 크기: 512 bytes → 128 bytes (목표 256 달성)
- 품질 손실: AUC-ROC -0.01 ~ -0.02 (허용 범위)

---

#### Experiment 2.3: Dimensionality Reduction to 64

**가설**: 128 → 64 차원으로 줄여도 핵심 정보 유지 가능

**설계**:
```python
class CompactEncoder(BaseEncoder):
    """
    1. 각 스케일 16차원 (총 64차원)
    2. PCA로 가장 중요한 방향만 유지
    3. Float32 × 64 = 256 bytes (목표 달성)
    """

    def __init__(self, dim=64):
        self.dim = dim
        self.sub_dim = dim // 4  # 16 per scale
```

---

### Phase 3: 하이브리드 접근

#### Experiment 3.1: Ensemble: Fast Filter + Precise Scorer

**가설**: 2단계 접근으로 속도와 품질 모두 확보

**설계**:
```python
class HybridEncoder:
    """
    Stage 1: Fast hash (SimHash-like) for candidate filtering
    Stage 2: Precise encoder for final scoring

    대량 데이터에서:
    1. Fast hash로 후보 1000개 추출 (ms)
    2. Precise encoder로 상위 100개 재정렬 (ms)
    """

    def __init__(self):
        self.fast_encoder = SimHashEncoder(bits=256)  # 빠른 필터
        self.precise_encoder = StructureNgramEncoder()  # 정밀 스코어러
```

---

#### Experiment 3.2: Structure-Aware Binary Signature

**가설**: Dense vector 대신 binary signature로 속도와 메모리 동시 최적화

**설계**:
```python
class BinaryStructureEncoder(BaseEncoder):
    """
    1. 구조 타입 감지 → 4 bits
    2. N-gram hash signatures → 124 bits
    3. Total: 128 bits = 16 bytes

    비교: Hamming distance (XOR + popcount)
    """

    def encode(self, text):
        type_bits = self.encode_type(text)  # 4 bits
        ngram_bits = self.simhash_ngrams(text)  # 124 bits
        return np.packbits(concat(type_bits, ngram_bits))  # 16 bytes

    def similarity(self, v1, v2):
        hamming = np.unpackbits(v1 ^ v2).sum()
        return 1.0 - hamming / (128)
```

**예상 개선**:
- 벡터 크기: 512 bytes → 16 bytes (96% 감소!)
- 속도: XOR + popcount는 매우 빠름 (100,000+/s)
- 품질: AUC-ROC ~0.90 예상 (약간 감소)

---

## 3. 실험 우선순위 및 일정

### 우선순위 매트릭스

| 실험 | 예상 영향 | 구현 난이도 | 우선순위 |
|------|----------|------------|----------|
| 1.1 N-gram Hashing | 높음 | 중간 | **1순위** |
| 1.2 Structure Type Detection | 높음 | 낮음 | **2순위** |
| 2.1 Numba JIT | 중간 | 낮음 | **3순위** |
| 2.2 Int8 Quantization | 중간 | 낮음 | **4순위** |
| 1.3 Contrastive Weighting | 중간 | 높음 | 5순위 |
| 3.2 Binary Signature | 높음 | 중간 | 6순위 |

### 권장 실험 순서

```
Step 1: Experiment 1.2 (Structure Type Detection)
        └─ 빠른 구현으로 separation 개선 확인

Step 2: Experiment 1.1 (N-gram Hashing)
        └─ Mean Neg Sim 개선의 핵심

Step 3: Experiment 2.1 (Numba JIT)
        └─ 속도 목표 달성

Step 4: Experiment 2.2 (Int8 Quantization)
        └─ 벡터 크기 목표 달성

Step 5: 통합 및 최종 평가
        └─ 최적 조합 선정
```

---

## 4. 성공 기준

### 최소 성공 (Minimum Viable)
- [ ] Mean Neg Sim ≤ 0.40 (현재 0.64에서 37.5% 개선)
- [ ] Separation ≥ 2.0 (현재 1.21에서 65% 개선)
- [ ] 기존 AUC-ROC 유지 (≥ 0.92)

### 목표 성공 (Target)
- [ ] 7개 기준 중 5개 이상 달성

### 완전 성공 (Full Success)
- [ ] 7개 기준 모두 달성

---

## 5. 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| N-gram 해싱이 효과 없음 | 높음 | TF-IDF SVD 참고하여 IDF 가중치 조정 |
| Numba 호환성 문제 | 중간 | Cython 또는 순수 NumPy 벡터화로 대체 |
| 양자화 품질 손실 큼 | 중간 | 학습 기반 양자화 또는 차원 유지 |
| 구조 감지 정확도 낮음 | 낮음 | 패턴 확장 및 fallback 로직 |

---

## 6. 다음 단계

1. **즉시**: Experiment 1.2 (Structure Type Detection) 구현
2. **단기**: Experiment 1.1 (N-gram Hashing) 구현
3. **중기**: Phase 2 효율성 최적화 진행
4. **평가**: 각 실험 후 전체 메트릭 재평가

---

*이 실험 계획은 패배 요인 분석(FAILURE_ANALYSIS.md)에 기반하여 작성되었습니다.*
