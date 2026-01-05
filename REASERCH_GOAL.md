# Text Structure Hashing Algorithm Research Protocol

## Mission Statement

당신은 **경량 텍스트 구조 유사성 알고리즘**을 연구하는 ML 리서처입니다.

**목표**: 의미론적 임베딩(BERT, OpenAI) 없이, 텍스트의 **구조적 특성**만으로 유사성을 판단하는 효율적인 해싱 알고리즘을 개발합니다.

**제약 조건**:
- 메모리: ≤ 256 bytes/text
- 인코딩 속도: ≥ 10,000 texts/sec (single core)
- 벡터 차원: 64-256 (벡터 DB 호환)
- 외부 의존성: 최소화 (no ML frameworks for inference)

**성공 기준**: 
- 구조적으로 유사한 텍스트 쌍에서 cosine similarity ≥ 0.8
- 구조적으로 다른 텍스트 쌍에서 cosine similarity ≤ 0.4
- OpenAI text-embedding-3-small 대비 1000x 빠른 인코딩

---

## 1. Problem Definition

### 1.1 Task Description

입력: 임의의 UTF-8 텍스트 문자열 (1자 ~ 100,000자)
출력: 고정 차원 실수 벡터 `float32[D]` where D ∈ {64, 128, 256}

**"구조적 유사성"의 정의**:
- 동일 카테고리: URL↔URL, 이메일↔이메일, 한글문장↔한글문장
- 동일 포맷: JSON↔JSON, 날짜↔날짜
- 동일 문자 구성: 영숫자혼합↔영숫자혼합
- 동일 길이 클래스: 단문↔단문, 장문↔장문

**"의미적 유사성"과의 구분**:
- "I love dogs" ↔ "I adore puppies" → 의미적으로 유사하지만 구조적으로도 유사 (둘 다 짧은 영문)
- "I love dogs" ↔ "나는 개를 좋아해" → 의미적으로 유사하지만 구조적으로 다름 (다른 스크립트)
- "https://a.com" ↔ "https://b.com" → 의미적으로 다르지만 구조적으로 동일 (URL)

### 1.2 Use Cases

1. **클립보드 매니저**: 복사한 텍스트 자동 분류
2. **중복 검출**: 구조적으로 유사한 항목 그룹화
3. **데이터 품질**: 이상 패턴 탐지
4. **검색 필터링**: 특정 구조 타입만 검색

---

## 2. Dataset Specification

### 2.1 Synthetic Dataset Generation

다음 카테고리별로 합성 데이터를 생성하세요:

```python
CATEGORIES = {
    # 구조화된 식별자
    "url": {
        "patterns": [
            "https://{domain}.{tld}/{path}",
            "http://{domain}.{tld}?{query}",
            "ftp://{user}@{domain}/{path}"
        ],
        "count": 1000
    },
    "email": {
        "patterns": [
            "{user}@{domain}.{tld}",
            "{first}.{last}@{company}.{tld}"
        ],
        "count": 1000
    },
    "phone": {
        "patterns": [
            "010-{4d}-{4d}",
            "+82-10-{4d}-{4d}",
            "({3d}) {3d}-{4d}"
        ],
        "count": 500
    },
    "date": {
        "patterns": [
            "{Y}-{M}-{D}",
            "{D}/{M}/{Y}",
            "{M}/{D}/{Y}",
            "{Y}년 {M}월 {D}일"
        ],
        "count": 500
    },
    "ipv4": {
        "patterns": ["xxx.xxx.xxx.xxx"],
        "count": 300
    },
    
    # 구조화된 데이터
    "json": {
        "depth_range": [1, 5],
        "key_count_range": [1, 20],
        "count": 1000
    },
    "xml": {
        "depth_range": [1, 5],
        "tag_count_range": [1, 20],
        "count": 500
    },
    "csv_row": {
        "column_range": [2, 10],
        "count": 500
    },
    
    # 자연어 텍스트
    "korean_sentence": {
        "length_range": [10, 200],
        "source": "synthetic_or_corpus",
        "count": 2000
    },
    "english_sentence": {
        "length_range": [10, 200],
        "source": "synthetic_or_corpus",
        "count": 2000
    },
    "chinese_sentence": {
        "length_range": [5, 100],
        "count": 500
    },
    "japanese_sentence": {
        "length_range": [5, 100],
        "count": 500
    },
    "mixed_language": {
        "scripts": ["latin", "hangul", "cjk"],
        "count": 500
    },
    
    # 코드 스니펫
    "code_javascript": {"count": 500},
    "code_python": {"count": 500},
    "code_sql": {"count": 300},
    
    # 특수 형식
    "hash_string": {
        "types": ["md5", "sha256", "uuid"],
        "count": 500
    },
    "base64": {"count": 300},
    "filepath": {
        "os": ["unix", "windows"],
        "count": 500
    },
    
    # 숫자 데이터
    "number_integer": {"range": [0, 1e12], "count": 300},
    "number_decimal": {"precision_range": [1, 10], "count": 300},
    "number_formatted": {"patterns": ["#,###", "#.##%"], "count": 300},
    
    # 기타
    "single_word": {"length_range": [1, 20], "count": 500},
    "random_string": {"length_range": [10, 100], "count": 500}
}

TOTAL_SAMPLES = ~15,000
```

### 2.2 Ground Truth Pairs

**Positive Pairs (같은 구조)**: 동일 카테고리 내 랜덤 샘플링
**Negative Pairs (다른 구조)**: 다른 카테고리 간 랜덤 샘플링

```python
def generate_pairs(dataset, n_positive=5000, n_negative=5000):
    positive_pairs = []
    negative_pairs = []
    
    # Positive: 같은 카테고리
    for category, samples in dataset.items():
        for _ in range(n_positive // len(dataset)):
            i, j = random.sample(range(len(samples)), 2)
            positive_pairs.append((samples[i], samples[j], 1.0))
    
    # Negative: 다른 카테고리
    categories = list(dataset.keys())
    for _ in range(n_negative):
        cat1, cat2 = random.sample(categories, 2)
        s1 = random.choice(dataset[cat1])
        s2 = random.choice(dataset[cat2])
        negative_pairs.append((s1, s2, 0.0))
    
    return positive_pairs + negative_pairs
```

### 2.3 Difficulty Levels

**Easy**: 명확히 다른 카테고리 (URL vs 한글문장)
**Medium**: 유사 형태 (JSON vs XML, 영문 vs 한글)
**Hard**: 미묘한 차이 (짧은 JSON vs 짧은 영문, 숫자열 vs 날짜)

---

## 3. Evaluation Metrics

### 3.1 Primary Metrics

```python
def evaluate(encoder, test_pairs):
    """
    Args:
        encoder: function(text) -> np.ndarray[float32]
        test_pairs: List[(text1, text2, label)]  # label: 1.0=similar, 0.0=dissimilar
    
    Returns:
        dict of metrics
    """
    predictions = []
    labels = []
    
    for text1, text2, label in test_pairs:
        vec1 = encoder(text1)
        vec2 = encoder(text2)
        sim = cosine_similarity(vec1, vec2)
        predictions.append(sim)
        labels.append(label)
    
    # 1. AUC-ROC: 분류 성능
    auc = roc_auc_score(labels, predictions)
    
    # 2. Precision@K: 상위 K개 정확도
    sorted_pairs = sorted(zip(predictions, labels), reverse=True)
    p_at_100 = sum(l for _, l in sorted_pairs[:100]) / 100
    p_at_1000 = sum(l for _, l in sorted_pairs[:1000]) / 1000
    
    # 3. Separation: positive와 negative 분포 분리도
    pos_sims = [p for p, l in zip(predictions, labels) if l == 1.0]
    neg_sims = [p for p, l in zip(predictions, labels) if l == 0.0]
    separation = (np.mean(pos_sims) - np.mean(neg_sims)) / (np.std(pos_sims) + np.std(neg_sims))
    
    # 4. Optimal Threshold F1
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = [1 if p >= threshold else 0 for p in predictions]
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return {
        "auc_roc": auc,
        "precision_at_100": p_at_100,
        "precision_at_1000": p_at_1000,
        "separation": separation,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "mean_pos_sim": np.mean(pos_sims),
        "mean_neg_sim": np.mean(neg_sims)
    }
```

### 3.2 Efficiency Metrics

```python
def benchmark_efficiency(encoder, test_texts, n_iterations=3):
    """
    Args:
        encoder: function(text) -> np.ndarray
        test_texts: List[str] (다양한 길이 포함)
    """
    import time
    import sys
    
    # 1. Encoding Speed
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        for text in test_texts:
            _ = encoder(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    texts_per_sec = len(test_texts) / avg_time
    
    # 2. Memory per Vector
    sample_vec = encoder(test_texts[0])
    vec_bytes = sample_vec.nbytes
    
    # 3. Comparison Speed
    vecs = [encoder(t) for t in test_texts[:1000]]
    start = time.perf_counter()
    for i in range(len(vecs)):
        for j in range(i+1, min(i+100, len(vecs))):
            _ = np.dot(vecs[i], vecs[j])
    comparison_time = time.perf_counter() - start
    comparisons_per_sec = (1000 * 99 / 2) / comparison_time
    
    return {
        "encoding_speed": texts_per_sec,  # texts/sec
        "vector_bytes": vec_bytes,
        "vector_dimensions": len(sample_vec),
        "comparison_speed": comparisons_per_sec,  # comparisons/sec
        "total_encoding_time_ms": avg_time * 1000
    }
```

### 3.3 Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| AUC-ROC | 0.85 | 0.92 | 0.96 |
| Separation | 1.5 | 2.5 | 3.5 |
| Best F1 | 0.80 | 0.88 | 0.93 |
| Encoding Speed | 5,000/s | 10,000/s | 50,000/s |
| Vector Bytes | ≤1024 | ≤256 | ≤64 |
| Mean Positive Sim | ≥0.75 | ≥0.85 | ≥0.90 |
| Mean Negative Sim | ≤0.45 | ≤0.35 | ≤0.25 |

---

## 4. Algorithm Candidates

### 4.1 Baseline Algorithms

구현하고 비교해야 할 기준 알고리즘:

**B1. Random Projection (RP)**
```python
class RandomProjection:
    def __init__(self, input_dim, output_dim, seed=42):
        np.random.seed(seed)
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(output_dim)
    
    def encode(self, text):
        # 문자 빈도 벡터 → 랜덤 투영
        char_freq = np.zeros(65536)  # Unicode BMP
        for c in text:
            char_freq[ord(c) % 65536] += 1
        char_freq /= (len(text) + 1)
        return self.projection.T @ char_freq[:self.projection.shape[0]]
```

**B2. SimHash**
```python
def simhash(text, dim=128):
    """
    텍스트의 n-gram에 대한 SimHash
    """
    v = np.zeros(dim)
    ngrams = [text[i:i+3] for i in range(len(text)-2)]
    
    for gram in ngrams:
        h = hash(gram)
        for i in range(dim):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1
    
    return (v > 0).astype(np.float32)
```

**B3. MinHash**
```python
def minhash(text, num_hashes=128):
    """
    문자 n-gram에 대한 MinHash
    """
    ngrams = set(text[i:i+3] for i in range(len(text)-2))
    signature = np.full(num_hashes, np.inf)
    
    for i in range(num_hashes):
        for gram in ngrams:
            h = hash((gram, i)) % (2**32)
            signature[i] = min(signature[i], h)
    
    # 정규화
    return (signature / (2**32)).astype(np.float32)
```

**B4. Character N-gram TF-IDF (Sparse → Dense)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class NgramTFIDF:
    def __init__(self, dim=128):
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=10000)
        self.svd = TruncatedSVD(n_components=dim)
        self.fitted = False
    
    def fit(self, texts):
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self.fitted = True
    
    def encode(self, text):
        tfidf = self.vectorizer.transform([text])
        return self.svd.transform(tfidf)[0].astype(np.float32)
```

### 4.2 Proposed Algorithms (실험 대상)

**P1. Multi-Scale Character Statistics**
```python
def multiscale_stats(text, dim=128):
    """
    여러 스케일의 문자 통계 결합
    - Byte-level: ASCII/Unicode 분포
    - Character-level: 스크립트 분포
    - Token-level: 공백 분리 토큰 통계
    - Pattern-level: 반복/엔트로피
    """
    vec = np.zeros(dim)
    
    # Scale 1: Byte distribution (32-dim)
    byte_hist = np.zeros(256)
    for b in text.encode('utf-8', errors='ignore'):
        byte_hist[b] += 1
    vec[:32] = pca_reduce(byte_hist, 32)  # 또는 random projection
    
    # Scale 2: Unicode category (32-dim)
    # ... 구현
    
    # Scale 3: Token statistics (32-dim)
    # ... 구현
    
    # Scale 4: Structural patterns (32-dim)
    # ... 구현
    
    return vec / (np.linalg.norm(vec) + 1e-8)
```

**P2. Learned Feature Weights (Siamese-inspired)**
```python
class LearnedWeights:
    """
    고정 피처 추출 + 학습된 가중치 행렬
    - 피처: 하드코딩이지만 가중치는 데이터에서 학습
    - Contrastive loss로 최적화
    """
    def __init__(self, feature_dim, output_dim):
        self.W = np.random.randn(feature_dim, output_dim) * 0.01
    
    def extract_features(self, text):
        # 고정된 피처 추출 (휴리스틱)
        return np.array([
            len(text),
            text.count(' '),
            sum(c.isalpha() for c in text),
            sum(c.isdigit() for c in text),
            # ... 더 많은 피처
        ])
    
    def encode(self, text):
        features = self.extract_features(text)
        return self.W.T @ features
    
    def train(self, positive_pairs, negative_pairs, epochs=100, lr=0.01):
        for epoch in range(epochs):
            loss = 0
            for t1, t2 in positive_pairs:
                v1, v2 = self.encode(t1), self.encode(t2)
                # Pull together
                grad = 2 * (v1 - v2)
                # ... gradient update
            
            for t1, t2 in negative_pairs:
                v1, v2 = self.encode(t1), self.encode(t2)
                # Push apart (with margin)
                # ... gradient update
```

**P3. Hierarchical Locality Sensitive Hashing**
```python
class HierarchicalLSH:
    """
    계층적 LSH: 
    1. Coarse level: 길이 + 스크립트로 버킷팅
    2. Fine level: 문자 n-gram LSH
    """
    def __init__(self, coarse_bits=16, fine_bits=112):
        self.coarse_bits = coarse_bits
        self.fine_bits = fine_bits
    
    def encode(self, text):
        coarse = self.coarse_hash(text)  # 길이, 주요 스크립트
        fine = self.fine_hash(text)      # n-gram MinHash
        return np.concatenate([coarse, fine])
    
    def coarse_hash(self, text):
        # 길이 버킷 (8 bits)
        len_bucket = min(int(np.log2(len(text) + 1)), 15)
        
        # 스크립트 비율 (8 bits)
        # ... 구현
        
        return np.array([...], dtype=np.float32)
    
    def fine_hash(self, text):
        # Character bigram MinHash
        # ... 구현
        pass
```

**P4. Autoencoder-Compressed Features**
```python
class CompressedFeatures:
    """
    큰 피처 벡터 → 작은 오토인코더로 압축
    - 학습 시에만 오토인코더 사용
    - 추론 시 인코더 부분만 사용 (작은 MLP)
    """
    def __init__(self, input_dim=1000, latent_dim=128):
        # 매우 작은 인코더: 1000 → 256 → 128
        self.encoder_w1 = np.random.randn(input_dim, 256) * 0.01
        self.encoder_w2 = np.random.randn(256, latent_dim) * 0.01
    
    def extract_raw_features(self, text):
        # 큰 피처 벡터 (예: 1000차원)
        # - 문자 빈도 (256)
        # - bigram 빈도 (512, 해시로 압축)
        # - 통계 피처 (50)
        # - 패턴 플래그 (32)
        # ...
        pass
    
    def encode(self, text):
        raw = self.extract_raw_features(text)
        h = np.maximum(0, raw @ self.encoder_w1)  # ReLU
        return h @ self.encoder_w2
```

---

## 5. Experiment Protocol

### 5.1 Phase 1: Data Generation (1-2 hours)

```bash
# 1. 합성 데이터 생성
python generate_dataset.py --output data/synthetic.jsonl --seed 42

# 2. 페어 생성
python generate_pairs.py --input data/synthetic.jsonl \
    --output data/pairs.jsonl \
    --n_positive 5000 --n_negative 5000

# 3. Train/Val/Test 분할 (60/20/20)
python split_dataset.py --input data/pairs.jsonl \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --test data/test.jsonl
```

### 5.2 Phase 2: Baseline Implementation (2-3 hours)

각 베이스라인 알고리즘 구현:
1. Random Projection
2. SimHash  
3. MinHash
4. N-gram TF-IDF + SVD

```bash
python implement_baselines.py
python evaluate.py --model baseline_rp --data data/test.jsonl
python evaluate.py --model baseline_simhash --data data/test.jsonl
python evaluate.py --model baseline_minhash --data data/test.jsonl
python evaluate.py --model baseline_tfidf --data data/test.jsonl
```

### 5.3 Phase 3: Proposed Algorithm Development (4-6 hours)

반복적 개선 사이클:

```
For each proposed algorithm P1-P4:
    1. 초기 구현
    2. Validation set에서 평가
    3. Error analysis: 어떤 카테고리 쌍에서 실패하는가?
    4. 피처/하이퍼파라미터 조정
    5. 반복 (최대 5회)
    6. Test set에서 최종 평가
```

### 5.4 Phase 4: Ablation Study (2-3 hours)

각 알고리즘의 구성 요소 기여도 분석:

```python
def ablation_study(encoder, test_pairs):
    """
    피처 그룹별 제거 후 성능 측정
    """
    results = {}
    
    # Full model
    results['full'] = evaluate(encoder, test_pairs)
    
    # 각 피처 그룹 제거
    for feature_group in ['byte_stats', 'unicode_script', 'token_stats', 'ngram_hash']:
        encoder_ablated = encoder.without(feature_group)
        results[f'without_{feature_group}'] = evaluate(encoder_ablated, test_pairs)
    
    return results
```

### 5.5 Phase 5: Efficiency Optimization (2-3 hours)

```python
# 1. Profiling
python -m cProfile -o profile.stats evaluate.py --model best_model

# 2. 병목 식별
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)

# 3. 최적화 적용
# - NumPy 벡터화
# - Numba JIT
# - 캐싱
# - 차원 축소

# 4. 최적화 후 재평가
python benchmark_efficiency.py --model optimized_model
```

---

## 6. Reporting Template

실험 완료 후 다음 형식으로 결과를 보고하세요:

```markdown
# Experiment Results: [Algorithm Name]

## Configuration
- Vector Dimension: [D]
- Key Hyperparameters: [list]
- Training Data Size: [N]

## Quality Metrics (Test Set)

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| AUC-ROC | 0.XX | +X.X% |
| Best F1 | 0.XX | +X.X% |
| Separation | X.XX | +X.X% |
| Mean Positive Sim | 0.XX | - |
| Mean Negative Sim | 0.XX | - |

## Efficiency Metrics

| Metric | Value | vs OpenAI Embedding |
|--------|-------|---------------------|
| Encoding Speed | X,XXX/s | XXXx faster |
| Vector Bytes | XXX | XXx smaller |
| Comparison Speed | X,XXX,XXX/s | - |

## Per-Category Analysis

| Category Pair | Accuracy | Common Errors |
|---------------|----------|---------------|
| URL ↔ URL | XX% | - |
| URL ↔ Email | XX% | [error description] |
| ... | ... | ... |

## Ablation Results

| Removed Component | AUC Change | Notes |
|-------------------|------------|-------|
| [component] | -X.X% | [insight] |

## Key Insights
1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

## Recommended Next Steps
1. [Recommendation 1]
2. [Recommendation 2]
```

---

## 7. Implementation Guidelines

### 7.1 Code Structure

```
text-structure-hash/
├── data/
│   ├── synthetic.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── src/
│   ├── __init__.py
│   ├── encoders/
│   │   ├── base.py          # Abstract base class
│   │   ├── random_projection.py
│   │   ├── simhash.py
│   │   ├── minhash.py
│   │   ├── tfidf_svd.py
│   │   └── proposed/
│   │       ├── multiscale.py
│   │       ├── learned_weights.py
│   │       ├── hierarchical_lsh.py
│   │       └── compressed_features.py
│   ├── data/
│   │   ├── generator.py
│   │   └── loader.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── benchmark.py
│   └── utils/
│       ├── unicode.py
│       └── hashing.py
├── scripts/
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── ablation.py
├── notebooks/
│   └── analysis.ipynb
├── results/
│   └── [experiment_name]/
│       ├── metrics.json
│       └── report.md
└── requirements.txt
```

### 7.2 Base Encoder Interface

```python
# src/encoders/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseEncoder(ABC):
    """모든 인코더가 구현해야 할 인터페이스"""
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """출력 벡터 차원"""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 벡터로 인코딩
        
        Args:
            text: UTF-8 문자열
            
        Returns:
            L2-normalized float32 벡터
        """
        pass
    
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """배치 인코딩 (기본: 순차 처리)"""
        return np.stack([self.encode(t) for t in texts])
    
    def similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 유사도"""
        v1, v2 = self.encode(text1), self.encode(text2)
        return float(np.dot(v1, v2))
```

### 7.3 Quality Requirements

- 모든 코드에 docstring 포함
- Type hints 사용
- 단위 테스트 작성 (pytest)
- 재현성을 위한 시드 고정

---

## 8. Iteration Guidelines

### 8.1 실험 우선순위

1. **먼저**: 베이스라인 구현 및 평가 (기준선 확보)
2. **그 다음**: 가장 단순한 제안 알고리즘 (P1: Multi-Scale Stats)
3. **이후**: 점진적으로 복잡한 알고리즘 시도
4. **마지막**: 최고 성능 알고리즘 최적화

### 8.2 조기 종료 조건

- 베이스라인 대비 개선 없음 (3회 반복 후)
- 효율성 기준 미달 (인코딩 속도 < 1,000/s)
- 메모리 기준 초과 (벡터 > 1KB)

### 8.3 성공 시 확장

Target 기준 달성 시:
1. 더 어려운 테스트 케이스 추가
2. 실제 클립보드 데이터로 검증
3. 벡터 DB 통합 테스트 (pgvector, FAISS)
4. 양자화 실험 (float32 → int8 → binary)

---

## 9. Reference Resources

### 9.1 관련 논문

1. **SimHash**: Charikar, M. (2002). "Similarity estimation techniques from rounding algorithms"
2. **MinHash**: Broder, A. (1997). "On the resemblance and containment of documents"
3. **LSH Survey**: Wang et al. (2014). "Hashing for Similarity Search: A Survey"
4. **TLSH**: Oliver et al. (2013). "TLSH - A Locality Sensitive Hash"

### 9.2 구현 참고

- [datasketch](https://github.com/ekzhu/datasketch): MinHash, LSH 구현
- [ssdeep](https://ssdeep-project.github.io/ssdeep/): Fuzzy hashing
- [FAISS](https://github.com/facebookresearch/faiss): 벡터 인덱싱

---

## 10. Checklist

실험 완료 전 확인:

- [ ] 합성 데이터셋 생성 완료 (≥15,000 samples)
- [ ] Positive/Negative 페어 생성 완료 (≥10,000 pairs)
- [ ] 4개 베이스라인 구현 및 평가 완료
- [ ] 최소 2개 제안 알고리즘 구현 및 평가 완료
- [ ] Ablation study 완료
- [ ] 효율성 벤치마크 완료
- [ ] 결과 보고서 작성 완료
- [ ] 최고 성능 알고리즘 코드 정리 완료

---

**End of Protocol**
