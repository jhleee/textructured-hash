# Text Structure Hashing Algorithm

**경량 텍스트 구조 유사성 알고리즘 연구 프로젝트**

## 목표

의미론적 임베딩(BERT, OpenAI) 없이, 텍스트의 **구조적 특성**만으로 유사성을 판단하는 효율적인 해싱 알고리즘을 개발합니다.

## 핵심 제약 조건

- **메모리**: ≤ 256 bytes/text
- **인코딩 속도**: ≥ 10,000 texts/sec (single core)
- **벡터 차원**: 64-256 (벡터 DB 호환)
- **외부 의존성**: 최소화 (no ML frameworks for inference)

## 성공 기준

| 메트릭 | 최소 | 목표 | 도전 |
|--------|------|------|------|
| AUC-ROC | 0.85 | 0.92 | 0.96 |
| Separation | 1.5 | 2.5 | 3.5 |
| Best F1 | 0.80 | 0.88 | 0.93 |
| Encoding Speed | 5,000/s | 10,000/s | 50,000/s |
| Vector Bytes | ≤1024 | ≤256 | ≤64 |
| Mean Positive Sim | ≥0.75 | ≥0.85 | ≥0.90 |
| Mean Negative Sim | ≤0.45 | ≤0.35 | ≤0.25 |

**성능 목표**: OpenAI text-embedding-3-small 대비 **1000x 빠른** 인코딩

## 구조적 유사성의 정의

- **동일 카테고리**: URL↔URL, 이메일↔이메일, 한글문장↔한글문장
- **동일 포맷**: JSON↔JSON, 날짜↔날짜
- **동일 문자 구성**: 영숫자혼합↔영숫자혼합
- **동일 길이 클래스**: 단문↔단문, 장문↔장문

## 주요 사용 사례

1. **클립보드 매니저**: 복사한 텍스트 자동 분류
2. **중복 검출**: 구조적으로 유사한 항목 그룹화
3. **데이터 품질**: 이상 패턴 탐지
4. **검색 필터링**: 특정 구조 타입만 검색

## 프로젝트 구조

```
text-structure-hash/
├── data/                    # 데이터셋
│   ├── synthetic.jsonl      # 합성 데이터
│   ├── train.jsonl          # 학습 세트
│   ├── val.jsonl            # 검증 세트
│   └── test.jsonl           # 테스트 세트
├── src/                     # 소스 코드
│   ├── encoders/            # 인코더 구현
│   ├── data/                # 데이터 생성 및 로딩
│   ├── evaluation/          # 평가 메트릭
│   └── utils/               # 유틸리티
├── scripts/                 # 실행 스크립트
├── notebooks/               # 분석 노트북
├── results/                 # 실험 결과
├── RESEARCH_PROTOCOL.md     # 상세 연구 프로토콜
└── requirements.txt         # 의존성
```

## 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 생성

```bash
python scripts/generate_data.py --output data/synthetic.jsonl --seed 42
python scripts/generate_pairs.py --input data/synthetic.jsonl --output data/pairs.jsonl
python scripts/split_dataset.py --input data/pairs.jsonl
```

### 3. 베이스라인 평가

```bash
python scripts/evaluate.py --model baseline_simhash --data data/test.jsonl
```

### 4. 제안 알고리즘 학습 및 평가

```bash
python scripts/train.py --model multiscale --config configs/multiscale.yaml
python scripts/evaluate.py --model multiscale --data data/test.jsonl
```

## 알고리즘 후보

### 베이스라인

- **B1**: Random Projection
- **B2**: SimHash
- **B3**: MinHash
- **B4**: Character N-gram TF-IDF + SVD

### 제안 알고리즘

- **P1**: Multi-Scale Character Statistics
- **P2**: Learned Feature Weights (Siamese-inspired)
- **P3**: Hierarchical Locality Sensitive Hashing
- **P4**: Autoencoder-Compressed Features

## 평가 메트릭

### 품질 메트릭

- **AUC-ROC**: 분류 성능
- **Precision@K**: 상위 K개 정확도
- **Separation**: Positive/Negative 분포 분리도
- **Best F1**: 최적 임계값에서의 F1 스코어

### 효율성 메트릭

- **Encoding Speed**: 초당 인코딩 가능한 텍스트 수
- **Vector Bytes**: 벡터당 메모리 사용량
- **Comparison Speed**: 초당 벡터 비교 횟수

## 상세 문서

전체 연구 프로토콜은 [RESEARCH_PROTOCOL.md](./RESEARCH_PROTOCOL.md)를 참조하세요.

## 라이선스

MIT License

## 기여

이슈와 PR을 환영합니다.
