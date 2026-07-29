# Text Structure Hashing

의미 임베딩 없이 텍스트의 **형식과 구조**를 고정 길이 벡터로 표현하는 경량 인코더 연구 프로젝트입니다. URL, 로그, 설정, 코드, 표처럼 내용은 달라도 형식이 유사한 텍스트를 검색·분류·군집화하는 것이 목표입니다.

## 현재 결론

15개 구현 인코더를 동일한 root-first ID/template-OOD/family-OOD 프로토콜로 비교한 결과, **`GeneralizedStructureEncoder`가 최종 synthetic benchmark 우승자**입니다. 유일하게 21개 gate를 모두 통과했습니다.

| 항목 | Generalized 결과 |
|---|---:|
| Pair AUC — ID / template OOD / family OOD | 0.9369 / 0.9062 / 0.9821 |
| Retrieval nDCG@10 — ID / template / family | 0.9879 / 0.8572 / 0.9590 |
| Triage macro-F1 — ID / template OOD | 1.0000 / 0.8376 |
| Family novelty AUROC | 1.0000 |
| 저장 크기 | 256 bytes (`int8`, 256차원) |
| 인코딩 처리량 | 33,055 texts/s (5회 warm run 중앙값) |
| 판정 | **SYNTHETIC_GO, 21/21** |

2위 Fisher는 15/21 `NO_GO`, 초기 추천이었던 Multiscale V1은 7/21로 13위였습니다. 전체 순위와 모델별 실패 gate는 [`results/final_full_benchmark/report.md`](results/final_full_benchmark/report.md)에 있습니다.

> `SYNTHETIC_GO`는 고정된 합성 벤치마크를 모두 통과했다는 뜻입니다. 실제 조직의 데이터 분포, malformed input, drift, vector DB 비용까지 검증한 production 승인을 의미하지 않습니다.

## 목표와 범위

- 메모리: 배포 벡터 기준 ≤ 256 bytes/text
- 속도: single-process warm run 기준 ≥ 10,000 texts/s
- 출력: 64–256차원, cosine similarity에 사용 가능한 벡터
- 추론: heavyweight neural framework 없이 NumPy/Numba 기반
- 대상: 구조·형식 유사성
- 비대상: 의미, 사실, 엔티티 또는 문장 동의어 유사성

예를 들어 `https://a.example/x`와 `https://b.example/y`는 구조적으로 가깝지만, 서로 다른 언어로 작성된 동의 문장은 구조적으로 멀 수 있습니다.

## 설치

Python 3.11 환경을 권장합니다.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 최종 전체 벤치마크 재현

```bash
PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/benchmark_real_world.py \
  --roots-per-family 30 \
  --seed 20260729 \
  --output results/final_full_benchmark
```

인자를 생략하면 구현된 15개 후보를 모두 평가합니다. 결과는 `metrics.json`과 사람이 읽을 수 있는 `report.md`로 저장됩니다. Python 내장 `hash()`를 쓰는 후보의 재현성을 위해 `PYTHONHASHSEED=1`이 필수입니다.

특정 후보만 비교할 수도 있습니다.

```bash
PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/benchmark_real_world.py \
  --models fisher generalized \
  --roots-per-family 30 \
  --seed 20260729 \
  --output results/fisher_vs_generalized
```

## 단일 legacy pair 평가

초기 24-category 실험을 재현하거나 개별 구현을 디버깅할 때만 사용합니다.

```bash
python scripts/evaluate.py \
  --model generalized \
  --train data/train.jsonl \
  --test data/test.jsonl \
  --output results/generalized_pair_debug
```

`data/train.jsonl`과 `data/test.jsonl`은 pair 행 단위로 분할되어 동일 원문이 여러 split에 나타날 수 있고, `evaluate.py`는 test에서 best F1 threshold를 탐색합니다. 따라서 이 경로의 수치는 최종 우승자 선정 근거가 아닙니다.

## 평가 프로토콜

최종 벤치마크는 다음 원칙을 사용합니다.

1. **Root-first split:** pair 생성 전에 latent root를 train/validation/test로 분리합니다.
2. **두 OOD 축:** 새로운 renderer/template와 완전히 보지 못한 family를 따로 평가합니다.
3. **고정 threshold:** validation에서 한 번 선택하고 ID/OOD test에서 재조정하지 않습니다.
4. **실제 배포 표현:** `encode_int8`이 있으면 int8 벡터로 품질과 저장 크기를 함께 측정합니다.
5. **다중 사용 사례:** pair classification뿐 아니라 retrieval, clustering, triage, novelty, mutation·boundary 안정성을 평가합니다.
6. **명시적 우승 규칙:** verdict → 통과 gate 수 → family/template OOD worst-domain retrieval → family-OOD AUC → ID retrieval → 처리량 순으로 정렬합니다.

세부 gate와 설계 근거는 [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md)를 참조하세요.

## 구현 후보

### Baselines

`random_projection`, `simhash`, `minhash`, `tfidf_svd`

### Proposed encoders

`multiscale`, `structure_type`, `structure_type_fast`, `structure_type_quantized`, `structure_type_quantized_256`, `ngram_hash`, `ngram_hash_multiscale`, `pattern_free`, `learned_weights`, `fisher`, `generalized`

`learned_weights`는 이번 최종 벤치마크에서 공통 CLI에 연결되었습니다.

## 저장소 구조

```text
.
├── data/                         # 초기 pair 데이터셋
├── notes/                        # 과거 연구 노트
├── results/
│   ├── final_full_benchmark/     # 현재 canonical 결과
│   └── ...                       # 개별·과거 실험 스냅샷
├── scripts/
│   ├── benchmark_real_world.py   # 최종 전체 벤치마크
│   ├── evaluate.py               # 모델 팩토리와 legacy pair 평가
│   ├── generate_holdout_data.py  # legacy Fisher holdout 데이터 생성
│   └── evaluate_holdout.py       # legacy Fisher 전용 holdout 평가
├── src/
│   ├── data/                     # leak-resistant benchmark generator
│   ├── encoders/                 # 15개 인코더 구현
│   └── evaluation/               # pair·효율성 메트릭
├── RESEARCH_PROTOCOL.md          # 현재 평가·선정 프로토콜
├── HOLDOUT_VALIDATION_PROTOCOL.md
└── results/FINAL_REPORT.md       # 최종 연구 요약
```

## 문서 상태

- **현재 기준:** `README.md`, `RESEARCH_PROTOCOL.md`, `HOLDOUT_VALIDATION_PROTOCOL.md`, `results/FINAL_REPORT.md`, `results/final_full_benchmark/`
- **과거 실험 스냅샷:** `EXPERIMENT_V2_RESULTS.md`, `EXPERIMENT_PATTERN_FREE.md`, `notes/`, 기타 `results/*/report.md`

과거 보고서는 당시 질문과 데이터셋에 대한 증거를 보존하기 위해 결과 본문을 유지하고 historical 안내만 추가합니다. 그 안의 “best”, “recommended”, “production-ready” 표현은 현재 전체 벤치마크 결론으로 해석하면 안 됩니다.

## 다음 검증

1. Generalized 구현과 routing weight를 freeze/checkpoint합니다.
2. 현재 generator/source lineage와 독립적인 post-freeze grouped holdout을 blind 실행합니다.
3. 실제 prevalence에서 abstention과 fallback을 보정합니다.
4. 비식별 shadow traffic과 목표 vector DB에서 품질·latency·index 크기를 검증합니다.

## 라이선스

MIT License
