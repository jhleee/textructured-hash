# Text Structure Hashing Research Protocol

이 문서는 현재 저장소에서 모델을 비교하고 우승자를 선정하는 **canonical protocol**입니다. 초기 24-category pair 실험은 구현 탐색용 legacy benchmark이며 최종 선정에는 사용하지 않습니다.

## 1. 연구 질문

의미 임베딩 없이 UTF-8 텍스트의 구조·형식을 작은 벡터로 표현하면서 다음을 동시에 만족할 수 있는가?

- 형식이 같은 텍스트는 가까워야 한다.
- 혼동하기 쉬운 다른 형식은 분리되어야 한다.
- 새로운 template과 완전히 보지 못한 family에도 전이되어야 한다.
- 배포 벡터는 256 bytes 이하이고 단일 프로세스에서 10,000 texts/s 이상이어야 한다.
- 빈 문자열부터 100,001자 입력까지 오류·NaN 없이 결정적으로 동작해야 한다.

이 연구는 의미, 사실, 엔티티, 번역 동등성 또는 exact-near-duplicate relevance를 평가하지 않습니다.

## 2. 후보 범위

공통 팩토리 `scripts.evaluate.get_encoder()`가 노출하는 모든 구현을 평가합니다.

- Baselines: `random_projection`, `simhash`, `minhash`, `tfidf_svd`
- Proposed: `multiscale`, `structure_type`, `structure_type_fast`, `structure_type_quantized`, `structure_type_quantized_256`, `ngram_hash`, `ngram_hash_multiscale`, `pattern_free`, `learned_weights`, `fisher`, `generalized`

학습형 후보는 동일한 train roots에서 생성한 pair만 사용합니다. TF-IDF/N-gram은 train text에 fit하고, LearnedWeights는 train positive/negative pair에 fit하며, Fisher/Generalized는 train category와 pair로 projection을 학습합니다.

새 후보는 공통 팩토리와 `ALL_MODELS`에 추가되어야 하며, 일부 후보만 제외한 결과를 “전체 벤치마크”라고 부르면 안 됩니다.

## 3. 데이터와 split

최종 benchmark generator는 `src/data/real_world_benchmark.py`입니다.

- Seed: `20260729`
- Roots per family: `30`
- Domains: `10`
- Format families: `20`
- Records: `1,560`
- 각 domain의 두 family 중 하나는 학습에서 완전히 제외된 family-OOD입니다.
- 알려진 family의 root는 pair 생성 전에 train 60% / validation 20% / test 20%로 나뉩니다.
- test 및 family-OOD root는 `standard`, `alternate`, `ood` renderer를 가집니다.

### 필수 integrity gate

실행 전 다음을 자동 검증합니다.

- root ID가 split 사이에 겹치지 않는다.
- exact text SHA-256이 split 사이에 겹치지 않는다.
- family-OOD family가 train에 존재하지 않는다.
- template-OOD renderer 조합이 train에 존재하지 않는다.
- 같은 seed로 재생성한 manifest SHA-256이 일치한다.

Integrity gate가 실패하면 결과를 생성하지 않습니다.

## 4. 평가 표현과 threshold

품질과 저장 크기는 서로 다른 표현으로 측정하면 안 됩니다.

- `encode_int8`이 있는 모델: 실제 int8 배포 벡터 사용
- 그 외: `encode`가 반환하는 float32 벡터 사용
- cosine 계산 전 평가 matrix는 float32로 변환하고 L2 normalize
- pair threshold는 validation에서 best F1 기준으로 한 번 선택
- 선택한 threshold를 ID, template-OOD, family-OOD에 그대로 적용
- test/OOD에서 threshold 또는 hyperparameter를 다시 조정하지 않음

## 5. 21개 mandatory gate

### Reliability — 4

1. Root/text/family/renderer isolation 통과
2. 같은 입력의 반복 encode가 bit-identical
3. Boundary input에서 error·non-finite output 없음
4. Mutation suite에서 encode error 없음

### Efficiency — 2

5. Stored vector ≤ 256 bytes
6. Warm single-process throughput 중앙값 ≥ 10,000 texts/s

처리량은 test text 160개를 warm-up한 후 5회 측정한 중앙값입니다. hardware, Python, NumPy, SciPy, scikit-learn, Numba, `PYTHONHASHSEED`를 결과에 기록합니다.

### Pair classification — 3

7. ID pair AUC ≥ 0.90
8. ID fixed-threshold F1 ≥ 0.80
9. Family-OOD pair AUC ≥ 0.85

### Retrieval — 7

10. ID precision@min(R,10) ≥ 0.80
11. ID nDCG@10 ≥ 0.90
12. ID worst-domain nDCG@10 ≥ 0.75
13. Template-OOD nDCG@10 ≥ 0.75
14. Template-OOD worst-domain nDCG@10 ≥ 0.50
15. Family-OOD nDCG@10 ≥ 0.65
16. Family-OOD worst-domain nDCG@10 ≥ 0.50

### Clustering — 2

17. ID ARI ≥ 0.65
18. Family-OOD ARI ≥ 0.50

KMeans에는 평가 family 수를 K로 제공합니다. 따라서 이 결과는 주어진 K에서의 grouping 성능이며 cluster count 자동 발견 성능이 아닙니다.

### Operations — 3

19. ID triage macro-F1 ≥ 0.85
20. Template-OOD triage macro-F1 ≥ 0.75
21. Family novelty AUROC ≥ 0.80

## 6. 판정과 우승자 선정

- `SYNTHETIC_GO`: 21/21 gate 통과
- `NO_GO`: 하나 이상의 mandatory gate 실패

후보 정렬은 사전에 코드에 선언된 다음 lexicographic policy를 사용합니다.

1. Verdict class
2. 통과 gate 수
3. Family-OOD worst-domain nDCG@10
4. Template-OOD worst-domain nDCG@10
5. Family-OOD pair AUC
6. ID retrieval nDCG@10
7. Throughput

첫 후보를 **synthetic benchmark winner**로 부릅니다. 여러 후보가 21/21을 통과하더라도 OOD worst-domain 성능을 평균보다 먼저 비교합니다.

`SYNTHETIC_GO` 또는 1위라는 이유만으로 production-ready라고 표현하지 않습니다. Production 승인은 별도의 post-freeze 독립 holdout과 실제 shadow validation이 필요합니다.

## 7. 재현 명령

Python 3.11 환경에서 의존성을 설치한 뒤 실행합니다.

```bash
pip install -r requirements.txt

PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/benchmark_real_world.py \
  --roots-per-family 30 \
  --seed 20260729 \
  --output results/final_full_benchmark
```

기본 후보 목록이 전체 구현 목록이므로 `--models`는 생략합니다. 후보 일부만 확인할 때는 명시적으로 이름을 나열하되 결과를 전체 순위로 보고하지 않습니다.

## 8. 산출물 계약

`results/final_full_benchmark/metrics.json`에는 다음이 포함되어야 합니다.

- config와 runtime environment
- data integrity audit와 manifest hash
- winner 및 전체 ranking
- 모델별 pair/retrieval/clustering/operations/mutation/performance 지표
- 모델별 21개 gate와 verdict
- model training/total wall time

`report.md`는 동일 JSON에서 생성하며, 숫자를 수동 복사해 별도 truth source를 만들지 않습니다.

## 9. Legacy pair benchmark의 용도

`data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`은 초기 24-category 개발 실험을 보존합니다. 다음 이유로 최종 선정에는 부적합합니다.

- pair 행을 나눈 것이므로 동일 원문이 여러 split의 다른 pair에 재등장할 수 있습니다.
- `scripts/evaluate.py`의 best F1 threshold는 test score에서 선택됩니다.
- retrieval, clustering, triage, novelty, template-OOD, family-OOD를 측정하지 않습니다.

이 benchmark는 과거 실험 재현, encoder smoke test, feature ablation에만 사용합니다.

## 10. 해석 제한과 후속 검증

현재 protocol도 다음 한계가 있습니다.

- 합성 generator와 feature/routing 설계가 같은 연구 과정에서 반복 관찰되었습니다.
- 지표는 grouped bootstrap confidence interval이 없는 point estimate입니다.
- throughput에는 vector DB, serialization, IPC, network overhead가 없습니다.
- retrieval relevance는 format-family relevance입니다.
- clustering은 정답 K를 사용합니다.
- 실제 class prevalence, malformed input 분포, drift를 반영하지 않습니다.

최종 후보를 배포하기 전 [`HOLDOUT_VALIDATION_PROTOCOL.md`](HOLDOUT_VALIDATION_PROTOCOL.md)의 post-freeze 검증을 수행해야 합니다.
