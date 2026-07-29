# Text Structure Hashing — Final Research Report

**Canonical result date:** 2026-07-29
**Canonical artifacts:** [`final_full_benchmark/metrics.json`](final_full_benchmark/metrics.json), [`final_full_benchmark/report.md`](final_full_benchmark/report.md)

## Executive summary

15개 구현 인코더를 동일한 leak-resistant ID/template-OOD/family-OOD 프로토콜로 다시 평가했습니다. **`GeneralizedStructureEncoder`가 유일하게 21/21 mandatory gate를 통과해 최종 synthetic benchmark 우승자**가 되었습니다.

초기 보고서에서 추천했던 Multiscale V1은 당시 5개 후보의 pair AUC만 비교한 결과였습니다. 전체 사용 사례와 OOD 평가에서는 7/21로 13위입니다. Fisher는 초기 24-category pair benchmark에서는 강했지만 template/family OOD retrieval과 routing이 무너져 15/21 `NO_GO`였습니다.

현재 결론은 다음과 같습니다.

- **전체 synthetic benchmark 우승자:** GeneralizedStructureEncoder
- **판정:** `SYNTHETIC_GO`, 21/21
- **배포 표현:** 256-dimensional int8, 256 bytes
- **처리량:** 33,055 texts/s, 5회 warm local run 중앙값
- **Production 상태:** 미승인. 독립 post-freeze holdout과 shadow validation 필요

## 문제 정의

목표는 의미 임베딩 없이 텍스트의 구조적 형식을 벡터화하는 것입니다. 같은 URL 형식, 로그 형식, 설정 형식, 코드/표 형식은 내용이 달라도 가까워야 하며, 혼동 가능한 다른 형식과는 분리되어야 합니다.

핵심 제약은 다음과 같습니다.

- stored vector ≤ 256 bytes
- encoding throughput ≥ 10,000 texts/s
- deterministic and boundary-safe inference
- ID뿐 아니라 새로운 template과 완전히 unseen family에서 일반화

의미, 사실, 엔티티, 번역 동등성은 이 연구의 relevance contract에 포함되지 않습니다.

## 최종 평가 설계

### 데이터

- Deterministic seed: `20260729`
- 1,560 records
- 10 domains
- 20 format families
- Family당 30 latent roots
- Known family root는 pair 생성 전에 train/validation/test로 분리
- Domain마다 한 family를 학습에서 완전히 제외
- Test와 family-OOD에는 standard/alternate/OOD renderer 제공

### Integrity

- Root overlap across splits: 0
- Exact-text overlap across splits: 0
- Family-OOD/train family overlap: 0
- Template-OOD/train renderer overlap: 0
- Repeated manifest generation: deterministic
- Manifest SHA-256: `2f776a48b57d16cae4e0efea611c434290c1ee02bf21acdd380cf7906fdb1a53`

### 평가 원칙

- Validation에서 threshold를 한 번 선택하고 test/OOD에 고정
- `encode_int8` 구현이 있으면 실제 int8 representation으로 품질 측정
- Pair, retrieval, clustering, triage, novelty, mutation, boundary, storage, throughput을 함께 평가
- 처리량은 warm run 5회의 중앙값
- 총 21개 mandatory gate 사용

자세한 기준은 [`../RESEARCH_PROTOCOL.md`](../RESEARCH_PROTOCOL.md)에 있습니다.

## 전체 결과

| Rank | Model | Gates | Pair AUC ID / template / family | Retrieval nDCG@10 ID / template / family | Speed/s | Bytes | Verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **generalized** | **21/21** | 0.9369 / 0.9062 / 0.9821 | 0.9879 / 0.8572 / 0.9590 | 33,055 | 256 | **SYNTHETIC_GO** |
| 2 | fisher | 15/21 | 1.0000 / 0.9374 / 0.7912 | 1.0000 / 0.7125 / 0.6000 | 74,848 | 256 | NO_GO |
| 3 | random_projection | 11/21 | 0.8162 / 0.8764 / 0.8008 | 0.6925 / 0.8359 / 0.6893 | 40,009 | 512 | NO_GO |
| 4 | structure_type_quantized | 11/21 | 0.7861 / 0.7926 / 0.7760 | 0.5305 / 0.6734 / 0.6543 | 6,434 | 128 | NO_GO |
| 5 | tfidf_svd | 10/21 | 0.9227 / 0.9836 / 0.7218 | 0.6821 / 0.8595 / 0.3880 | 956 | 512 | NO_GO |
| 6 | structure_type | 10/21 | 0.7866 / 0.7922 / 0.7767 | 0.5344 / 0.6694 / 0.6581 | 6,666 | 512 | NO_GO |
| 7 | structure_type_quantized_256 | 10/21 | 0.7687 / 0.7811 / 0.7716 | 0.5230 / 0.6664 / 0.6600 | 6,440 | 256 | NO_GO |
| 8 | structure_type_fast | 9/21 | 0.7547 / 0.7385 / 0.7793 | 0.4758 / 0.5877 / 0.6998 | 9,777 | 512 | NO_GO |
| 9 | ngram_hash_multiscale | 8/21 | 0.7349 / 0.6768 / 0.7856 | 0.4170 / 0.4034 / 0.5542 | 5,069 | 512 | NO_GO |
| 10 | ngram_hash | 8/21 | 0.5562 / 0.6292 / 0.7275 | 0.2504 / 0.2836 / 0.4512 | 4,707 | 512 | NO_GO |
| 11 | pattern_free | 8/21 | 0.6911 / 0.7424 / 0.7393 | 0.4419 / 0.5331 / 0.4300 | 2,537 | 512 | NO_GO |
| 12 | minhash | 8/21 | 0.6159 / 0.5772 / 0.6685 | 0.2013 / 0.2441 / 0.2693 | 339 | 512 | NO_GO |
| 13 | multiscale | 7/21 | 0.7113 / 0.7052 / 0.7852 | 0.3827 / 0.4439 / 0.4300 | 6,687 | 512 | NO_GO |
| 14 | simhash | 6/21 | 0.6729 / 0.5761 / 0.6906 | 0.3258 / 0.2653 / 0.2991 | 646 | 512 | NO_GO |
| 15 | learned_weights | 5/21 | 0.5923 / 0.7084 / 0.6032 | 0.2193 / 0.3375 / 0.3410 | 2,555 | 512 | NO_GO |

순위는 verdict, 통과 gate 수, family/template OOD worst-domain nDCG, family-OOD pair AUC, ID nDCG, throughput 순으로 결정했습니다. 상세 worst-domain 값과 모든 실패 gate는 canonical report에 있습니다.

## 우승 모델 분석

Generalized encoder는 작은 supervised Fisher branch와 label-free lexical, run-shape, layout branch를 결합합니다. 각 branch를 독립 정규화한 뒤 가중 연결해 supervised branch가 cosine을 지배하지 않도록 했습니다. Generic column evidence에 따라 table/delimited mixture를 선택합니다.

### 강점

- ID retrieval nDCG@10 `0.9879`
- Template-OOD retrieval nDCG@10 `0.8572`, worst-domain `0.5184`
- Family-OOD retrieval nDCG@10 `0.9590`, worst-domain `0.6316`
- Family-OOD pair AUC `0.9821`
- ID/family-OOD clustering ARI `1.0000/1.0000`
- ID/template-OOD triage macro-F1 `1.0000/0.8376`
- Family novelty AUROC `1.0000`
- Mutation cosine mean/p10 `0.9654/0.9215`
- Boundary errors `0`
- 256-byte int8와 33,055 texts/s를 동시에 달성

### 주의점

- Fisher보다 ID pair AUC는 낮습니다: `0.9369` vs `1.0000`.
- Adaptive visual-column/machine-delimiter routing은 이 benchmark를 관찰하며 선택되었습니다.
- 결과는 grouped bootstrap CI가 없는 point estimate입니다.
- Clustering은 정답 family 수를 K로 사용합니다.
- 처리량에 vector DB, serialization, network 비용은 포함되지 않습니다.

## 왜 Fisher와 Multiscale이 최종 우승자가 아닌가

### Fisher

Fisher는 과거 24-category pair benchmark에서 AUC 약 `0.998`, F1 약 `0.987`, separation 약 `3.68`로 7/7 목표를 달성했습니다. 그러나 최종 OOD 평가에서는 다음 gate를 실패했습니다.

- Family-OOD pair AUC: `0.7912` < `0.85`
- Template-OOD retrieval nDCG@10: `0.7125` < `0.75`
- Family-OOD retrieval nDCG@10: `0.6000` < `0.65`
- Template-OOD triage F1: `0.4833` < `0.75`
- Template/family OOD worst-domain retrieval: `0.0000`

즉 알려진 class에는 매우 강하지만 renderer와 family transfer가 부족했습니다.

### Multiscale V1

Multiscale은 초기 5개 후보 비교에서 pair AUC `0.9549`로 “best quality”였습니다. 하지만 그 평가는 pair-only이고 split leakage 가능성이 있으며 OOD/retrieval/routing을 포함하지 않았습니다. 최종 공통 평가에서 Multiscale은 7/21, ID pair AUC `0.7113`, family-OOD nDCG@10 `0.4300`, 512 bytes로 우승 조건을 만족하지 못했습니다.

## 연구 흐름

1. **초기 baselines와 Multiscale:** pair AUC 기준으로 Multiscale이 초기 1위였으나 2/7 목표만 통과
2. **StructureType/quantization:** 초기 pair F1과 저장 크기를 개선했지만 separation·속도 한계
3. **Pattern-free:** hardcoded detector 제거 가설을 시험했지만 후속 OOD에서 8/21
4. **Fisher:** 초기 category benchmark에서 7/7을 달성했으나 family/template transfer 실패
5. **Extended real-world/OOD:** pair-only 결과가 실제 retrieval/routing 일반화를 과대평가함을 확인
6. **Generalized:** label-free invariant branch와 bounded supervised branch를 결합해 최종 21/21 달성

과거 실험 문서는 당시 증거를 보존하는 historical snapshot입니다. 그 문서의 추천 문구는 이 보고서로 대체됩니다.

## 재현

```bash
PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/benchmark_real_world.py \
  --roots-per-family 30 \
  --seed 20260729 \
  --output results/final_full_benchmark
```

Canonical run environment는 Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, Numba 0.66.0입니다. 전체 runtime과 모델별 runtime은 `metrics.json`에 기록됩니다.

## 최종 결론과 다음 단계

**GeneralizedStructureEncoder를 repository-wide synthetic benchmark의 최종 우승자로 채택합니다.** 다만 이 결론을 production-ready 선언으로 확대하지 않습니다.

다음 단계는 [`../HOLDOUT_VALIDATION_PROTOCOL.md`](../HOLDOUT_VALIDATION_PROTOCOL.md)에 따라 구현과 checkpoint를 freeze하고, 현재 generator/source lineage와 독립적인 grouped holdout을 한 번 blind 실행하는 것입니다. 그 후 비식별 shadow traffic에서 relevance, abstention, drift, vector DB end-to-end 비용을 검증해야 합니다.
