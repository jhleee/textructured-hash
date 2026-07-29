# Post-Freeze Holdout Validation Protocol

## 상태

최종 전체 synthetic benchmark에서 `GeneralizedStructureEncoder`가 21/21 gate를 통과해 우승했습니다. 그러나 generalized의 label-free branch 구성과 adaptive routing weight는 같은 benchmark를 관찰하며 선택되었습니다. 따라서 현재 결과는 **synthetic benchmark winner**를 결정하기에는 충분하지만 production 승인을 위한 독립 holdout은 아닙니다.

이 문서는 우승 모델을 freeze한 뒤 수행해야 할 다음 검증을 정의합니다. 저장소의 기존 `generate_holdout_data.py`와 `evaluate_holdout.py`는 Fisher 전용 legacy pair validation이므로 Generalized 최종 승인 근거로 사용하지 않습니다.

## 1. 목적

다음을 독립적으로 확인합니다.

- 현재 generator/template/family에 과적합되지 않았는가?
- 완전히 새로운 source family와 renderer에서도 retrieval·routing 성능이 유지되는가?
- 실제 prevalence와 malformed input에서 abstention/fallback을 보정할 수 있는가?
- 목표 vector DB를 포함한 end-to-end 비용이 제약을 만족하는가?

## 2. Freeze 대상

Holdout 데이터 또는 aggregate 결과를 보기 전에 다음을 고정합니다.

- `GeneralizedStructureEncoder` feature extraction 코드
- Fisher branch 차원과 학습 절차
- branch weight 및 table/delimiter routing 조건
- vector 차원, dtype, quantization 방식
- pair threshold 선택 방식
- 21개 gate와 우승자 선정 순서
- dependency lock 또는 실행 환경 manifest
- git commit SHA와 model checkpoint SHA-256

Holdout 공개 후 위 항목을 변경하면 해당 데이터는 development set으로 전환되며 새 holdout이 필요합니다.

## 3. 독립성 요구사항

### Generator/source independence

- `src/data/real_world_benchmark.py`의 renderer를 import하거나 복사하지 않습니다.
- 기존 20 family와 다른 source lineage에서 최소 10개 family를 구성합니다.
- 실제 데이터가 허용되면 비식별·승인된 shadow sample을 우선합니다.
- 합성 데이터라면 구현자가 아닌 별도 작성자 또는 사전 고정된 외부 fixture가 생성합니다.
- family·template·root ID를 pair 생성 전에 부여합니다.

### Split independence

- Root 또는 source document 단위로 split합니다.
- Exact text와 canonicalized text hash overlap을 모두 검사합니다.
- 최소 한 축은 family 전체 holdout이어야 합니다.
- Threshold는 holdout test가 아닌 별도 calibration split에서만 선택합니다.
- 최종 test label은 한 번만 엽니다.

### 규모 권장

- 10개 이상 domain
- 20개 이상 family
- family당 50개 이상 latent roots
- 각 root당 2개 이상 renderer
- pair class별 2,000개 이상
- retrieval query 1,000개 이상
- 실제 prevalence를 반영한 novelty set과 50/50 balanced diagnostic set을 둘 다 보고

## 4. 평가 항목

현재 [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md)의 21개 gate를 그대로 적용하고 다음을 추가합니다.

1. Root/domain/family grouped bootstrap 95% CI
2. Worst-domain 및 worst-family 지표와 표본 수
3. Calibration threshold의 coverage-risk curve
4. Abstention 시 human/fallback 경로까지 포함한 utility
5. 입력 길이·Unicode·malformed format별 오류율
6. 목표 vector DB에서 index 크기, build time, recall, p50/p95/p99 latency
7. Cold start와 warm steady-state 처리량
8. 실제 prevalence에서 novelty precision/recall

Hard quality gate는 가능한 경우 point estimate가 아니라 95% CI lower bound로 판정합니다.

## 5. 판정

| Outcome | 조건 | 해석 |
|---|---|---|
| `HOLDOUT_PASS` | Reliability/efficiency hard gate와 사전 등록된 quality CI gate 모두 통과 | 제한된 shadow pilot 후보 |
| `HOLDOUT_LIMITED` | Known-format use case는 통과하지만 template/family OOD gate 일부 실패 | 명시한 범위에서 human-assisted 사용만 고려 |
| `HOLDOUT_FAIL` | Reliability 실패, 핵심 OOD 품질 실패 또는 CI가 기준 미달 | 재설계 후 새로운 holdout 필요 |

`HOLDOUT_PASS`도 자동 production 승인을 의미하지 않습니다. 실제 shadow traffic, privacy/security review, 운영 rollback 계획이 별도로 필요합니다.

## 6. 결과 보고 계약

최종 보고서는 최소한 다음을 포함해야 합니다.

- freeze commit/checkpoint/environment hash
- dataset provenance와 승인 정보
- split 및 leakage audit
- 사전 등록 gate와 실제 결과
- aggregate, grouped CI, worst-group 결과
- calibration/abstention 정책
- known limitation과 실패 사례
- holdout 공개 후 변경 여부

실패 사례를 보고 feature나 routing을 수정한 경우, 수정된 모델은 같은 holdout 결과를 최종 점수로 재사용하지 않습니다.

## 7. 현재 저장소의 legacy Fisher holdout

다음 명령은 과거 Fisher encoder의 24 known + 4 new category pair 진단을 재현합니다.

```bash
python scripts/evaluate_holdout.py --phase save_model \
  --train data/train.jsonl \
  --model_path results/holdout_validation/frozen_model.npz

python scripts/generate_holdout_data.py \
  --output data/holdout \
  --seed 2024 \
  --pair_seed 7777 \
  --n_positive 3000 \
  --n_negative 3000

python scripts/evaluate_holdout.py --phase evaluate \
  --model_path results/holdout_validation/frozen_model.npz \
  --holdout_pairs data/holdout/pairs.jsonl \
  --output results/holdout_validation
```

이 경로의 제한은 다음과 같습니다.

- `evaluate_holdout.py`가 `FisherStructureEncoder`만 load합니다.
- pair-only 평가이며 retrieval·clustering·triage를 포함하지 않습니다.
- `evaluate()`가 holdout에서 best F1 threshold를 찾습니다.
- intra-family group에는 단일 class만 있어 “new families AUC”를 직접 계산할 수 없습니다.
- `SOFT PASS` 구현은 new-family AUC 대신 positive mean similarity를 사용합니다.

따라서 이 결과는 Fisher regression diagnostic일 뿐, 현재 Generalized 우승자를 뒤집거나 승인하는 근거가 아닙니다.

## 8. 현재 결론

- Repository-wide synthetic benchmark winner: **GeneralizedStructureEncoder**
- Synthetic verdict: **SYNTHETIC_GO, 21/21**
- Post-freeze independent holdout: **아직 수행되지 않음**
- Production status: **미승인; bounded shadow pilot 이전 단계**
