# Extended Real-World Validation Results

This benchmark uses deterministic, privacy-safe synthetic operational data. It is more realistic than the original category interpolation test, but it is not a substitute for a de-identified production pilot.

## Dataset and integrity

- Records: 1560; domains: 10; format families: 20
- Root/exact-text leakage across splits: none
- OOD family/renderer isolation: verified
- ID train/test intentionally share renderer families while roots and values remain disjoint
- Deterministic manifest SHA-256: `2f776a48b57d16cae4e0efea611c434290c1ee02bf21acdd380cf7906fdb1a53`
- Family-OOD policy: one of two families per domain is entirely excluded from training
- Pair threshold policy: selected on validation once and frozen for test/OOD

## Results summary

| Model | Cross-template Pair AUC ID / template OOD / family OOD | Retrieval nDCG@10 ID / template / family | Clustering ARI ID / family | Triage F1 ID / template | Family novelty AUROC | Speed/s | Bytes/dtype | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fisher | 1.0000/0.9374/0.7912 | 1.0000/0.7125/0.6000 | 1.0000/1.0000 | 1.0000/0.4833 | 1.0000 | 71453 | 256/int8 | **NO_GO** |
| generalized | 0.9369/0.9062/0.9821 | 0.9879/0.8572/0.9590 | 1.0000/1.0000 | 1.0000/0.8376 | 1.0000 | 33864 | 256/int8 | **SYNTHETIC_GO** |

## Gate details

### fisher: NO_GO

Passed 15/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: family_ood_pair_auc_gte_085, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9621 / 0.8997
- Latency p50/p95/p99: 0.0138 / 0.0149 / 0.0165 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 1.0000 / 1.0000

### generalized: SYNTHETIC_GO

Passed 21/21 gates. Basis: all synthetic benchmark gates passed; independent production validation is still required. Failed: none.

- Mutation invariance mean/p10 cosine: 0.9654 / 0.9215
- Latency p50/p95/p99: 0.0291 / 0.0326 / 0.0351 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 1.0000 / 1.0000

## Practical use-case assessment

The strongest candidate is **generalized**, and its synthetic benchmark verdict is **SYNTHETIC_GO**. Bounded known-format pilot eligibility: **True**.

| Use case | Evidence from strongest candidate | Recommendation |
|---|---|---|
| Known-format similarity search | ID nDCG@10 0.9879, P@min(R,10) 0.9611 (6.0000x random P@10) | **Pilot** for bounded, versioned format catalogs |
| New-template search | nDCG@10 0.8572, worst-domain 0.5184 | **Shadow candidate**; calibrate abstention/fallback before pilot; failures: none |
| Completely unseen-family search | nDCG@10 0.9590, worst-domain 0.6316 | **Shadow candidate only**; independent source-family holdout required; failures: none |
| Offline format discovery/clustering | ID/family-OOD ARI 1.0000/1.0000 | **Promising** for analyst-reviewed grouping |
| Known-format routing/triage | ID macro-F1 1.0000; template-OOD macro-F1 0.8376 | **Shadow candidate**; abstention coverage-risk is not yet measured |
| Drift/anomaly candidate ranking | Balanced family-novelty AUROC 1.0000, AP 1.0000 | **Shadow alerts only** until real prevalence is measured |
| Clipboard/log/config organization | Mutation p10 cosine 0.9215; boundary errors 0 | **Suitable pilot** with explicit UTF-8 and length policy |
| Semantic or entity-level retrieval | Benchmark labels structure, not meaning | **Not supported** by this research |

### Remaining risks and transfer profile

- Cross-template pair AUC/F1 for ID, template-OOD, and family-OOD: 0.9369/0.8652, 0.9062/0.8278, 0.9821/0.9404.
- Average OOD scores hide complete domain failures. Template-OOD failed in no domain; family-OOD failed in no domain.
- Clustering uses the true number of families as K; it supports analyst-reviewed grouping, not automatic discovery of cluster count.
- ID evaluation deliberately reuses renderer families with disjoint roots. Only template-OOD and family-OOD results support generalization claims.
- Retrieval gates use precision/nDCG and random baselines because Hit@10 is high even under random ranking when 10% of the corpus is relevant.
- The original pair-only benchmark would have missed these retrieval and routing failures.

## Interpretation limits

- Synthetic realism removes privacy/licensing risk and supports exact reproducibility, but cannot reproduce an organization's true class mix, malformed inputs, or drift.
- Results are point estimates without grouped bootstrap confidence intervals, and generator/source families are not independently held out.
- Retrieval relevance is format-family relevance. A product seeking semantic, entity, or exact-near-duplicate relevance needs a separate label contract.
- Quality metrics use the same deployed representation counted by the storage gate (true int8 where available; otherwise float32).
- Throughput is a single-process warm local measurement without vector-database or network overhead.
- `SYNTHETIC_GO` means every gate on this fixed synthetic development benchmark passed; it is not a production guarantee. `NO_GO` means one or more mandatory benchmark gates failed.
- The adaptive visual-column and machine-delimiter experts were selected on this benchmark. Their gains require confirmation on a post-freeze independent generator/source-family holdout.
- Abstention and fallback recommendations are intentionally deferred because no threshold, coverage-risk, or fallback evaluation is included.

## Reproduce

```bash
PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_real_world.py --models fisher generalized --roots-per-family 30 --seed 20260729 --output results/generalized_real_world
```

## Recommended next validation

1. Run a shadow pilot on de-identified traffic, with source/template groups assigned before train/test splitting.
2. Manually adjudicate the highest-impact false matches in the failed domains and define the exact relevance contract.
3. Calibrate routing abstention and anomaly thresholds on real prevalence; do not reuse the synthetic threshold.
4. Add a vector-database integration benchmark at the intended corpus size before committing to latency or index-size targets.
