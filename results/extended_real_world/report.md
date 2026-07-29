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
| fisher | 1.0000/0.9374/0.7912 | 1.0000/0.7125/0.6000 | 1.0000/1.0000 | 1.0000/0.4833 | 1.0000 | 66770 | 256/int8 | **NO_GO** |
| pattern_free | 0.6911/0.7424/0.7393 | 0.4419/0.5331/0.4300 | 1.0000/0.9421 | 1.0000/0.1667 | 0.9272 | 2569 | 512/float32 | **NO_GO** |
| structure_type_quantized_256 | 0.7687/0.7811/0.7716 | 0.5230/0.6664/0.6600 | 1.0000/1.0000 | 1.0000/0.3672 | 0.8875 | 6506 | 256/int8 | **NO_GO** |
| tfidf_svd | 0.9227/0.9836/0.7218 | 0.6821/0.8595/0.3880 | 1.0000/1.0000 | 1.0000/0.4833 | 0.7772 | 985 | 512/float32 | **NO_GO** |

## Gate details

### fisher: NO_GO

Passed 15/21 gates. Basis: one or more mandatory general-production gates failed. Failed: family_ood_pair_auc_gte_085, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9621 / 0.8997
- Latency p50/p95/p99: 0.0147 / 0.0163 / 0.0181 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 1.0000 / 1.0000

### pattern_free: NO_GO

Passed 8/21 gates. Basis: one or more mandatory general-production gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9897 / 0.9613
- Latency p50/p95/p99: 0.3871 / 0.5073 / 0.5446 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.9272 / 0.9381

### structure_type_quantized_256: NO_GO

Passed 10/21 gates. Basis: one or more mandatory general-production gates failed. Failed: throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9988 / 0.9975
- Latency p50/p95/p99: 0.1465 / 0.2299 / 0.2461 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.8875 / 0.9130

### tfidf_svd: NO_GO

Passed 10/21 gates. Basis: one or more mandatory general-production gates failed. Failed: storage_lte_256, throughput_gte_10k, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075, family_novelty_auroc_gte_080.

- Mutation invariance mean/p10 cosine: 0.9984 / 0.9972
- Latency p50/p95/p99: 1.0106 / 1.0975 / 1.1557 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.7772 / 0.8589

## Practical use-case assessment

The strongest candidate is **fisher**, and its general-production verdict is **NO_GO**. Bounded known-format pilot eligibility: **True**.

| Use case | Evidence from strongest candidate | Recommendation |
|---|---|---|
| Known-format similarity search | ID nDCG@10 1.0000, P@min(R,10) 1.0000 (6.0000x random P@10) | **Pilot** for bounded, versioned format catalogs |
| New-template search | nDCG@10 0.7125, worst-domain 0.0000 | **Human-assisted only**; failures: code_build, identifiers, observability, tables_cli |
| Completely unseen-family search | nDCG@10 0.6000, worst-domain 0.0000 | **Do not automate**; failures: documents, operations, tables_cli, web_api |
| Offline format discovery/clustering | ID/family-OOD ARI 1.0000/1.0000 | **Promising** for analyst-reviewed grouping |
| Known-format routing/triage | ID macro-F1 1.0000; template-OOD macro-F1 0.4833 | **Known templates only**, with abstention and fallback |
| Drift/anomaly candidate ranking | Balanced family-novelty AUROC 1.0000, AP 1.0000 | **Shadow alerts only** until real prevalence is measured |
| Clipboard/log/config organization | Mutation p10 cosine 0.8997; boundary errors 0 | **Suitable pilot** with explicit UTF-8 and length policy |
| Semantic or entity-level retrieval | Benchmark labels structure, not meaning | **Not supported** by this research |

### Main failure modes

- Cross-template pair transfer drops from ID AUC/F1 1.0000/1.0000 to family-OOD AUC/F1 0.7912/0.4557.
- Average OOD scores hide complete domain failures. Template-OOD failed in code_build, identifiers, observability, tables_cli; family-OOD failed in documents, operations, tables_cli, web_api.
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
- `NO_GO` means no general production claim under these gates; it does not mean the encoder has no useful narrower application.

## Reproduce

```bash
PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_real_world.py --roots-per-family 30 --output results/extended_real_world
```

## Recommended next validation

1. Run a shadow pilot on de-identified traffic, with source/template groups assigned before train/test splitting.
2. Manually adjudicate the highest-impact false matches in the failed domains and define the exact relevance contract.
3. Calibrate routing abstention and anomaly thresholds on real prevalence; do not reuse the synthetic threshold.
4. Add a vector-database integration benchmark at the intended corpus size before committing to latency or index-size targets.
