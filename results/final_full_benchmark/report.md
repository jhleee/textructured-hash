# Final Full Benchmark Results

This benchmark compares every implemented encoder under one deterministic, leak-resistant protocol. It uses privacy-safe synthetic operational data and is not a substitute for an independent post-freeze holdout or a de-identified production pilot.

## Dataset and integrity

- Records: 1560; domains: 10; format families: 20
- Root/exact-text leakage across splits: none
- OOD family/renderer isolation: verified
- ID train/test intentionally share renderer families while roots and values remain disjoint
- Deterministic manifest SHA-256: `2f776a48b57d16cae4e0efea611c434290c1ee02bf21acdd380cf7906fdb1a53`
- Family-OOD policy: one of two families per domain is entirely excluded from training
- Pair threshold policy: selected on validation once and frozen for test/OOD
- Candidate coverage: 15 selected encoders (random_projection, simhash, minhash, tfidf_svd, multiscale, structure_type, structure_type_fast, structure_type_quantized, structure_type_quantized_256, ngram_hash, ngram_hash_multiscale, pattern_free, learned_weights, fisher, generalized)
- Benchmark scope: `repository_full`
- Deployment representation: `encode_int8` when implemented, otherwise float32 `encode` output

## Results summary

| Model | Cross-template Pair AUC ID / template OOD / family OOD | Retrieval nDCG@10 ID / template / family | Clustering ARI ID / family | Triage F1 ID / template | Family novelty AUROC | Speed/s | Bytes/dtype | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| random_projection | 0.8162/0.8764/0.8008 | 0.6925/0.8359/0.6893 | 1.0000/1.0000 | 1.0000/0.7333 | 0.8950 | 40009 | 512/float32 | **NO_GO** |
| simhash | 0.6729/0.5761/0.6906 | 0.3258/0.2653/0.2991 | 0.7525/0.7967 | 0.6728/0.2690 | 0.7257 | 646 | 512/float32 | **NO_GO** |
| minhash | 0.6159/0.5772/0.6685 | 0.2013/0.2441/0.2693 | 0.7609/1.0000 | 0.9832/0.2622 | 0.8806 | 339 | 512/float32 | **NO_GO** |
| tfidf_svd | 0.9227/0.9836/0.7218 | 0.6821/0.8595/0.3880 | 1.0000/1.0000 | 1.0000/0.4833 | 0.7772 | 956 | 512/float32 | **NO_GO** |
| multiscale | 0.7113/0.7052/0.7852 | 0.3827/0.4439/0.4300 | 1.0000/0.7999 | 0.6700/0.1490 | 0.8764 | 6687 | 512/float32 | **NO_GO** |
| structure_type | 0.7866/0.7922/0.7767 | 0.5344/0.6694/0.6581 | 1.0000/1.0000 | 1.0000/0.4681 | 0.9000 | 6666 | 512/float32 | **NO_GO** |
| structure_type_fast | 0.7547/0.7385/0.7793 | 0.4758/0.5877/0.6998 | 1.0000/1.0000 | 0.9832/0.3474 | 0.8989 | 9777 | 512/float32 | **NO_GO** |
| structure_type_quantized | 0.7861/0.7926/0.7760 | 0.5305/0.6734/0.6543 | 1.0000/1.0000 | 1.0000/0.4681 | 0.9053 | 6434 | 128/int8 | **NO_GO** |
| structure_type_quantized_256 | 0.7687/0.7811/0.7716 | 0.5230/0.6664/0.6600 | 1.0000/1.0000 | 1.0000/0.3672 | 0.8875 | 6440 | 256/int8 | **NO_GO** |
| ngram_hash | 0.5562/0.6292/0.7275 | 0.2504/0.2836/0.4512 | 1.0000/1.0000 | 1.0000/0.2929 | 0.9119 | 4707 | 512/float32 | **NO_GO** |
| ngram_hash_multiscale | 0.7349/0.6768/0.7856 | 0.4170/0.4034/0.5542 | 0.8966/1.0000 | 0.9646/0.4670 | 0.8972 | 5069 | 512/float32 | **NO_GO** |
| pattern_free | 0.6911/0.7424/0.7393 | 0.4419/0.5331/0.4300 | 1.0000/0.9421 | 1.0000/0.1667 | 0.9272 | 2537 | 512/float32 | **NO_GO** |
| learned_weights | 0.5923/0.7084/0.6032 | 0.2193/0.3375/0.3410 | 0.4604/0.9651 | 0.1000/0.2080 | 0.2076 | 2555 | 512/float32 | **NO_GO** |
| fisher | 1.0000/0.9374/0.7912 | 1.0000/0.7125/0.6000 | 1.0000/1.0000 | 1.0000/0.4833 | 1.0000 | 74848 | 256/int8 | **NO_GO** |
| generalized | 0.9369/0.9062/0.9821 | 0.9879/0.8572/0.9590 | 1.0000/1.0000 | 1.0000/0.8376 | 1.0000 | 33055 | 256/int8 | **SYNTHETIC_GO** |

## Winner and ranking

**Final synthetic benchmark winner: `generalized`.**

Ranking policy (declared in code): verdict class, gates passed, family-OOD worst-domain nDCG@10, template-OOD worst-domain nDCG@10, family-OOD pair AUC, ID nDCG@10, then throughput. Test thresholds are never retuned.

| Rank | Model | Decision | Gates | Family OOD worst-domain nDCG@10 | Template OOD worst-domain nDCG@10 | Family OOD pair AUC | Throughput/s |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | generalized | SYNTHETIC_GO | 21/21 | 0.6316 | 0.5184 | 0.9821 | 33055 |
| 2 | fisher | NO_GO | 15/21 | 0.0000 | 0.0000 | 0.7912 | 74848 |
| 3 | random_projection | NO_GO | 11/21 | 0.0000 | 0.2984 | 0.8008 | 40009 |
| 4 | structure_type_quantized | NO_GO | 11/21 | 0.0000 | 0.0000 | 0.7760 | 6434 |
| 5 | tfidf_svd | NO_GO | 10/21 | 0.0000 | 0.3433 | 0.7218 | 956 |
| 6 | structure_type | NO_GO | 10/21 | 0.0000 | 0.0000 | 0.7767 | 6666 |
| 7 | structure_type_quantized_256 | NO_GO | 10/21 | 0.0000 | 0.0000 | 0.7716 | 6440 |
| 8 | structure_type_fast | NO_GO | 9/21 | 0.0000 | 0.0000 | 0.7793 | 9777 |
| 9 | ngram_hash_multiscale | NO_GO | 8/21 | 0.0000 | 0.0684 | 0.7856 | 5069 |
| 10 | ngram_hash | NO_GO | 8/21 | 0.0000 | 0.0146 | 0.7275 | 4707 |
| 11 | pattern_free | NO_GO | 8/21 | 0.0000 | 0.0000 | 0.7393 | 2537 |
| 12 | minhash | NO_GO | 8/21 | 0.0000 | 0.0000 | 0.6685 | 339 |
| 13 | multiscale | NO_GO | 7/21 | 0.0000 | 0.0000 | 0.7852 | 6687 |
| 14 | simhash | NO_GO | 6/21 | 0.0000 | 0.0000 | 0.6906 | 646 |
| 15 | learned_weights | NO_GO | 5/21 | 0.0000 | 0.0000 | 0.6032 | 2555 |

## Gate details

### random_projection: NO_GO

Passed 11/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9911 / 0.9771
- Latency p50/p95/p99: 0.0251 / 0.0357 / 0.0384 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.8950 / 0.9134

### simhash: NO_GO

Passed 6/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_id_f1_gte_085, triage_template_ood_f1_gte_075, family_novelty_auroc_gte_080.

- Mutation invariance mean/p10 cosine: 0.8790 / 0.8125
- Latency p50/p95/p99: 1.4579 / 2.8042 / 3.0476 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.7257 / 0.6998

### minhash: NO_GO

Passed 8/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9682 / 0.9005
- Latency p50/p95/p99: 2.8169 / 4.9417 / 5.2538 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.8806 / 0.8561

### tfidf_svd: NO_GO

Passed 10/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075, family_novelty_auroc_gte_080.

- Mutation invariance mean/p10 cosine: 0.9984 / 0.9972
- Latency p50/p95/p99: 1.0411 / 1.1255 / 1.1660 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.7772 / 0.8589

### multiscale: NO_GO

Passed 7/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_id_f1_gte_085, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9861 / 0.9273
- Latency p50/p95/p99: 0.1481 / 0.2305 / 0.2535 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.8764 / 0.9108

### structure_type: NO_GO

Passed 10/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9992 / 0.9981
- Latency p50/p95/p99: 0.1446 / 0.2322 / 0.2617 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.9000 / 0.9179

### structure_type_fast: NO_GO

Passed 9/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9990 / 0.9977
- Latency p50/p95/p99: 0.1012 / 0.1278 / 0.1435 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.8989 / 0.9211

### structure_type_quantized: NO_GO

Passed 11/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: throughput_gte_10k, id_pair_auc_gte_090, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9989 / 0.9977
- Latency p50/p95/p99: 0.1494 / 0.2335 / 0.2560 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.9053 / 0.9203

### structure_type_quantized_256: NO_GO

Passed 10/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9988 / 0.9975
- Latency p50/p95/p99: 0.1484 / 0.2334 / 0.2540 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.8875 / 0.9130

### ngram_hash: NO_GO

Passed 8/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9524 / 0.8467
- Latency p50/p95/p99: 0.2105 / 0.2612 / 0.2833 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.9119 / 0.8772

### ngram_hash_multiscale: NO_GO

Passed 8/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9561 / 0.8740
- Latency p50/p95/p99: 0.1920 / 0.2550 / 0.2779 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.8972 / 0.8597

### pattern_free: NO_GO

Passed 8/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9897 / 0.9613
- Latency p50/p95/p99: 0.3851 / 0.5395 / 0.6412 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.9272 / 0.9381

### learned_weights: NO_GO

Passed 5/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: storage_lte_256, throughput_gte_10k, id_pair_auc_gte_090, id_pair_fixed_f1_gte_080, family_ood_pair_auc_gte_085, id_retrieval_precision_at_min_r_10_gte_080, id_retrieval_ndcg10_gte_090, id_retrieval_worst_domain_ndcg_gte_075, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, id_clustering_ari_gte_065, triage_id_f1_gte_085, triage_template_ood_f1_gte_075, family_novelty_auroc_gte_080.

- Mutation invariance mean/p10 cosine: 0.9946 / 0.9981
- Latency p50/p95/p99: 0.3732 / 0.6482 / 0.7100 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 0.2076 / 0.3813

### fisher: NO_GO

Passed 15/21 gates. Basis: one or more mandatory benchmark gates failed. Failed: family_ood_pair_auc_gte_085, template_ood_ndcg10_gte_075, template_ood_worst_domain_ndcg_gte_050, family_ood_ndcg10_gte_065, family_ood_worst_domain_ndcg_gte_050, triage_template_ood_f1_gte_075.

- Mutation invariance mean/p10 cosine: 0.9621 / 0.8997
- Latency p50/p95/p99: 0.0129 / 0.0151 / 0.0209 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 1.0000 / 1.0000

### generalized: SYNTHETIC_GO

Passed 21/21 gates. Basis: all synthetic benchmark gates passed; independent production validation is still required. Failed: none.

- Mutation invariance mean/p10 cosine: 0.9654 / 0.9215
- Latency p50/p95/p99: 0.0296 / 0.0340 / 0.0424 ms
- Balanced family-novelty AUROC/AP (50% baseline prevalence): 1.0000 / 1.0000

## Practical use-case assessment

The strongest candidate in this run is **generalized**, and its synthetic benchmark verdict is **SYNTHETIC_GO**. Synthetic bounded-known-format eligibility: **True**; this is not operational approval.

| Use case | Evidence from strongest candidate | Recommendation |
|---|---|---|
| Known-format similarity search | ID nDCG@10 0.9879, P@min(R,10) 0.9611 (6.0000x random P@10) | **Synthetic evidence only**; independent holdout required before any shadow pilot |
| New-template search | nDCG@10 0.8572, worst-domain 0.5184 | **Synthetic evidence only**; require independent holdout and abstention calibration; failures: none |
| Completely unseen-family search | nDCG@10 0.9590, worst-domain 0.6316 | **Synthetic evidence only**; require independent source-family holdout; failures: none |
| Offline format discovery/clustering | ID/family-OOD ARI 1.0000/1.0000 | **Promising** for analyst-reviewed grouping |
| Known-format routing/triage | ID macro-F1 1.0000; template-OOD macro-F1 0.8376 | **Synthetic evidence only**; require holdout coverage-risk evaluation |
| Drift/anomaly candidate ranking | Balanced family-novelty AUROC 1.0000, AP 1.0000 | **Offline analysis only** until independent holdout and real prevalence are measured |
| Clipboard/log/config organization | Mutation p10 cosine 0.9215; boundary errors 0 | **Synthetic evidence only**; independent holdout required before any shadow pilot |
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
- Throughput is the median of five single-process warm local runs without vector-database or network overhead.
- `SYNTHETIC_GO` means every gate on this fixed synthetic development benchmark passed; it is not a production guarantee. `NO_GO` means one or more mandatory benchmark gates failed.
- The adaptive visual-column and machine-delimiter experts were selected on this benchmark. Their gains require confirmation on a post-freeze independent generator/source-family holdout.
- Abstention and fallback recommendations are intentionally deferred because no threshold, coverage-risk, or fallback evaluation is included.

## Reproduce

```bash
PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_real_world.py --models random_projection simhash minhash tfidf_svd multiscale structure_type structure_type_fast structure_type_quantized structure_type_quantized_256 ngram_hash ngram_hash_multiscale pattern_free learned_weights fisher generalized --roots-per-family 30 --seed 20260729 --output results/final_full_benchmark
```

## Recommended next validation

1. Freeze the winner and run the independent post-freeze grouped holdout before any shadow pilot.
2. If the holdout passes, run a de-identified shadow pilot with source/template groups assigned before splitting.
3. Manually adjudicate high-impact false matches and calibrate abstention on real prevalence.
4. Add a vector-database integration benchmark at the intended corpus size before committing to latency or index-size targets.
