"""Run leak-resistant real-world and OOD validation for text encoders."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate import get_encoder
from src.data.real_world_benchmark import (
    audit_records,
    boundary_cases,
    build_pairs,
    build_training_pairs,
    challenge_cases,
    generate_records,
    serializable_specs,
)

ALL_MODELS = (
    "random_projection",
    "simhash",
    "minhash",
    "tfidf_svd",
    "multiscale",
    "structure_type",
    "structure_type_fast",
    "structure_type_quantized",
    "structure_type_quantized_256",
    "ngram_hash",
    "ngram_hash_multiscale",
    "pattern_free",
    "learned_weights",
    "fisher",
    "generalized",
)
DEFAULT_MODELS = ALL_MODELS
CANONICAL_ROOTS_PER_FAMILY = 30
CANONICAL_SEED = 20260729


def _package_version(name: str) -> str:
    """Return an installed package version for benchmark provenance."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


def _deployment_vector(encoder, text: str) -> np.ndarray:
    """Return the representation actually counted by the storage gate."""
    if hasattr(encoder, "encode_int8"):
        return np.asarray(encoder.encode_int8(text))
    return np.asarray(encoder.encode(text))


def _encode_texts(encoder, texts: Sequence[str]) -> Tuple[np.ndarray, int]:
    raw_vectors = [_deployment_vector(encoder, text) for text in texts]
    vectors = _normalize(np.stack(raw_vectors))
    stored_bytes = int(raw_vectors[0].nbytes) if raw_vectors else 0
    return vectors, stored_bytes


def _best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best = (-1.0, 0.5)
    for threshold in candidates:
        score = f1_score(labels, scores >= threshold, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def _pair_scores(encoder, pairs: Sequence[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}

    def vector(text: str) -> np.ndarray:
        if text not in cache:
            value = np.asarray(_deployment_vector(encoder, text), dtype=np.float32)
            cache[text] = value / max(float(np.linalg.norm(value)), 1e-10)
        return cache[text]

    scores = [float(np.dot(vector(str(pair["text1"])), vector(str(pair["text2"])))) for pair in pairs]
    labels = [int(float(pair["label"])) for pair in pairs]
    return np.asarray(labels), np.asarray(scores)


def pair_metrics(encoder, pairs: Sequence[Dict[str, object]], threshold: float | None = None) -> Dict[str, float]:
    labels, scores = _pair_scores(encoder, pairs)
    if threshold is None:
        threshold = _best_threshold(labels, scores)
    positive, negative = scores[labels == 1], scores[labels == 0]
    return {
        "pairs": int(len(pairs)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "f1_at_val_threshold": float(f1_score(labels, scores >= threshold, zero_division=0)),
        "threshold": float(threshold),
        "mean_positive": float(np.mean(positive)),
        "mean_negative": float(np.mean(negative)),
        "separation": float((np.mean(positive) - np.mean(negative)) / (np.std(positive) + np.std(negative) + 1e-10)),
    }


def _slice_records(records: Sequence[Dict[str, object]], split: str, query_template: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    selected = [record for record in records if record["split"] == split]
    corpus = [record for record in selected if record["template"] == "standard"]
    queries = [record for record in selected if record["template"] == query_template]
    return corpus, queries


def retrieval_metrics(encoder, records: Sequence[Dict[str, object]], split: str, query_template: str) -> Dict[str, object]:
    corpus, queries = _slice_records(records, split, query_template)
    corpus_vectors, _ = _encode_texts(encoder, [str(item["text"]) for item in corpus])
    query_vectors, _ = _encode_texts(encoder, [str(item["text"]) for item in queries])
    scores = query_vectors @ corpus_vectors.T
    reciprocal_ranks: List[float] = []
    hits = {1: [], 5: [], 10: []}
    recalls = {1: [], 5: [], 10: []}
    precision10: List[float] = []
    precision_at_relevant_cutoff: List[float] = []
    ndcg10: List[float] = []
    random_hit10: List[float] = []
    per_domain_ndcg: Dict[str, List[float]] = defaultdict(list)
    for query_index, query in enumerate(queries):
        relevant = np.asarray([item["family"] == query["family"] for item in corpus], dtype=bool)
        order = np.argsort(-scores[query_index])
        ranked = relevant[order]
        positions = np.flatnonzero(ranked)
        rank = int(positions[0]) + 1 if len(positions) else len(corpus) + 1
        reciprocal_ranks.append(1.0 / rank)
        relevant_total = max(int(relevant.sum()), 1)
        for k in hits:
            hits[k].append(float(ranked[:k].any()))
            recalls[k].append(float(ranked[:k].sum() / relevant_total))
        top_k = min(10, len(ranked))
        precision10.append(float(ranked[:top_k].mean()))
        relevant_cutoff = min(relevant_total, top_k)
        precision_at_relevant_cutoff.append(float(ranked[:relevant_cutoff].mean()))
        gains = ranked[:top_k].astype(float)
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal_count = min(relevant_total, top_k)
        idcg = float(np.sum(discounts[:ideal_count]))
        query_ndcg = dcg / idcg if idcg else 0.0
        ndcg10.append(query_ndcg)
        per_domain_ndcg[str(query["domain"])].append(query_ndcg)
        non_relevant = len(corpus) - relevant_total
        if top_k > non_relevant:
            random_hit10.append(1.0)
        else:
            random_hit10.append(1.0 - math.comb(non_relevant, top_k) / math.comb(len(corpus), top_k))
    random_precision = float(np.mean([sum(item["family"] == query["family"] for item in corpus) / len(corpus) for query in queries]))
    return {
        "corpus": len(corpus),
        "queries": len(queries),
        "hit_at_1": float(np.mean(hits[1])),
        "hit_at_5": float(np.mean(hits[5])),
        "hit_at_10": float(np.mean(hits[10])),
        "random_hit_at_10": float(np.mean(random_hit10)),
        "precision_at_10": float(np.mean(precision10)),
        "precision_at_min_r_10": float(np.mean(precision_at_relevant_cutoff)),
        "random_precision_at_10": random_precision,
        "precision_lift_at_10": float(np.mean(precision10) / max(random_precision, 1e-10)),
        "recall_at_10": float(np.mean(recalls[10])),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg_at_10": float(np.mean(ndcg10)),
        "worst_domain_ndcg_at_10": float(min(np.mean(values) for values in per_domain_ndcg.values())),
        "per_domain_ndcg_at_10": {name: float(np.mean(values)) for name, values in sorted(per_domain_ndcg.items())},
    }


def clustering_metrics(encoder, records: Sequence[Dict[str, object]], split: str, template: str) -> Dict[str, float]:
    selected = [record for record in records if record["split"] == split and record["template"] == template]
    vectors, _ = _encode_texts(encoder, [str(item["text"]) for item in selected])
    labels = [str(item["family"]) for item in selected]
    unique = sorted(set(labels))
    label_ids = np.asarray([unique.index(label) for label in labels])
    predicted = KMeans(n_clusters=len(unique), random_state=42, n_init=10).fit_predict(vectors)
    return {
        "samples": len(selected),
        "clusters": len(unique),
        "ari": float(adjusted_rand_score(label_ids, predicted)),
        "nmi": float(normalized_mutual_info_score(label_ids, predicted)),
    }


def _centroids(encoder, records: Sequence[Dict[str, object]]) -> Tuple[List[str], np.ndarray]:
    selected = [record for record in records if record["split"] == "train"]
    labels = sorted({str(record["family"]) for record in selected})
    vectors, _ = _encode_texts(encoder, [str(record["text"]) for record in selected])
    centroids = []
    for label in labels:
        centroid = vectors[[record["family"] == label for record in selected]].mean(axis=0)
        centroids.append(centroid / max(float(np.linalg.norm(centroid)), 1e-10))
    return labels, np.stack(centroids)


def triage_and_anomaly_metrics(encoder, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    labels, centroids = _centroids(encoder, records)

    def selected_for(split: str, template: str, per_family: int | None = None) -> List[Dict[str, object]]:
        selected = [record for record in records if record["split"] == split and record["template"] == template]
        if per_family is None:
            return selected
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for record in selected:
            grouped[str(record["family"])].append(record)
        return [record for family in sorted(grouped) for record in grouped[family][:per_family]]

    def scores_for(selected: Sequence[Dict[str, object]]) -> np.ndarray:
        vectors, _ = _encode_texts(encoder, [str(record["text"]) for record in selected])
        return vectors @ centroids.T

    known = selected_for("test", "alternate")
    template_ood = selected_for("test", "ood")
    family_known = selected_for("test", "standard")
    family_unknown = selected_for("family_ood", "standard", per_family=max(1, len(family_known) // 10))
    known_scores = scores_for(known)
    template_scores = scores_for(template_ood)
    family_known_scores = scores_for(family_known)
    family_unknown_scores = scores_for(family_unknown)

    def classification(selected: Sequence[Dict[str, object]], scores: np.ndarray) -> float:
        true = [labels.index(str(record["family"])) for record in selected]
        predicted = np.argmax(scores, axis=1)
        return float(f1_score(true, predicted, average="macro", zero_division=0))

    family_known_confidence = np.max(family_known_scores, axis=1)
    family_unknown_confidence = np.max(family_unknown_scores, axis=1)
    family_labels = np.concatenate([np.zeros(len(family_known_confidence)), np.ones(len(family_unknown_confidence))])
    family_scores = np.concatenate([1.0 - family_known_confidence, 1.0 - family_unknown_confidence])

    template_known_confidence = np.max(family_known_scores, axis=1)
    template_ood_confidence = np.max(template_scores, axis=1)
    template_labels = np.concatenate([np.zeros(len(template_known_confidence)), np.ones(len(template_ood_confidence))])
    template_novelty = np.concatenate([1.0 - template_known_confidence, 1.0 - template_ood_confidence])
    return {
        "triage_macro_f1_id": classification(known, known_scores),
        "triage_macro_f1_template_ood": classification(template_ood, template_scores),
        "family_novelty_samples_known": len(family_known_confidence),
        "family_novelty_samples_unknown": len(family_unknown_confidence),
        "family_novelty_positive_prevalence": float(np.mean(family_labels)),
        "family_novelty_auroc": float(roc_auc_score(family_labels, family_scores)),
        "family_novelty_average_precision": float(average_precision_score(family_labels, family_scores)),
        "template_shift_auroc": float(roc_auc_score(template_labels, template_novelty)),
        "template_shift_average_precision": float(average_precision_score(template_labels, template_novelty)),
        "mean_known_confidence": float(np.mean(family_known_confidence)),
        "mean_unknown_confidence": float(np.mean(family_unknown_confidence)),
    }


def mutation_metrics(encoder, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_mutation: Dict[str, List[float]] = defaultdict(list)
    errors: List[str] = []
    for case in challenge_cases(records):
        try:
            vectors, _ = _encode_texts(encoder, [str(case["original"]), str(case["mutated"])])
            by_mutation[str(case["mutation"])].append(float(np.dot(vectors[0], vectors[1])))
        except Exception as exc:  # robustness result, not silent benchmark failure
            errors.append(f"{case['family']}:{case['mutation']}:{type(exc).__name__}")
    return {
        "cases": sum(len(values) for values in by_mutation.values()) + len(errors),
        "errors": len(errors),
        "error_examples": errors[:5],
        "mean_similarity": float(np.mean([score for values in by_mutation.values() for score in values])),
        "p10_similarity": float(np.percentile([score for values in by_mutation.values() for score in values], 10)),
        "by_mutation": {name: float(np.mean(values)) for name, values in sorted(by_mutation.items())},
    }


def performance_metrics(encoder, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    texts = [str(record["text"]) for record in records if record["split"] == "test"][:160]
    for text in texts:
        _deployment_vector(encoder, text)
    latencies = []
    throughput_runs = []
    for _ in range(5):
        start = time.perf_counter()
        for text in texts:
            item_start = time.perf_counter()
            _deployment_vector(encoder, text)
            latencies.append((time.perf_counter() - item_start) * 1000)
        elapsed = time.perf_counter() - start
        throughput_runs.append(float(len(texts) / elapsed))
    _, stored_bytes = _encode_texts(encoder, texts[:1])
    repeated_first = _deployment_vector(encoder, texts[0])
    repeated_second = _deployment_vector(encoder, texts[0])
    repeat_encode_bit_identical = bool(np.array_equal(repeated_first, repeated_second))

    boundary = []
    for case in boundary_cases():
        try:
            case_start = time.perf_counter()
            vector = _deployment_vector(encoder, str(case["text"]))
            duration = (time.perf_counter() - case_start) * 1000
            valid = bool(np.isfinite(vector).all())
            boundary.append({"name": case["name"], "length": case["length"], "latency_ms": duration, "valid": valid, "error": None})
        except Exception as exc:
            boundary.append({"name": case["name"], "length": case["length"], "latency_ms": None, "valid": False, "error": type(exc).__name__})
    return {
        "texts": len(texts),
        "throughput_texts_per_sec": float(np.median(throughput_runs)),
        "throughput_repetitions": len(throughput_runs),
        "throughput_runs_texts_per_sec": throughput_runs,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
        "stored_vector_bytes": stored_bytes,
        "evaluation_vector_dtype": str(repeated_first.dtype),
        "float_vector_bytes": int(np.asarray(encoder.encode(texts[0])).nbytes),
        "repeat_encode_bit_identical": repeat_encode_bit_identical,
        "boundary_errors": sum(not item["valid"] for item in boundary),
        "boundary_cases": boundary,
    }


def verdict(metrics: Dict[str, object], audit: Dict[str, object]) -> Dict[str, object]:
    gates = {
        "challenge_isolation": bool(audit["leakage_free"]),
        "repeat_encode_bit_identical": metrics["performance"]["repeat_encode_bit_identical"],
        "boundary_safe": metrics["performance"]["boundary_errors"] == 0,
        "mutation_safe": metrics["mutations"]["errors"] == 0,
        "storage_lte_256": metrics["performance"]["stored_vector_bytes"] <= 256,
        "throughput_gte_10k": metrics["performance"]["throughput_texts_per_sec"] >= 10000,
        "id_pair_auc_gte_090": metrics["pair_id"]["roc_auc"] >= 0.90,
        "id_pair_fixed_f1_gte_080": metrics["pair_id"]["f1_at_val_threshold"] >= 0.80,
        "family_ood_pair_auc_gte_085": metrics["pair_family_ood"]["roc_auc"] >= 0.85,
        "id_retrieval_precision_at_min_r_10_gte_080": metrics["retrieval_id"]["precision_at_min_r_10"] >= 0.80,
        "id_retrieval_ndcg10_gte_090": metrics["retrieval_id"]["ndcg_at_10"] >= 0.90,
        "id_retrieval_worst_domain_ndcg_gte_075": metrics["retrieval_id"]["worst_domain_ndcg_at_10"] >= 0.75,
        "template_ood_ndcg10_gte_075": metrics["retrieval_template_ood"]["ndcg_at_10"] >= 0.75,
        "template_ood_worst_domain_ndcg_gte_050": metrics["retrieval_template_ood"]["worst_domain_ndcg_at_10"] >= 0.50,
        "family_ood_ndcg10_gte_065": metrics["retrieval_family_ood"]["ndcg_at_10"] >= 0.65,
        "family_ood_worst_domain_ndcg_gte_050": metrics["retrieval_family_ood"]["worst_domain_ndcg_at_10"] >= 0.50,
        "id_clustering_ari_gte_065": metrics["clustering_id"]["ari"] >= 0.65,
        "family_ood_clustering_ari_gte_050": metrics["clustering_family_ood"]["ari"] >= 0.50,
        "triage_id_f1_gte_085": metrics["operations"]["triage_macro_f1_id"] >= 0.85,
        "triage_template_ood_f1_gte_075": metrics["operations"]["triage_macro_f1_template_ood"] >= 0.75,
        "family_novelty_auroc_gte_080": metrics["operations"]["family_novelty_auroc"] >= 0.80,
    }
    reliability_names = ("challenge_isolation", "repeat_encode_bit_identical", "boundary_safe", "mutation_safe")
    efficiency_names = ("storage_lte_256", "throughput_gte_10k")
    bounded_pilot_names = (
        *reliability_names,
        *efficiency_names,
        "id_pair_auc_gte_090",
        "id_retrieval_precision_at_min_r_10_gte_080",
        "id_retrieval_ndcg10_gte_090",
        "id_retrieval_worst_domain_ndcg_gte_075",
        "id_clustering_ari_gte_065",
        "triage_id_f1_gte_085",
    )
    bounded_pilot_eligible = all(gates[name] for name in bounded_pilot_names)
    if all(gates.values()):
        decision = "SYNTHETIC_GO"
        basis = "all synthetic benchmark gates passed; independent production validation is still required"
    else:
        decision = "NO_GO"
        basis = "one or more mandatory benchmark gates failed"
    return {
        "decision": decision,
        "decision_basis": basis,
        "bounded_known_format_pilot_eligible": bounded_pilot_eligible,
        "passed": sum(gates.values()),
        "total": len(gates),
        "gates": gates,
    }


def evaluate_model(model_name: str, records: Sequence[Dict[str, object]], train_pairs: Sequence[Dict[str, object]], audit: Dict[str, object]) -> Dict[str, object]:
    print(f"\n{'=' * 72}\n{model_name}\n{'=' * 72}")
    model_start = time.perf_counter()
    training_start = time.perf_counter()
    encoder = get_encoder(model_name, list(train_pairs))
    training_seconds = time.perf_counter() - training_start
    val_pairs = build_pairs(records, "val", per_family=30)
    _, val_scores = _pair_scores(encoder, val_pairs)
    val_labels = np.asarray([int(float(pair["label"])) for pair in val_pairs])
    threshold = _best_threshold(val_labels, val_scores)

    metrics: Dict[str, object] = {
        "pair_validation": pair_metrics(encoder, val_pairs, threshold),
        "pair_id": pair_metrics(encoder, build_pairs(records, "test", per_family=30), threshold),
        "pair_template_ood": pair_metrics(encoder, build_pairs(records, "test", per_family=30, target_template="ood"), threshold),
        "pair_family_ood": pair_metrics(encoder, build_pairs(records, "family_ood", per_family=30, target_template="ood"), threshold),
        "retrieval_id": retrieval_metrics(encoder, records, "test", "alternate"),
        "retrieval_template_ood": retrieval_metrics(encoder, records, "test", "ood"),
        "retrieval_family_ood": retrieval_metrics(encoder, records, "family_ood", "ood"),
        "clustering_id": clustering_metrics(encoder, records, "test", "alternate"),
        "clustering_template_ood": clustering_metrics(encoder, records, "test", "ood"),
        "clustering_family_ood": clustering_metrics(encoder, records, "family_ood", "ood"),
        "operations": triage_and_anomaly_metrics(encoder, records),
        "mutations": mutation_metrics(encoder, records),
        "performance": performance_metrics(encoder, records),
    }
    metrics["verdict"] = verdict(metrics, audit)
    metrics["runtime"] = {
        "training_seconds": float(training_seconds),
        "total_seconds": float(time.perf_counter() - model_start),
    }
    print(f"Decision: {metrics['verdict']['decision']} ({metrics['verdict']['passed']}/{metrics['verdict']['total']} gates)")
    return metrics


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def rank_models(results: Dict[str, Dict[str, object]]) -> List[str]:
    """Rank candidates by predeclared gates, then OOD quality and efficiency."""
    decision_rank = {"SYNTHETIC_GO": 2, "LIMITED_GO": 1, "NO_GO": 0}

    def score(model_name: str) -> Tuple[object, ...]:
        metrics = results[model_name]
        return (
            decision_rank[metrics["verdict"]["decision"]],
            metrics["verdict"]["passed"],
            metrics["retrieval_family_ood"]["worst_domain_ndcg_at_10"],
            metrics["retrieval_template_ood"]["worst_domain_ndcg_at_10"],
            metrics["pair_family_ood"]["roc_auc"],
            metrics["retrieval_id"]["ndcg_at_10"],
            metrics["performance"]["throughput_texts_per_sec"],
        )

    return sorted(results, key=score, reverse=True)


def write_report(output_dir: Path, results: Dict[str, Dict[str, object]], audit: Dict[str, object], config: Dict[str, object]) -> None:
    ranking = rank_models(results)
    winner = ranking[0]
    is_full_benchmark = config["benchmark_scope"] == "repository_full"
    title = "Final Full Benchmark Results" if is_full_benchmark else "Partial Candidate Benchmark Results"
    scope_statement = (
        "This benchmark compares every implemented encoder under one deterministic, leak-resistant protocol."
        if is_full_benchmark
        else "This is a partial comparison of selected encoders; it must not be used to claim a repository-wide winner."
    )
    winner_statement = (
        f"**Final synthetic benchmark winner: `{winner}`.**"
        if is_full_benchmark
        else f"**Strongest candidate in this partial run: `{winner}`.**"
    )
    lines = [
        f"# {title}",
        "",
        f"{scope_statement} It uses privacy-safe synthetic operational data and is not a substitute for an independent post-freeze holdout or a de-identified production pilot.",
        "",
        "## Dataset and integrity",
        "",
        f"- Records: {audit['records']}; domains: {audit['domains']}; format families: {audit['families']}",
        f"- Root/exact-text leakage across splits: {'none' if audit['root_and_exact_leakage_free'] else 'detected'}",
        f"- OOD family/renderer isolation: {'verified' if audit['challenge_isolation'] else 'failed'}",
        "- ID train/test intentionally share renderer families while roots and values remain disjoint",
        f"- Deterministic manifest SHA-256: `{audit['manifest_sha256']}`",
        "- Family-OOD policy: one of two families per domain is entirely excluded from training",
        "- Pair threshold policy: selected on validation once and frozen for test/OOD",
        f"- Candidate coverage: {len(results)} selected encoders ({', '.join(results)})",
        f"- Benchmark scope: `{config['benchmark_scope']}`",
        "- Deployment representation: `encode_int8` when implemented, otherwise float32 `encode` output",
        "",
        "## Results summary",
        "",
        "| Model | Cross-template Pair AUC ID / template OOD / family OOD | Retrieval nDCG@10 ID / template / family | Clustering ARI ID / family | Triage F1 ID / template | Family novelty AUROC | Speed/s | Bytes/dtype | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, metrics in results.items():
        lines.append(
            "| {model} | {p1}/{p2}/{p3} | {r1}/{r2}/{r3} | {c1}/{c2} | {t} | {u} | {s:.0f} | {b} | **{v}** |".format(
                model=model,
                p1=_fmt(metrics["pair_id"]["roc_auc"]), p2=_fmt(metrics["pair_template_ood"]["roc_auc"]), p3=_fmt(metrics["pair_family_ood"]["roc_auc"]),
                r1=_fmt(metrics["retrieval_id"]["ndcg_at_10"]), r2=_fmt(metrics["retrieval_template_ood"]["ndcg_at_10"]), r3=_fmt(metrics["retrieval_family_ood"]["ndcg_at_10"]),
                c1=_fmt(metrics["clustering_id"]["ari"]), c2=_fmt(metrics["clustering_family_ood"]["ari"]),
                t=f"{_fmt(metrics['operations']['triage_macro_f1_id'])}/{_fmt(metrics['operations']['triage_macro_f1_template_ood'])}", u=_fmt(metrics["operations"]["family_novelty_auroc"]),
                s=metrics["performance"]["throughput_texts_per_sec"], b=f"{metrics['performance']['stored_vector_bytes']}/{metrics['performance']['evaluation_vector_dtype']}", v=metrics["verdict"]["decision"],
            )
        )
    lines.extend([
        "",
        "## Winner and ranking" if is_full_benchmark else "## Selected-candidate ranking",
        "",
        winner_statement,
        "",
        "Ranking policy (declared in code): verdict class, gates passed, family-OOD worst-domain nDCG@10, template-OOD worst-domain nDCG@10, family-OOD pair AUC, ID nDCG@10, then throughput. Test thresholds are never retuned.",
        "",
        "| Rank | Model | Decision | Gates | Family OOD worst-domain nDCG@10 | Template OOD worst-domain nDCG@10 | Family OOD pair AUC | Throughput/s |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for index, model in enumerate(ranking, start=1):
        metrics = results[model]
        lines.append(
            f"| {index} | {model} | {metrics['verdict']['decision']} | "
            f"{metrics['verdict']['passed']}/{metrics['verdict']['total']} | "
            f"{_fmt(metrics['retrieval_family_ood']['worst_domain_ndcg_at_10'])} | "
            f"{_fmt(metrics['retrieval_template_ood']['worst_domain_ndcg_at_10'])} | "
            f"{_fmt(metrics['pair_family_ood']['roc_auc'])} | "
            f"{metrics['performance']['throughput_texts_per_sec']:.0f} |"
        )
    lines.extend(["", "## Gate details", ""])
    for model, metrics in results.items():
        verdict_data = metrics["verdict"]
        failed = [name for name, passed in verdict_data["gates"].items() if not passed]
        lines.extend([
            f"### {model}: {verdict_data['decision']}",
            "",
            f"Passed {verdict_data['passed']}/{verdict_data['total']} gates. Basis: {verdict_data['decision_basis']}. Failed: {', '.join(failed) if failed else 'none'}.",
            "",
            f"- Mutation invariance mean/p10 cosine: {_fmt(metrics['mutations']['mean_similarity'])} / {_fmt(metrics['mutations']['p10_similarity'])}",
            f"- Latency p50/p95/p99: {_fmt(metrics['performance']['latency_ms_p50'])} / {_fmt(metrics['performance']['latency_ms_p95'])} / {_fmt(metrics['performance']['latency_ms_p99'])} ms",
            f"- Balanced family-novelty AUROC/AP (50% baseline prevalence): {_fmt(metrics['operations']['family_novelty_auroc'])} / {_fmt(metrics['operations']['family_novelty_average_precision'])}",
            "",
        ])
    best_model = winner
    best = results[best_model]
    template_failures = [name for name, value in best["retrieval_template_ood"]["per_domain_ndcg_at_10"].items() if value < 0.5]
    family_failures = [name for name, value in best["retrieval_family_ood"]["per_domain_ndcg_at_10"].items() if value < 0.5]
    template_ready = best["verdict"]["gates"]["template_ood_ndcg10_gte_075"] and best["verdict"]["gates"]["template_ood_worst_domain_ndcg_gte_050"]
    family_ready = best["verdict"]["gates"]["family_ood_ndcg10_gte_065"] and best["verdict"]["gates"]["family_ood_worst_domain_ndcg_gte_050"]
    routing_ready = best["verdict"]["gates"]["triage_template_ood_f1_gte_075"]
    template_recommendation = "**Synthetic evidence only**; require independent holdout and abstention calibration" if template_ready else "**Human-assisted only**"
    family_recommendation = "**Synthetic evidence only**; require independent source-family holdout" if family_ready else "**Do not automate**"
    routing_recommendation = "**Synthetic evidence only**; require holdout coverage-risk evaluation" if routing_ready else "**Known templates only**, with abstention and fallback"
    models_display = " ".join(str(model) for model in config["models"])
    output_display = output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir
    lines.extend([
        "## Practical use-case assessment",
        "",
        f"The strongest candidate in this run is **{best_model}**, and its synthetic benchmark verdict is **{best['verdict']['decision']}**. Synthetic bounded-known-format eligibility: **{best['verdict']['bounded_known_format_pilot_eligible']}**; this is not operational approval.",
        "",
        "| Use case | Evidence from strongest candidate | Recommendation |",
        "|---|---|---|",
        f"| Known-format similarity search | ID nDCG@10 {_fmt(best['retrieval_id']['ndcg_at_10'])}, P@min(R,10) {_fmt(best['retrieval_id']['precision_at_min_r_10'])} ({_fmt(best['retrieval_id']['precision_lift_at_10'])}x random P@10) | **Synthetic evidence only**; independent holdout required before any shadow pilot |",
        f"| New-template search | nDCG@10 {_fmt(best['retrieval_template_ood']['ndcg_at_10'])}, worst-domain {_fmt(best['retrieval_template_ood']['worst_domain_ndcg_at_10'])} | {template_recommendation}; failures: {', '.join(template_failures) or 'none'} |",
        f"| Completely unseen-family search | nDCG@10 {_fmt(best['retrieval_family_ood']['ndcg_at_10'])}, worst-domain {_fmt(best['retrieval_family_ood']['worst_domain_ndcg_at_10'])} | {family_recommendation}; failures: {', '.join(family_failures) or 'none'} |",
        f"| Offline format discovery/clustering | ID/family-OOD ARI {_fmt(best['clustering_id']['ari'])}/{_fmt(best['clustering_family_ood']['ari'])} | **Promising** for analyst-reviewed grouping |",
        f"| Known-format routing/triage | ID macro-F1 {_fmt(best['operations']['triage_macro_f1_id'])}; template-OOD macro-F1 {_fmt(best['operations']['triage_macro_f1_template_ood'])} | {routing_recommendation} |",
        f"| Drift/anomaly candidate ranking | Balanced family-novelty AUROC {_fmt(best['operations']['family_novelty_auroc'])}, AP {_fmt(best['operations']['family_novelty_average_precision'])} | **Offline analysis only** until independent holdout and real prevalence are measured |",
        f"| Clipboard/log/config organization | Mutation p10 cosine {_fmt(best['mutations']['p10_similarity'])}; boundary errors {best['performance']['boundary_errors']} | **Synthetic evidence only**; independent holdout required before any shadow pilot |",
        "| Semantic or entity-level retrieval | Benchmark labels structure, not meaning | **Not supported** by this research |",
        "",
        "### Remaining risks and transfer profile",
        "",
        f"- Cross-template pair AUC/F1 for ID, template-OOD, and family-OOD: {_fmt(best['pair_id']['roc_auc'])}/{_fmt(best['pair_id']['f1_at_val_threshold'])}, {_fmt(best['pair_template_ood']['roc_auc'])}/{_fmt(best['pair_template_ood']['f1_at_val_threshold'])}, {_fmt(best['pair_family_ood']['roc_auc'])}/{_fmt(best['pair_family_ood']['f1_at_val_threshold'])}.",
        f"- Average OOD scores hide complete domain failures. Template-OOD failed in {', '.join(template_failures) or 'no domain'}; family-OOD failed in {', '.join(family_failures) or 'no domain'}.",
        "- Clustering uses the true number of families as K; it supports analyst-reviewed grouping, not automatic discovery of cluster count.",
        "- ID evaluation deliberately reuses renderer families with disjoint roots. Only template-OOD and family-OOD results support generalization claims.",
        "- Retrieval gates use precision/nDCG and random baselines because Hit@10 is high even under random ranking when 10% of the corpus is relevant.",
        "- The original pair-only benchmark would have missed these retrieval and routing failures.",
        "",
        "## Interpretation limits",
        "",
        "- Synthetic realism removes privacy/licensing risk and supports exact reproducibility, but cannot reproduce an organization's true class mix, malformed inputs, or drift.",
        "- Results are point estimates without grouped bootstrap confidence intervals, and generator/source families are not independently held out.",
        "- Retrieval relevance is format-family relevance. A product seeking semantic, entity, or exact-near-duplicate relevance needs a separate label contract.",
        "- Quality metrics use the same deployed representation counted by the storage gate (true int8 where available; otherwise float32).",
        "- Throughput is the median of five single-process warm local runs without vector-database or network overhead.",
        "- `SYNTHETIC_GO` means every gate on this fixed synthetic development benchmark passed; it is not a production guarantee. `NO_GO` means one or more mandatory benchmark gates failed.",
        "- The adaptive visual-column and machine-delimiter experts were selected on this benchmark. Their gains require confirmation on a post-freeze independent generator/source-family holdout.",
        "- Abstention and fallback recommendations are intentionally deferred because no threshold, coverage-risk, or fallback evaluation is included.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_real_world.py --models {models_display} --roots-per-family {config['roots_per_family']} --seed {config['seed']} --output {output_display}",
        "```",
        "",
        "## Recommended next validation",
        "",
        "1. Freeze the winner and run the independent post-freeze grouped holdout before any shadow pilot.",
        "2. If the holdout passes, run a de-identified shadow pilot with source/template groups assigned before splitting.",
        "3. Manually adjudicate high-impact false matches and calibrate abstention on real prevalence.",
        "4. Add a vector-database integration benchmark at the intended corpus size before committing to latency or index-size targets.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": config,
                "audit": audit,
                "winner": winner,
                "winner_scope": config["benchmark_scope"],
                "ranking": ranking,
                "models": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--roots-per-family", type=int, default=CANONICAL_ROOTS_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "extended_real_world")
    args = parser.parse_args()

    records = generate_records(args.roots_per_family, args.seed)
    repeated = generate_records(args.roots_per_family, args.seed)
    audit = audit_records(records)
    repeated_audit = audit_records(repeated)
    audit["deterministic_regeneration"] = audit["manifest_sha256"] == repeated_audit["manifest_sha256"]
    if not audit["leakage_free"] or not audit["deterministic_regeneration"]:
        raise RuntimeError(f"Dataset integrity gate failed: {audit}")

    train_pairs = build_training_pairs(records, args.seed)
    benchmark_scope = (
        "repository_full"
        if tuple(args.models) == ALL_MODELS
        and args.roots_per_family == CANONICAL_ROOTS_PER_FAMILY
        and args.seed == CANONICAL_SEED
        else "partial"
    )
    config = {
        "seed": args.seed,
        "roots_per_family": args.roots_per_family,
        "models": args.models,
        "benchmark_scope": benchmark_scope,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "scikit_learn": _package_version("scikit-learn"),
        "numba": _package_version("numba"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not-set"),
        "platform": platform.platform(),
        "family_specs": serializable_specs(),
        "validation_threshold_frozen": True,
        "winner_selection_policy": [
            "verdict",
            "gates_passed",
            "family_ood_worst_domain_ndcg_at_10",
            "template_ood_worst_domain_ndcg_at_10",
            "family_ood_pair_auc",
            "id_ndcg_at_10",
            "throughput_texts_per_sec",
        ],
    }
    benchmark_start = time.perf_counter()
    results = {model: evaluate_model(model, records, train_pairs, audit) for model in args.models}
    config["total_benchmark_seconds"] = float(time.perf_counter() - benchmark_start)
    args.output.mkdir(parents=True, exist_ok=True)
    write_report(args.output, results, audit, config)
    print(f"\nWrote {args.output / 'metrics.json'}")
    print(f"Wrote {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
