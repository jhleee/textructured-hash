"""Evaluation metrics for text structure hashing"""

import numpy as np
from typing import List, Tuple, Dict, Callable
from sklearn.metrics import roc_auc_score, f1_score
import time


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def evaluate(encoder: Callable, test_pairs: List[Dict]) -> Dict:
    """
    Evaluate encoder on test pairs

    Args:
        encoder: function(text) -> np.ndarray[float32]
        test_pairs: List of dicts with 'text1', 'text2', 'label' keys

    Returns:
        dict of metrics
    """
    predictions = []
    labels = []

    print(f"Evaluating on {len(test_pairs)} pairs...")

    for i, pair in enumerate(test_pairs):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(test_pairs)}")

        vec1 = encoder(pair['text1'])
        vec2 = encoder(pair['text2'])
        sim = cosine_similarity(vec1, vec2)
        predictions.append(sim)
        labels.append(pair['label'])

    predictions = np.array(predictions)
    labels = np.array(labels)

    # 1. AUC-ROC: classification performance
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.0

    # 2. Precision@K: accuracy in top K
    sorted_pairs = sorted(zip(predictions, labels), reverse=True)
    p_at_100 = sum(l for _, l in sorted_pairs[:min(100, len(sorted_pairs))]) / min(100, len(sorted_pairs))
    p_at_1000 = sum(l for _, l in sorted_pairs[:min(1000, len(sorted_pairs))]) / min(1000, len(sorted_pairs))

    # 3. Separation: distribution separation
    pos_sims = predictions[labels == 1.0]
    neg_sims = predictions[labels == 0.0]

    if len(pos_sims) > 0 and len(neg_sims) > 0:
        mean_pos = np.mean(pos_sims)
        mean_neg = np.mean(neg_sims)
        std_pos = np.std(pos_sims)
        std_neg = np.std(neg_sims)
        separation = (mean_pos - mean_neg) / (std_pos + std_neg + 1e-10)
    else:
        mean_pos = mean_neg = std_pos = std_neg = separation = 0.0

    # 4. Optimal Threshold F1
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.95, 0.05):
        preds = (predictions >= threshold).astype(int)
        try:
            f1 = f1_score(labels.astype(int), preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        except:
            pass

    return {
        "auc_roc": float(auc),
        "precision_at_100": float(p_at_100),
        "precision_at_1000": float(p_at_1000),
        "separation": float(separation),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "mean_pos_sim": float(mean_pos),
        "mean_neg_sim": float(mean_neg),
        "std_pos_sim": float(std_pos),
        "std_neg_sim": float(std_neg)
    }


def benchmark_efficiency(encoder: Callable, test_texts: List[str], n_iterations: int = 3) -> Dict:
    """
    Benchmark encoding and comparison efficiency

    Args:
        encoder: function(text) -> np.ndarray
        test_texts: List of test strings

    Returns:
        dict of efficiency metrics
    """
    print(f"\nBenchmarking efficiency on {len(test_texts)} texts...")

    # 1. Encoding Speed
    times = []
    for iteration in range(n_iterations):
        start = time.perf_counter()
        for text in test_texts:
            _ = encoder(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {iteration + 1}/{n_iterations}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    texts_per_sec = len(test_texts) / avg_time

    # 2. Memory per Vector
    sample_vec = encoder(test_texts[0])
    vec_bytes = sample_vec.nbytes
    vec_dims = len(sample_vec)

    # 3. Comparison Speed
    print(f"  Benchmarking comparison speed...")
    n_compare = min(1000, len(test_texts))
    vecs = [encoder(t) for t in test_texts[:n_compare]]

    start = time.perf_counter()
    n_comparisons = 0
    for i in range(len(vecs)):
        for j in range(i + 1, min(i + 100, len(vecs))):
            _ = np.dot(vecs[i], vecs[j])
            n_comparisons += 1
    comparison_time = time.perf_counter() - start

    comparisons_per_sec = n_comparisons / comparison_time if comparison_time > 0 else 0

    return {
        "encoding_speed": float(texts_per_sec),  # texts/sec
        "vector_bytes": int(vec_bytes),
        "vector_dimensions": int(vec_dims),
        "comparison_speed": float(comparisons_per_sec),  # comparisons/sec
        "total_encoding_time_ms": float(avg_time * 1000)
    }


def per_category_analysis(encoder: Callable, test_pairs: List[Dict]) -> Dict:
    """
    Analyze performance per category pair

    Args:
        encoder: function(text) -> np.ndarray
        test_pairs: List of test pairs

    Returns:
        dict mapping category pairs to metrics
    """
    from collections import defaultdict

    category_pairs = defaultdict(lambda: {'predictions': [], 'labels': []})

    for pair in test_pairs:
        cat1 = pair.get('category1', 'unknown')
        cat2 = pair.get('category2', 'unknown')
        cat_pair = tuple(sorted([cat1, cat2]))

        vec1 = encoder(pair['text1'])
        vec2 = encoder(pair['text2'])
        sim = cosine_similarity(vec1, vec2)

        category_pairs[cat_pair]['predictions'].append(sim)
        category_pairs[cat_pair]['labels'].append(pair['label'])

    # Compute metrics per category pair
    results = {}
    for cat_pair, data in category_pairs.items():
        preds = np.array(data['predictions'])
        labels = np.array(data['labels'])

        mean_sim = np.mean(preds)
        accuracy = np.mean((preds >= 0.5) == labels)

        results[f"{cat_pair[0]}↔{cat_pair[1]}"] = {
            'count': len(preds),
            'mean_similarity': float(mean_sim),
            'accuracy': float(accuracy)
        }

    return results
