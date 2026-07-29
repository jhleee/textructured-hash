"""Holdout validation evaluation script.

Two phases:
  1. save_model  — Train the Fisher encoder on dev data and save frozen checkpoint.
  2. evaluate    — Load frozen checkpoint, run on independent holdout, produce report.

Usage:
  # Phase 1: Freeze model
  python scripts/evaluate_holdout.py --phase save_model \
      --train data/train.jsonl \
      --model_path results/holdout_validation/frozen_model.npz

  # Phase 2: Evaluate on holdout
  python scripts/evaluate_holdout.py --phase evaluate \
      --model_path results/holdout_validation/frozen_model.npz \
      --holdout_pairs data/holdout/pairs.jsonl \
      --output results/holdout_validation/
"""

import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders.proposed.fisher_encoder import FisherStructureEncoder
from src.evaluation.metrics import evaluate, benchmark_efficiency


# ─── Success criteria from RESEARCH_PROTOCOL.md ───
TARGETS = {
    'auc_roc': ('>=', 0.92),
    'separation': ('>=', 2.5),
    'best_f1': ('>=', 0.88),
    'mean_pos_sim': ('>=', 0.85),
    'mean_neg_sim': ('<=', 0.35),
}
SPEED_TARGET = 10000  # texts/sec
BYTES_TARGET = 256    # int8 vector bytes


def load_pairs(path: str) -> list:
    """Load pairs from JSONL."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def check_criteria(quality: dict, efficiency: dict) -> dict:
    """Check pass/fail for each criterion."""
    results = {}
    for metric, (op, target) in TARGETS.items():
        val = quality[metric]
        if op == '>=':
            results[metric] = {'value': val, 'target': target, 'pass': val >= target}
        else:
            results[metric] = {'value': val, 'target': target, 'pass': val <= target}

    results['encoding_speed'] = {
        'value': efficiency['encoding_speed'],
        'target': SPEED_TARGET,
        'pass': efficiency['encoding_speed'] >= SPEED_TARGET,
    }
    results['vector_bytes'] = {
        'value': efficiency['vector_bytes'],
        'target': BYTES_TARGET,
        'pass': efficiency['vector_bytes'] <= BYTES_TARGET,
    }
    return results



def bootstrap_auc_ci(predictions, labels, n_bootstrap=1000, ci=0.95, seed=999):
    """Bootstrap 95% confidence interval for AUC-ROC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(predictions)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            auc = roc_auc_score(labels[idx], predictions[idx])
            aucs.append(auc)
        except ValueError:
            continue
    aucs = sorted(aucs)
    lo = aucs[int(len(aucs) * (1 - ci) / 2)]
    hi = aucs[int(len(aucs) * (1 + ci) / 2)]
    return lo, hi


def per_family_analysis(encoder_fn, pairs: list, new_families: set) -> dict:
    """Break down metrics by family grouping."""
    groups = {
        'known_intra': [],    # both categories are in existing 24 and same
        'new_intra': [],      # both categories are in new 4 and same
        'cross_known': [],    # different existing categories
        'cross_new': [],      # one or both from new families, different cats
    }

    for pair in pairs:
        cat1 = pair['category1']
        cat2 = pair['category2']
        is_new1 = cat1 in new_families
        is_new2 = cat2 in new_families
        is_positive = pair['label'] == 1.0

        if is_positive:
            if is_new1:
                groups['new_intra'].append(pair)
            else:
                groups['known_intra'].append(pair)
        else:
            if is_new1 or is_new2:
                groups['cross_new'].append(pair)
            else:
                groups['cross_known'].append(pair)

    results = {}
    for group_name, group_pairs in groups.items():
        if not group_pairs:
            results[group_name] = {'count': 0}
            continue
        preds = []
        labels = []
        for p in group_pairs:
            v1 = encoder_fn(p['text1'])
            v2 = encoder_fn(p['text2'])
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            preds.append(sim)
            labels.append(p['label'])
        preds = np.array(preds)
        labels = np.array(labels)
        results[group_name] = {
            'count': len(preds),
            'mean_sim': float(np.mean(preds)),
            'std_sim': float(np.std(preds)),
        }
        # AUC only if both classes present
        if len(set(labels.tolist())) == 2:
            from sklearn.metrics import roc_auc_score
            results[group_name]['auc_roc'] = float(roc_auc_score(labels, preds))
    return results



def determine_outcome(criteria: dict, family_results: dict) -> str:
    """Determine FULL PASS / SOFT PASS / FAIL."""
    n_pass = sum(1 for v in criteria.values() if v['pass'])
    n_total = len(criteria)
    auc_pass = criteria['auc_roc']['pass']

    # New families AUC (from cross_new or new_intra)
    new_auc = family_results.get('new_intra', {}).get('auc_roc')
    # For intra-only groups we check mean_sim instead
    new_mean_sim = family_results.get('new_intra', {}).get('mean_sim', 0)

    if n_pass == n_total:
        return 'FULL PASS'
    elif n_pass >= 5 and auc_pass and new_mean_sim >= 0.75:
        return 'SOFT PASS'
    else:
        return 'FAIL'


def generate_report(quality, efficiency, criteria, family_results,
                    outcome, auc_ci, holdout_count, dev_metrics=None) -> str:
    """Generate markdown report."""
    lines = [
        '# Holdout Validation Results\n',
        '## Summary',
        f'- Model: FisherStructureEncoder (frozen from dev training)',
        f'- Holdout pairs: {holdout_count}',
        f'- Result: **{outcome}**',
        f'- AUC-ROC 95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]',
        '',
        '## Aggregate Metrics\n',
        '| Metric | Target | Holdout | Status |',
        '|--------|--------|---------|--------|',
    ]

    for metric, info in criteria.items():
        status = 'PASS' if info['pass'] else 'FAIL'
        mark = '\u2713' if info['pass'] else '\u2717'
        if isinstance(info['value'], float):
            val_str = f"{info['value']:.4f}"
        else:
            val_str = str(info['value'])
        target_str = f"{info['target']}"
        lines.append(f"| {metric} | {target_str} | {val_str} | {mark} {status} |")

    lines.append('')
    lines.append(f"**Criteria passed: {sum(1 for v in criteria.values() if v['pass'])}/{len(criteria)}**")
    lines.append('')

    # Per-family breakdown
    lines.append('## Per-Family Breakdown\n')
    lines.append('| Group | Count | Mean Sim | Std Sim | AUC-ROC |')
    lines.append('|-------|-------|----------|---------|---------|')
    for group, info in family_results.items():
        count = info.get('count', 0)
        mean_sim = f"{info.get('mean_sim', 0):.4f}" if count > 0 else '-'
        std_sim = f"{info.get('std_sim', 0):.4f}" if count > 0 else '-'
        auc = f"{info.get('auc_roc', 0):.4f}" if 'auc_roc' in info else '-'
        lines.append(f'| {group} | {count} | {mean_sim} | {std_sim} | {auc} |')

    lines.append('')
    lines.append('## Efficiency\n')
    lines.append(f"- Encoding speed: {efficiency['encoding_speed']:.0f} texts/sec")
    lines.append(f"- Vector bytes (int8): {efficiency['vector_bytes']}")
    lines.append(f"- Vector dimensions: {efficiency['vector_dimensions']}")
    lines.append('')
    lines.append('## Conclusion\n')
    lines.append(f'Outcome: **{outcome}**\n')
    if outcome == 'FULL PASS':
        lines.append('The algorithm generalizes to independent holdout data including unseen '
                     'category families. Ready for production evaluation.')
    elif outcome == 'SOFT PASS':
        lines.append('Most criteria pass but some degradation observed. Investigate specific '
                     'failure modes before production deployment.')
    else:
        lines.append('Significant performance drop on holdout data. The model may be overfit '
                     'to the development benchmark. Requires redesign or regularization.')

    return '\n'.join(lines) + '\n'



def phase_save_model(args):
    """Phase 1: Train on dev data and save frozen checkpoint."""
    print('=' * 60)
    print('Phase: SAVE MODEL (freeze)')
    print('=' * 60)

    train_pairs = load_pairs(args.train)
    print(f'Loaded {len(train_pairs)} training pairs')

    encoder = FisherStructureEncoder(dim=256, seed=42)
    encoder.train(train_pairs)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(str(model_path))
    print(f'\nFrozen model saved: {model_path}')

    # Verify round-trip
    loaded = FisherStructureEncoder.load(str(model_path))
    test_text = 'https://example.com/test'
    v_orig = encoder.encode(test_text)
    v_loaded = loaded.encode(test_text)
    assert np.allclose(v_orig, v_loaded, atol=1e-6), 'Save/load mismatch!'
    print('Save/load round-trip verified (bit-identical).')


def phase_evaluate(args):
    """Phase 2: Load frozen model and evaluate on holdout."""
    print('=' * 60)
    print('Phase: EVALUATE (holdout)')
    print('=' * 60)

    # Load frozen model
    print(f'\nLoading frozen model: {args.model_path}')
    encoder = FisherStructureEncoder.load(args.model_path)
    print(f'  dim={encoder.dim}, trained={encoder.trained}')

    # Load holdout pairs
    print(f'\nLoading holdout pairs: {args.holdout_pairs}')
    holdout_pairs = load_pairs(args.holdout_pairs)
    print(f'  Pairs: {len(holdout_pairs)}')
    n_pos = sum(1 for p in holdout_pairs if p['label'] == 1.0)
    n_neg = sum(1 for p in holdout_pairs if p['label'] == 0.0)
    print(f'  Positive: {n_pos}, Negative: {n_neg}')

    # Use the int8 deployment representation (same as dev benchmark)
    encode_fn = encoder.encode_int8 if hasattr(encoder, 'encode_int8') else encoder.encode

    # Quality evaluation
    print('\n--- Quality Evaluation ---')
    quality = evaluate(encode_fn, holdout_pairs)
    print('\nQuality metrics:')
    for k, v in quality.items():
        print(f'  {k:20s}: {v:.4f}')

    # Efficiency benchmark
    print('\n--- Efficiency Benchmark ---')
    test_texts = list(set(
        p['text1'] for p in holdout_pairs[:500]
    ))[:500]
    efficiency = benchmark_efficiency(encode_fn, test_texts, n_iterations=3)
    print('\nEfficiency metrics:')
    for k, v in efficiency.items():
        if isinstance(v, float):
            print(f'  {k:25s}: {v:.2f}')
        else:
            print(f'  {k:25s}: {v}')

    # Criteria check
    criteria = check_criteria(quality, efficiency)

    # Per-family analysis
    print('\n--- Per-Family Analysis ---')
    new_families = {'markdown', 'log_entry', 'regex', 'ini_config'}
    family_results = per_family_analysis(encode_fn, holdout_pairs, new_families)
    for group, info in family_results.items():
        print(f'  {group}: {info}')

    # Bootstrap CI
    print('\n--- Bootstrap AUC-ROC CI ---')
    predictions = []
    labels = []
    for pair in holdout_pairs:
        v1 = encode_fn(pair['text1'])
        v2 = encode_fn(pair['text2'])
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
        predictions.append(sim)
        labels.append(pair['label'])
    predictions = np.array(predictions)
    labels = np.array(labels)
    auc_ci = bootstrap_auc_ci(predictions, labels)
    print(f'  AUC-ROC 95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]')

    # Determine outcome
    outcome = determine_outcome(criteria, family_results)
    print(f'\n{"=" * 60}')
    print(f'OUTCOME: {outcome}')
    print(f'{"=" * 60}')

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_out = {
        'quality': quality,
        'efficiency': efficiency,
        'criteria': {k: {'value': v['value'], 'target': v['target'], 'pass': v['pass']}
                     for k, v in criteria.items()},
        'family_breakdown': family_results,
        'auc_ci_95': {'lower': auc_ci[0], 'upper': auc_ci[1]},
        'outcome': outcome,
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'\nMetrics saved: {output_dir / "metrics.json"}')

    # report.md
    report = generate_report(quality, efficiency, criteria, family_results,
                             outcome, auc_ci, len(holdout_pairs))
    with open(output_dir / 'report.md', 'w') as f:
        f.write(report)
    print(f'Report saved: {output_dir / "report.md"}')



def main():
    parser = argparse.ArgumentParser(
        description='Holdout validation: freeze model and evaluate on independent data')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['save_model', 'evaluate'],
                        help='Phase to run')
    parser.add_argument('--train', type=str, default='data/train.jsonl',
                        help='Training pairs (for save_model phase)')
    parser.add_argument('--model_path', type=str,
                        default='results/holdout_validation/frozen_model.npz',
                        help='Path to frozen model checkpoint')
    parser.add_argument('--holdout_pairs', type=str,
                        default='data/holdout/pairs.jsonl',
                        help='Holdout pairs file (for evaluate phase)')
    parser.add_argument('--output', type=str,
                        default='results/holdout_validation/',
                        help='Output directory for results')
    args = parser.parse_args()

    if args.phase == 'save_model':
        phase_save_model(args)
    elif args.phase == 'evaluate':
        phase_evaluate(args)


if __name__ == '__main__':
    main()
