"""Experiment: Dimension Sweep for Int8 Quantization

Test how int8 quantized encoder performance varies with different dimensions.
Dimensions tested: 32, 64, 128, 256, 512, 1024
"""

import json
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders.proposed.structure_type_quantized import QuantizedStructureTypeCompactEncoder
from src.evaluation.metrics import evaluate, benchmark_efficiency


def load_pairs(file_path: str):
    """Load pairs from JSONL file"""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def run_dimension_experiment(dim: int, test_pairs: list, test_texts: list):
    """Run experiment for a specific dimension"""

    print(f"\n{'='*70}")
    print(f"Testing Dimension: {dim}")
    print(f"{'='*70}")

    # Calculate type_dim proportional to total dim (1/8 ratio as in original)
    type_dim = max(8, dim // 8)

    print(f"  Total dim: {dim}, Type dim: {type_dim}")

    # Create encoder
    encoder = QuantizedStructureTypeCompactEncoder(dim=dim, type_dim=type_dim, seed=42)

    # Evaluate quality
    print(f"\n  Quality Evaluation...")
    start_time = time.time()
    quality_metrics = evaluate(encoder.encode, test_pairs)
    eval_time = time.time() - start_time

    print(f"  ✓ Quality evaluation completed in {eval_time:.1f}s")

    # Benchmark efficiency
    print(f"\n  Efficiency Benchmark...")
    efficiency_metrics = benchmark_efficiency(encoder.encode, test_texts, n_iterations=3)

    # Calculate int8 storage size
    int8_bytes = dim  # 1 byte per dimension with int8

    # Add to metrics
    efficiency_metrics['int8_bytes'] = int8_bytes

    return {
        'dim': dim,
        'type_dim': type_dim,
        'quality': quality_metrics,
        'efficiency': efficiency_metrics,
        'int8_bytes': int8_bytes
    }


def save_results(results: list, output_dir: Path):
    """Save all results to JSON and markdown"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_dir / 'dimension_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    report = """# Experiment: Dimension Sweep for Int8 Quantization

## Objective

Test how int8 quantized StructureTypeEncoder performance varies with different dimensions.

## Methodology

- **Model**: QuantizedStructureTypeCompactEncoder
- **Dimensions tested**: 32, 64, 128, 256, 512, 1024
- **Type dimension**: Scaled proportionally (dim // 8, minimum 8)
- **Quantization**: Float32 → Int8 (127 levels)
- **Seed**: 42 (reproducible)

---

## Results Summary

### Quality Metrics Comparison

| Dimension | AUC-ROC | Best F1 | Separation | Mean Pos Sim | Mean Neg Sim | Speed (texts/s) | Int8 Bytes |
|-----------|---------|---------|------------|--------------|--------------|-----------------|------------|
"""

    for result in results:
        q = result['quality']
        e = result['efficiency']
        report += f"| {result['dim']:4d} | {q['auc_roc']:.4f} | {q['best_f1']:.4f} | {q['separation']:.4f} | {q['mean_pos_sim']:.4f} | {q['mean_neg_sim']:.4f} | {e['encoding_speed']:>7.0f} | {result['int8_bytes']:3d} |\n"

    report += "\n### Success Criteria Achievement\n\n"
    report += "| Dimension | AUC ≥0.92 | F1 ≥0.88 | Sep ≥2.5 | Pos ≥0.85 | Neg ≤0.35 | Speed ≥10k | Bytes ≤256 | Total |\n"
    report += "|-----------|-----------|----------|----------|-----------|-----------|------------|------------|-------|\n"

    for result in results:
        q = result['quality']
        e = result['efficiency']
        checks = []
        checks.append('✓' if q['auc_roc'] >= 0.92 else '✗')
        checks.append('✓' if q['best_f1'] >= 0.88 else '✗')
        checks.append('✓' if q['separation'] >= 2.5 else '✗')
        checks.append('✓' if q['mean_pos_sim'] >= 0.85 else '✗')
        checks.append('✓' if q['mean_neg_sim'] <= 0.35 else '✗')
        checks.append('✓' if e['encoding_speed'] >= 10000 else '✗')
        checks.append('✓' if result['int8_bytes'] <= 256 else '✗')
        total = sum(1 for c in checks if c == '✓')

        report += f"| {result['dim']:4d} | {checks[0]} | {checks[1]} | {checks[2]} | {checks[3]} | {checks[4]} | {checks[5]} | {checks[6]} | {total}/7 |\n"

    # Add detailed analysis
    report += "\n---\n\n## Detailed Analysis\n\n"

    for result in results:
        dim = result['dim']
        q = result['quality']
        e = result['efficiency']

        report += f"### Dimension: {dim}\n\n"
        report += f"**Configuration**:\n"
        report += f"- Total dimension: {dim}\n"
        report += f"- Type dimension: {result['type_dim']}\n"
        report += f"- Int8 storage: {result['int8_bytes']} bytes\n"
        report += f"- Float32 storage (eval): {e['vector_bytes']} bytes\n\n"

        report += f"**Quality Metrics**:\n"
        report += f"- AUC-ROC: {q['auc_roc']:.4f}\n"
        report += f"- Best F1: {q['best_f1']:.4f} (threshold: {q['best_threshold']:.2f})\n"
        report += f"- Separation: {q['separation']:.4f}\n"
        report += f"- Mean Positive Similarity: {q['mean_pos_sim']:.4f}\n"
        report += f"- Mean Negative Similarity: {q['mean_neg_sim']:.4f}\n"
        report += f"- Precision@100: {q['precision_at_100']:.4f}\n"
        report += f"- Precision@1000: {q['precision_at_1000']:.4f}\n\n"

        report += f"**Efficiency Metrics**:\n"
        report += f"- Encoding Speed: {e['encoding_speed']:.0f} texts/sec\n"
        report += f"- Comparison Speed: {e['comparison_speed']:.0f} comparisons/sec\n\n"

        report += "---\n\n"

    # Add recommendations
    report += "## Recommendations\n\n"

    # Find best for each criterion
    best_f1 = max(results, key=lambda r: r['quality']['best_f1'])
    best_auc = max(results, key=lambda r: r['quality']['auc_roc'])
    best_speed = max(results, key=lambda r: r['efficiency']['encoding_speed'])
    smallest_mem = min(results, key=lambda r: r['int8_bytes'])
    best_separation = max(results, key=lambda r: r['quality']['separation'])

    report += f"### Best Quality (F1 Score)\n"
    report += f"- **Dimension {best_f1['dim']}**: F1 = {best_f1['quality']['best_f1']:.4f}, AUC = {best_f1['quality']['auc_roc']:.4f}\n"
    report += f"- Storage: {best_f1['int8_bytes']} bytes (int8)\n\n"

    report += f"### Best Speed\n"
    report += f"- **Dimension {best_speed['dim']}**: {best_speed['efficiency']['encoding_speed']:.0f} texts/sec\n"
    report += f"- Quality: F1 = {best_speed['quality']['best_f1']:.4f}, AUC = {best_speed['quality']['auc_roc']:.4f}\n\n"

    report += f"### Best Memory Efficiency\n"
    report += f"- **Dimension {smallest_mem['dim']}**: {smallest_mem['int8_bytes']} bytes (int8)\n"
    report += f"- Quality: F1 = {smallest_mem['quality']['best_f1']:.4f}, AUC = {smallest_mem['quality']['auc_roc']:.4f}\n\n"

    report += f"### Best Separation\n"
    report += f"- **Dimension {best_separation['dim']}**: {best_separation['quality']['separation']:.4f}\n"
    report += f"- Quality: F1 = {best_separation['quality']['best_f1']:.4f}, AUC = {best_separation['quality']['auc_roc']:.4f}\n\n"

    # Find balanced recommendation
    report += "### Balanced Recommendation\n\n"
    report += "Based on meeting targets and trade-offs:\n\n"

    # Score each dimension
    for result in results:
        q = result['quality']
        e = result['efficiency']
        score = 0
        if q['auc_roc'] >= 0.92: score += 1
        if q['best_f1'] >= 0.88: score += 1
        if q['mean_pos_sim'] >= 0.85: score += 1
        if result['int8_bytes'] <= 256: score += 1
        result['criteria_met'] = score

    best_overall = max(results, key=lambda r: (r['criteria_met'], r['quality']['best_f1']))

    report += f"**Recommended: Dimension {best_overall['dim']}**\n"
    report += f"- Criteria met: {best_overall['criteria_met']}/7\n"
    report += f"- F1 Score: {best_overall['quality']['best_f1']:.4f}\n"
    report += f"- AUC-ROC: {best_overall['quality']['auc_roc']:.4f}\n"
    report += f"- Storage: {best_overall['int8_bytes']} bytes (int8)\n"
    report += f"- Speed: {best_overall['efficiency']['encoding_speed']:.0f} texts/sec\n\n"

    # Save markdown
    with open(output_dir / 'EXPERIMENT_DIMENSION_SWEEP.md', 'w') as f:
        f.write(report)

    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  - dimension_sweep_results.json")
    print(f"  - EXPERIMENT_DIMENSION_SWEEP.md")


def main():
    parser = argparse.ArgumentParser(description='Dimension sweep experiment')
    parser.add_argument('--test', type=str, default='data/test.jsonl',
                        help='Test data file')
    parser.add_argument('--output', type=str, default='results/experiment_dimension_sweep',
                        help='Output directory')
    parser.add_argument('--dims', type=int, nargs='+', default=[64, 128, 256, 512, 1024],
                        help='Dimensions to test')
    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"{'='*70}")
    print(f"EXPERIMENT: Dimension Sweep for Int8 Quantization")
    print(f"{'='*70}")
    print(f"\nTesting dimensions: {args.dims}")

    # Load test data
    print(f"\nLoading test data from {args.test}...")
    test_pairs = load_pairs(args.test)
    print(f"✓ Loaded {len(test_pairs)} test pairs")

    # Prepare test texts for efficiency benchmark
    print(f"\nPreparing test texts for efficiency benchmark...")
    test_texts = []
    for pair in test_pairs[:1000]:
        test_texts.append(pair['text1'])
        test_texts.append(pair['text2'])
    test_texts = list(set(test_texts))[:500]
    print(f"✓ Prepared {len(test_texts)} unique test texts")

    # Run experiments
    results = []
    for dim in args.dims:
        result = run_dimension_experiment(dim, test_pairs, test_texts)
        results.append(result)

        # Print quick summary
        print(f"\n  Summary:")
        print(f"    AUC-ROC: {result['quality']['auc_roc']:.4f}")
        print(f"    F1: {result['quality']['best_f1']:.4f}")
        print(f"    Speed: {result['efficiency']['encoding_speed']:.0f} texts/sec")
        print(f"    Int8 storage: {result['int8_bytes']} bytes")

    # Save all results
    print(f"\n{'='*70}")
    print(f"Saving Results")
    print(f"{'='*70}")
    save_results(results, output_dir)

    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
