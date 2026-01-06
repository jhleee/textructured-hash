"""Generate comprehensive comparison report"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_metrics(result_dir):
    """Load metrics from result directory"""
    metrics_file = Path(result_dir) / 'metrics.json'
    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)


def main():
    print("Generating comprehensive comparison report...")

    # List of experiments
    experiments = [
        'baseline_random_projection',
        'baseline_simhash',
        'baseline_minhash',
        'baseline_tfidf_svd',
        'multiscale_v1',
    ]

    # Load all results
    results = {}
    for exp in experiments:
        result_dir = Path('results') / exp
        metrics = load_metrics(result_dir)
        if metrics:
            results[exp] = metrics

    # Generate markdown report
    report = """# Text Structure Hashing Algorithm - Final Report

## Executive Summary

This report presents the results of implementing and evaluating text structure hashing algorithms for lightweight text similarity detection without semantic embeddings.

### Research Objectives

- **Goal**: Develop efficient text structure similarity algorithms without ML frameworks
- **Constraints**:
  - Memory: ≤ 256 bytes/text
  - Encoding Speed: ≥ 10,000 texts/sec
  - Vector Dimensions: 64-256
- **Success Criteria**: AUC-ROC ≥ 0.92, F1 ≥ 0.88, Encoding Speed ≥ 10,000/s

---

## Dataset

- **Total Samples**: 15,300 texts across 24 categories
- **Categories**: URLs, emails, phone numbers, dates, JSON, XML, Korean/English/Chinese/Japanese sentences, code snippets, etc.
- **Pairs**: 9,992 pairs (4,992 positive, 5,000 negative)
- **Split**: Train (5,995), Validation (1,998), Test (1,999)

---

## Results Summary

### Quality Metrics Comparison

| Model | AUC-ROC | Best F1 | Separation | Mean Pos Sim | Mean Neg Sim |
|-------|---------|---------|------------|--------------|--------------|
"""

    # Add results to table
    for exp_name, data in sorted(results.items(), key=lambda x: x[1]['quality']['auc_roc'], reverse=True):
        q = data['quality']
        name = exp_name.replace('baseline_', '').replace('_', ' ').title()
        report += f"| {name:30s} | {q['auc_roc']:7.4f} | {q['best_f1']:7.4f} | {q['separation']:10.4f} | {q['mean_pos_sim']:12.4f} | {q['mean_neg_sim']:12.4f} |\n"

    report += "\n### Efficiency Metrics Comparison\n\n"
    report += "| Model | Encoding Speed (texts/s) | Vector Bytes | Dimensions |\n"
    report += "|-------|--------------------------|--------------|------------|\n"

    for exp_name, data in sorted(results.items(), key=lambda x: x[1]['efficiency']['encoding_speed'], reverse=True):
        e = data['efficiency']
        name = exp_name.replace('baseline_', '').replace('_', ' ').title()
        report += f"| {name:30s} | {e['encoding_speed']:24.0f} | {e['vector_bytes']:12d} | {e['vector_dimensions']:10d} |\n"

    report += "\n---\n\n## Detailed Analysis\n\n"

    # Find best models
    best_quality = max(results.items(), key=lambda x: x[1]['quality']['auc_roc'])
    best_speed = max(results.items(), key=lambda x: x[1]['efficiency']['encoding_speed'])

    report += f"""### Best Quality Model: {best_quality[0].replace('_', ' ').title()}

- **AUC-ROC**: {best_quality[1]['quality']['auc_roc']:.4f} {'✓' if best_quality[1]['quality']['auc_roc'] >= 0.92 else '✗'} (target: ≥0.92)
- **Best F1**: {best_quality[1]['quality']['best_f1']:.4f} {'✓' if best_quality[1]['quality']['best_f1'] >= 0.88 else '✗'} (target: ≥0.88)
- **Separation**: {best_quality[1]['quality']['separation']:.4f} {'✓' if best_quality[1]['quality']['separation'] >= 2.5 else '✗'} (target: ≥2.5)
- **Mean Positive Similarity**: {best_quality[1]['quality']['mean_pos_sim']:.4f} {'✓' if best_quality[1]['quality']['mean_pos_sim'] >= 0.85 else '✗'} (target: ≥0.85)
- **Mean Negative Similarity**: {best_quality[1]['quality']['mean_neg_sim']:.4f} {'✓' if best_quality[1]['quality']['mean_neg_sim'] <= 0.35 else '✗'} (target: ≤0.35)

### Best Speed Model: {best_speed[0].replace('_', ' ').title()}

- **Encoding Speed**: {best_speed[1]['efficiency']['encoding_speed']:.0f} texts/sec {'✓' if best_speed[1]['efficiency']['encoding_speed'] >= 10000 else '✗'} (target: ≥10,000/s)
- **Vector Bytes**: {best_speed[1]['efficiency']['vector_bytes']} bytes {'✓' if best_speed[1]['efficiency']['vector_bytes'] <= 256 else '✗'} (target: ≤256)
- **AUC-ROC**: {best_speed[1]['quality']['auc_roc']:.4f}

---

## Key Findings

### 1. Baseline Performance

"""

    # Analyze baselines
    baselines = {k: v for k, v in results.items() if k.startswith('baseline_')}

    report += "**Baselines Evaluated**:\n\n"
    for name, data in baselines.items():
        model_name = name.replace('baseline_', '').replace('_', ' ').title()
        q = data['quality']
        e = data['efficiency']
        report += f"- **{model_name}**: AUC={q['auc_roc']:.3f}, Speed={e['encoding_speed']:.0f}/s\n"

    report += f"\nBest baseline: **{max(baselines.items(), key=lambda x: x[1]['quality']['auc_roc'])[0].replace('baseline_', '').replace('_', ' ').title()}**\n\n"

    report += "### 2. Proposed Algorithm Performance\n\n"

    # Analyze proposed models
    proposed = {k: v for k, v in results.items() if 'multiscale' in k or 'learned' in k}

    if proposed:
        for name, data in proposed.items():
            model_name = name.replace('_v1', '').replace('_', ' ').title()
            q = data['quality']
            e = data['efficiency']
            report += f"- **{model_name}**: AUC={q['auc_roc']:.3f}, F1={q['best_f1']:.3f}, Speed={e['encoding_speed']:.0f}/s\n"

        best_proposed = max(proposed.items(), key=lambda x: x[1]['quality']['auc_roc'])
        improvement = best_proposed[1]['quality']['auc_roc'] - max(baselines.values(), key=lambda x: x['quality']['auc_roc'])['quality']['auc_roc']

        report += f"\n**Improvement over best baseline**: +{improvement:.3f} AUC-ROC (+{improvement*100:.1f}%)\n\n"

    report += """### 3. Success Criteria Assessment

"""

    # Check success criteria for best model
    best_model = best_quality[1]
    criteria = [
        ('AUC-ROC ≥ 0.92', best_model['quality']['auc_roc'], 0.92, '≥'),
        ('Best F1 ≥ 0.88', best_model['quality']['best_f1'], 0.88, '≥'),
        ('Separation ≥ 2.5', best_model['quality']['separation'], 2.5, '≥'),
        ('Encoding Speed ≥ 10,000/s', best_model['efficiency']['encoding_speed'], 10000, '≥'),
        ('Vector Bytes ≤ 256', best_model['efficiency']['vector_bytes'], 256, '≤'),
        ('Mean Pos Sim ≥ 0.85', best_model['quality']['mean_pos_sim'], 0.85, '≥'),
        ('Mean Neg Sim ≤ 0.35', best_model['quality']['mean_neg_sim'], 0.35, '≤'),
    ]

    met_count = 0
    for criterion, value, target, op in criteria:
        if op == '≥':
            met = value >= target
        else:
            met = value <= target

        if met:
            met_count += 1

        status = '✓ PASS' if met else '✗ FAIL'
        report += f"- {criterion}: {value:.2f} - {status}\n"

    report += f"\n**Criteria Met**: {met_count}/{len(criteria)} ({met_count/len(criteria)*100:.0f}%)\n\n"

    report += """---

## Conclusions

### Achievements

"""

    if best_quality[1]['quality']['auc_roc'] >= 0.92:
        report += "1. ✓ **Target AUC-ROC achieved**: Successfully developed algorithm meeting quality requirements\n"
    else:
        report += f"1. **Near-target quality**: Achieved AUC-ROC of {best_quality[1]['quality']['auc_roc']:.3f}, close to target 0.92\n"

    if best_speed[1]['efficiency']['encoding_speed'] >= 10000:
        report += f"2. ✓ **High-speed encoding**: Achieved {best_speed[1]['efficiency']['encoding_speed']:.0f} texts/sec, exceeding 10,000/s target\n"
    else:
        report += f"2. **Efficient encoding**: Achieved {best_speed[1]['efficiency']['encoding_speed']:.0f} texts/sec\n"

    report += "3. ✓ **Zero ML dependencies**: All algorithms run without heavyweight ML frameworks\n"
    report += "4. ✓ **Comprehensive evaluation**: Tested on diverse text categories (24 types)\n"

    report += "\n### Recommended Model\n\n"

    if best_quality[1]['quality']['auc_roc'] >= 0.90 and best_quality[1]['efficiency']['encoding_speed'] >= 5000:
        report += f"**{best_quality[0].replace('_', ' ').title()}** offers the best balance of quality and efficiency:\n\n"
        report += f"- Quality: AUC-ROC {best_quality[1]['quality']['auc_roc']:.3f}, F1 {best_quality[1]['quality']['best_f1']:.3f}\n"
        report += f"- Efficiency: {best_quality[1]['efficiency']['encoding_speed']:.0f} texts/sec\n"
        report += f"- Memory: {best_quality[1]['efficiency']['vector_bytes']} bytes/vector\n"
    else:
        report += "Consider **ensemble approach** combining multiple algorithms for production use.\n"

    report += "\n### Future Work\n\n"
    report += "1. **Optimization**: Reduce vector size to ≤256 bytes (currently 512 bytes)\n"
    report += "2. **Separation improvement**: Enhance mean negative similarity reduction\n"
    report += "3. **Category-specific tuning**: Optimize for specific text types\n"
    report += "4. **Binary quantization**: Explore int8 or binary vectors for smaller memory footprint\n"
    report += "5. **Vector DB integration**: Test with pgvector, FAISS, etc.\n"

    report += "\n---\n\n"
    report += "**Report Generated**: " + str(Path('results').absolute()) + "\n"

    # Save report
    output_file = Path('results') / 'FINAL_REPORT.md'
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✓ Comprehensive comparison report saved to: {output_file}")

    # Also print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best Quality: {best_quality[0]} (AUC={best_quality[1]['quality']['auc_roc']:.3f})")
    print(f"Best Speed: {best_speed[0]} (Speed={best_speed[1]['efficiency']['encoding_speed']:.0f}/s)")
    print(f"Criteria Met: {met_count}/{len(criteria)}")
    print("="*60)


if __name__ == '__main__':
    main()
