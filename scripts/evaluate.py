"""Evaluate encoder on test set"""

import json
import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders.random_projection import RandomProjectionEncoder
from src.encoders.simhash import SimHashEncoder
from src.encoders.minhash import MinHashEncoder
from src.encoders.tfidf_svd import TfidfSvdEncoder
from src.encoders.proposed.multiscale import MultiScaleEncoder
from src.encoders.proposed.structure_type import StructureTypeEncoder
from src.encoders.proposed.structure_type_fast import StructureTypeFastEncoder
from src.encoders.proposed.structure_type_quantized import QuantizedEncoder, QuantizedStructureTypeCompactEncoder
from src.encoders.proposed.ngram_hash import NgramHashEncoder, NgramHashMultiscaleEncoder
from src.evaluation.metrics import evaluate, benchmark_efficiency


def load_pairs(file_path: str):
    """Load pairs from JSONL file"""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def get_encoder(model_name: str, train_pairs=None):
    """Get encoder instance by name"""
    if model_name == 'random_projection':
        return RandomProjectionEncoder(output_dim=128, seed=42)
    elif model_name == 'simhash':
        return SimHashEncoder(dim=128, ngram_size=3)
    elif model_name == 'minhash':
        return MinHashEncoder(dim=128, ngram_size=3, seed=42)
    elif model_name == 'tfidf_svd':
        encoder = TfidfSvdEncoder(dim=128, max_features=10000)
        # Fit on training data
        if train_pairs:
            train_texts = []
            for pair in train_pairs:
                train_texts.append(pair['text1'])
                train_texts.append(pair['text2'])
            # Remove duplicates
            train_texts = list(set(train_texts))
            encoder.fit(train_texts)
        return encoder
    elif model_name == 'multiscale':
        return MultiScaleEncoder(dim=128, seed=42)
    elif model_name == 'structure_type':
        return StructureTypeEncoder(dim=128, type_dim=16, seed=42)
    elif model_name == 'structure_type_fast':
        return StructureTypeFastEncoder(dim=128, type_dim=16, seed=42)
    elif model_name == 'structure_type_quantized':
        return QuantizedEncoder(dim=128, type_dim=16, seed=42)
    elif model_name == 'structure_type_quantized_256':
        return QuantizedStructureTypeCompactEncoder(dim=256, type_dim=32, seed=42)
    elif model_name == 'ngram_hash':
        encoder = NgramHashEncoder(dim=128, n_grams=(2, 3, 4), vocab_size=8192, seed=42)
        # Fit IDF on training data
        if train_pairs:
            train_texts = []
            for pair in train_pairs:
                train_texts.append(pair['text1'])
                train_texts.append(pair['text2'])
            train_texts = list(set(train_texts))
            encoder.fit(train_texts)
        return encoder
    elif model_name == 'ngram_hash_multiscale':
        encoder = NgramHashMultiscaleEncoder(dim=128, ngram_dim_ratio=0.75, seed=42)
        # Fit IDF on training data
        if train_pairs:
            train_texts = []
            for pair in train_pairs:
                train_texts.append(pair['text1'])
                train_texts.append(pair['text2'])
            train_texts = list(set(train_texts))
            encoder.fit(train_texts)
        return encoder
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_report(output_dir: Path, model_name: str, config: dict, quality_metrics: dict, efficiency_metrics: dict):
    """Save evaluation report"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save metrics JSON
    metrics = {
        'quality': quality_metrics,
        'efficiency': efficiency_metrics
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate markdown report
    report = f"""# Experiment Results: {model_name}

## Configuration

- Model: {model_name}
- Vector Dimension: {config.get('dim', 128)}
- Key Parameters: {config.get('params', {})}

## Quality Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC-ROC | {quality_metrics['auc_roc']:.4f} |
| Best F1 | {quality_metrics['best_f1']:.4f} |
| Best Threshold | {quality_metrics['best_threshold']:.2f} |
| Separation | {quality_metrics['separation']:.4f} |
| Precision@100 | {quality_metrics['precision_at_100']:.4f} |
| Precision@1000 | {quality_metrics['precision_at_1000']:.4f} |
| Mean Positive Sim | {quality_metrics['mean_pos_sim']:.4f} |
| Mean Negative Sim | {quality_metrics['mean_neg_sim']:.4f} |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Encoding Speed | {efficiency_metrics['encoding_speed']:.0f} texts/sec |
| Vector Bytes | {efficiency_metrics['vector_bytes']} bytes |
| Vector Dimensions | {efficiency_metrics['vector_dimensions']} |
| Comparison Speed | {efficiency_metrics['comparison_speed']:.0f} comparisons/sec |

## Success Criteria Check

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | ≥0.92 | {quality_metrics['auc_roc']:.4f} | {'✓' if quality_metrics['auc_roc'] >= 0.92 else '✗'} |
| Separation | ≥2.5 | {quality_metrics['separation']:.4f} | {'✓' if quality_metrics['separation'] >= 2.5 else '✗'} |
| Best F1 | ≥0.88 | {quality_metrics['best_f1']:.4f} | {'✓' if quality_metrics['best_f1'] >= 0.88 else '✗'} |
| Encoding Speed | ≥10,000/s | {efficiency_metrics['encoding_speed']:.0f}/s | {'✓' if efficiency_metrics['encoding_speed'] >= 10000 else '✗'} |
| Vector Bytes | ≤256 | {efficiency_metrics['vector_bytes']} | {'✓' if efficiency_metrics['vector_bytes'] <= 256 else '✗'} |
| Mean Positive Sim | ≥0.85 | {quality_metrics['mean_pos_sim']:.4f} | {'✓' if quality_metrics['mean_pos_sim'] >= 0.85 else '✗'} |
| Mean Negative Sim | ≤0.35 | {quality_metrics['mean_neg_sim']:.4f} | {'✓' if quality_metrics['mean_neg_sim'] <= 0.35 else '✗'} |
"""

    with open(output_dir / 'report.md', 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to {output_dir}/")
    print(f"  - config.yaml")
    print(f"  - metrics.json")
    print(f"  - report.md")


def main():
    parser = argparse.ArgumentParser(description='Evaluate encoder')
    parser.add_argument('--model', type=str, required=True,
                        choices=['random_projection', 'simhash', 'minhash', 'tfidf_svd', 'multiscale',
                                 'structure_type', 'structure_type_fast',
                                 'structure_type_quantized', 'structure_type_quantized_256',
                                 'ngram_hash', 'ngram_hash_multiscale'],
                        help='Model name')
    parser.add_argument('--test', type=str, default='data/test.jsonl',
                        help='Test data file')
    parser.add_argument('--train', type=str, default='data/train.jsonl',
                        help='Train data file (for TF-IDF+SVD)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    # Set output directory
    if args.output is None:
        args.output = f'results/baseline_{args.model}'
    output_dir = Path(args.output)

    print(f"{'='*60}")
    print(f"Evaluating: {args.model}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading test data from {args.test}...")
    test_pairs = load_pairs(args.test)
    print(f"Loaded {len(test_pairs)} test pairs")

    train_pairs = None
    if args.model in ['tfidf_svd', 'ngram_hash', 'ngram_hash_multiscale']:
        print(f"\nLoading train data from {args.train}...")
        train_pairs = load_pairs(args.train)
        print(f"Loaded {len(train_pairs)} train pairs")

    # Get encoder
    print(f"\nInitializing {args.model} encoder...")
    encoder = get_encoder(args.model, train_pairs)

    # Evaluate quality
    print(f"\n{'='*60}")
    print("Quality Evaluation")
    print(f"{'='*60}")
    quality_metrics = evaluate(encoder.encode, test_pairs)

    print("\nQuality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  {key:20s}: {value:.4f}")

    # Benchmark efficiency
    print(f"\n{'='*60}")
    print("Efficiency Benchmark")
    print(f"{'='*60}")

    # Get unique texts for benchmarking
    test_texts = []
    for pair in test_pairs[:1000]:  # Use first 1000 pairs
        test_texts.append(pair['text1'])
        test_texts.append(pair['text2'])
    test_texts = list(set(test_texts))[:500]  # Unique texts

    efficiency_metrics = benchmark_efficiency(encoder.encode, test_texts, n_iterations=3)

    print("\nEfficiency Metrics:")
    for key, value in efficiency_metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.2f}")
        else:
            print(f"  {key:25s}: {value}")

    # Save report
    config = {
        'model': args.model,
        'dim': encoder.dim,
        'params': {}
    }

    if args.model == 'random_projection':
        config['params'] = {'input_dim': 1024, 'seed': 42}
    elif args.model in ['simhash', 'minhash']:
        config['params'] = {'ngram_size': 3}
    elif args.model == 'tfidf_svd':
        config['params'] = {'max_features': 10000, 'ngram_range': (2, 4)}

    save_report(output_dir, args.model, config, quality_metrics, efficiency_metrics)

    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
