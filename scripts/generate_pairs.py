"""Generate positive and negative pairs from dataset"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_dataset(input_path: str):
    """Load dataset from JSONL file"""
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def generate_pairs(samples, n_positive=5000, n_negative=5000, seed=42):
    """Generate positive and negative pairs"""
    random.seed(seed)

    # Group by category
    by_category = defaultdict(list)
    for sample in samples:
        by_category[sample['category']].append(sample)

    print(f"Found {len(by_category)} categories")

    pairs = []

    # Generate positive pairs (same category)
    print(f"\nGenerating {n_positive} positive pairs...")
    categories = list(by_category.keys())
    pairs_per_category = n_positive // len(categories)

    for category in categories:
        cat_samples = by_category[category]
        if len(cat_samples) < 2:
            continue

        for _ in range(min(pairs_per_category, len(cat_samples) * (len(cat_samples) - 1) // 2)):
            s1, s2 = random.sample(cat_samples, 2)
            pairs.append({
                'text1': s1['text'],
                'text2': s2['text'],
                'category1': category,
                'category2': category,
                'label': 1.0
            })

    # Generate negative pairs (different categories)
    print(f"Generating {n_negative} negative pairs...")
    for _ in range(n_negative):
        cat1, cat2 = random.sample(categories, 2)
        s1 = random.choice(by_category[cat1])
        s2 = random.choice(by_category[cat2])

        pairs.append({
            'text1': s1['text'],
            'text2': s2['text'],
            'category1': cat1,
            'category2': cat2,
            'label': 0.0
        })

    # Shuffle
    random.shuffle(pairs)

    print(f"\nTotal pairs generated: {len(pairs)}")
    print(f"  Positive: {sum(1 for p in pairs if p['label'] == 1.0)}")
    print(f"  Negative: {sum(1 for p in pairs if p['label'] == 0.0)}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Generate pairs from dataset')
    parser.add_argument('--input', type=str, default='data/synthetic.jsonl',
                        help='Input dataset file')
    parser.add_argument('--output', type=str, default='data/pairs.jsonl',
                        help='Output pairs file')
    parser.add_argument('--n_positive', type=int, default=5000,
                        help='Number of positive pairs')
    parser.add_argument('--n_negative', type=int, default=5000,
                        help='Number of negative pairs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    samples = load_dataset(args.input)
    print(f"Loaded {len(samples)} samples")

    # Generate pairs
    pairs = generate_pairs(samples, args.n_positive, args.n_negative, args.seed)

    # Save pairs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"✓ Pairs saved: {len(pairs)} pairs")


if __name__ == '__main__':
    main()
