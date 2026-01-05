"""Generate synthetic dataset"""

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--output', type=str, default='data/synthetic.jsonl',
                        help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"Generating synthetic dataset with seed={args.seed}...")
    generator = DatasetGenerator(seed=args.seed)
    samples = generator.generate_all()

    # Save to JSONL
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✓ Dataset saved: {len(samples)} samples")

    # Print statistics
    categories = {}
    for sample in samples:
        cat = sample['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:25s}: {count:5d}")


if __name__ == '__main__':
    main()
