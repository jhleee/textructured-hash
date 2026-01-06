"""Split dataset into train/val/test sets"""

import json
import random
import argparse
from pathlib import Path


def load_pairs(input_path: str):
    """Load pairs from JSONL file"""
    pairs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def split_dataset(pairs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """Split dataset into train/val/test"""
    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    return train, val, test


def save_split(pairs, output_path):
    """Save split to JSONL file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--input', type=str, default='data/pairs.jsonl',
                        help='Input pairs file')
    parser.add_argument('--train', type=str, default='data/train.jsonl',
                        help='Output train file')
    parser.add_argument('--val', type=str, default='data/val.jsonl',
                        help='Output validation file')
    parser.add_argument('--test', type=str, default='data/test.jsonl',
                        help='Output test file')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Load pairs
    print(f"Loading pairs from {args.input}...")
    pairs = load_pairs(args.input)
    print(f"Loaded {len(pairs)} pairs")

    # Split
    print(f"\nSplitting dataset (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})...")
    train, val, test = split_dataset(pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    print(f"  Train: {len(train)} pairs")
    print(f"  Val:   {len(val)} pairs")
    print(f"  Test:  {len(test)} pairs")

    # Save splits
    print(f"\nSaving splits...")
    save_split(train, Path(args.train))
    save_split(val, Path(args.val))
    save_split(test, Path(args.test))

    print(f"✓ Splits saved")
    print(f"  {args.train}")
    print(f"  {args.val}")
    print(f"  {args.test}")


if __name__ == '__main__':
    main()
