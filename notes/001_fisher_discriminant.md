# Experiment 001: Fisher Discriminant Structure Encoder

> **Historical experiment snapshot.** Fisher의 7/7은 초기 24-category pair benchmark 결과입니다. 후속 OOD 평가와 현재 우승자는 [`../results/FINAL_REPORT.md`](../results/FINAL_REPORT.md)를 참조하세요.

## Date
2025-01-XX

## Hypothesis

The existing StructureTypeEncoder achieves high AUC-ROC (0.958) but fails on separation (1.08)
and mean negative similarity (0.645) because different structural types (e.g., code_python vs
code_javascript, url vs email) produce nearly identical feature vectors. A Fisher Linear
Discriminant approach can learn a projection matrix that maximizes between-class scatter while
minimizing within-class scatter, effectively pulling same-type texts together and pushing
different-type texts apart in the embedding space.

## Approach

### Feature Extraction (320 dimensions, numba-compiled)

1. **Byte histogram** (64 dims): Full 256-bin histogram grouped into 64 bins (4 bytes/bin)
2. **Character class ratios** (12 dims): Alpha, digit, upper, lower, space, punct, bracket,
   math, special, ASCII ratio, non-ASCII ratio, digit/alpha ratio - computed via precomputed
   lookup tables
3. **Bigram hash features** (64 dims): Byte bigrams hashed into 64 bins
4. **Structural character frequencies** (32 dims): Individual counts for `.`, `/`, `:`, `@`,
   `#`, `=`, `?`, `&`, `;`, `,`, `{`, `}`, `[`, `]`, `<`, `>`, `(`, `)`, `"`, `'`, `-`,
   `_`, `\`, `|`, `~`, `+`, `*`, `%`, `!`, `\n`, `\t`, space
5. **Statistical features** (45 dims): Byte mean/std, length features, unique byte ratio,
   entropy, change rate, whitespace features, bracket balance, leading byte indicators,
   pattern indicators, positional bytes, quartile means, separator regularity
6. **Trigram hash features** (64 dims): Byte trigrams hashed into 64 bins
7. **Quadgram hash features** (32 dims): Byte quadgrams hashed into 32 bins
8. **Run-length features** (4 dims): Max digit/alpha runs, UTF-8 continuation ratio

### Key Speed Optimization

All feature extraction is compiled with **numba @njit** decorator. This eliminates Python
interpreter overhead and achieves native-speed loops. The per-character Python loops that
plagued the original encoder (3,600 texts/sec) are replaced by compiled C-speed loops
(75,000+ texts/sec).

### Training Procedure

1. **Feature extraction**: Extract 320-dim features for all ~6,870 unique texts from training
   pairs, grouped by 24 categories
2. **Feature normalization**: Standardize features (zero mean, unit variance)
3. **Scatter matrices**: Compute between-class scatter Sb and within-class scatter Sw
4. **Fisher LDA**: Solve generalized eigenvalue problem `Sb w = lambda Sw w` to find
   optimal projection directions (top 256 eigenvectors)
5. **Contrastive fine-tuning**: 5 epochs of SGD on 3,000 sampled pairs, pulling positive
   pairs together (target sim > 0.95) and pushing negative pairs apart (margin = 0.1)

### Encoding Pipeline (inference)

1. Encode text to UTF-8 bytes
2. Extract 320 features via numba-compiled function
3. Standardize: `(features - mean) / std`
4. Project: `vec = W^T @ features` (320 -> 256 dims)
5. L2 normalize: `vec = vec / ||vec||`
6. Return float32 (int8 quantizable for 256-byte storage)

## Results

| Metric | Target | Baseline (StructureType) | Fisher Encoder | Status |
|--------|--------|--------------------------|----------------|--------|
| AUC-ROC | >= 0.92 | 0.9582 | **0.9977** | PASS |
| Separation | >= 2.5 | 1.0843 | **3.7027** | PASS |
| Best F1 | >= 0.88 | 0.8782 | **0.9874** | PASS |
| Mean Pos Sim | >= 0.85 | 0.9645 | **0.9469** | PASS |
| Mean Neg Sim | <= 0.35 | 0.6457 | **-0.0377** | PASS |
| Speed | >= 10,000/s | 11,347/s | **74,495/s** | PASS |
| Vector Bytes (int8) | <= 256 | 256 | **256** | PASS |

**All 7 criteria achieved.**

## Analysis

### Why Fisher LDA Works So Well Here

The core problem was that character-level statistics alone cannot discriminate between
structural types that share similar character distributions (code_python vs code_javascript,
url vs email, filepath vs email). Fisher LDA finds the *linear combinations* of features
that best separate the 24 categories. For example:

- URLs have high `/`, `:`, `.` combined with `http` prefix -- Fisher learns to weight this
  combination differently from filepaths (which have `/` and `.` but no `:` after `http`)
- Python code has `def`, indentation, `:` at end of lines -- different from JavaScript's
  `function`, `{`, `}`
- The trigram/quadgram hash features capture these multi-byte patterns and Fisher learns
  which hash bins discriminate between types

### Speed Achievement

The numba JIT compilation transforms Python loops into native machine code. Despite the
feature extraction involving ~320 features with multiple passes over the byte array, the
compiled code achieves 74,495 texts/sec (7.4x the target). This is 6.6x faster than the
baseline encoder which uses Python-level character iteration.

### Negative Similarity Near Zero

The mean negative similarity of -0.0377 (vs baseline 0.6457) shows near-orthogonal
embeddings for different-type pairs. This is because Fisher LDA explicitly maximizes the
ratio of between-class to within-class variance. The 24 categories are projected into
nearly orthogonal subspaces of the 256-dim output.

### Contrastive Fine-tuning Impact

The contrastive fine-tuning (5 epochs on 3,000 pairs) provides modest improvement over
pure Fisher LDA by directly optimizing the pairwise similarity objective rather than just
class separation. It helps handle cases where texts from the same category have different
sub-patterns.

## Conclusions

1. **Fisher LDA is highly effective** for this structural similarity problem because the
   24 text categories have distinct statistical signatures that are linearly separable
   in the high-dimensional feature space.

2. **Numba JIT is essential** for achieving speed requirements. The same algorithm in pure
   Python/numpy achieves only 3,600 texts/sec vs 74,495 with numba.

3. **Feature design matters**: The combination of byte histograms, n-gram hashes, structural
   character counts, and positional features provides a rich representation that Fisher LDA
   can effectively project.

4. **All 7 criteria satisfied** with significant margins on most metrics, demonstrating
   that the approach is robust and well-suited to this problem.
