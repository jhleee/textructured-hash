"""Structure-Aware N-gram Hashing Encoder (Experiment 1.1)

TF-IDF weighted n-gram hash features with random projection.
"""

import numpy as np
from ..base import BaseEncoder


class NgramHashEncoder(BaseEncoder):
    """
    Experiment 1.1: Structure-Aware N-gram Hashing

    Hypothesis: Character n-gram hash-based features with TF-IDF weighting
    will better capture structural differences between texts.

    Expected improvement:
    - Mean Neg Sim: 0.64 → 0.25 (TF-IDF effect)
    - Speed: ~15,000/s (simple hash operations)
    """

    def __init__(
        self,
        dim: int = 128,
        n_grams: tuple = (2, 3, 4),
        vocab_size: int = 8192,
        seed: int = 42,
        use_idf: bool = True
    ):
        """
        Args:
            dim: Output vector dimension
            n_grams: N-gram sizes to extract (e.g., (2, 3, 4))
            vocab_size: Hash table size (number of buckets)
            seed: Random seed
            use_idf: Whether to apply IDF weighting
        """
        self._dim = dim
        self.n_grams = n_grams
        self.vocab_size = vocab_size
        self.use_idf = use_idf

        np.random.seed(seed)

        # Random projection matrix: vocab_size -> dim
        self.projection = np.random.randn(vocab_size, dim).astype(np.float32)
        self.projection /= np.sqrt(dim)

        # IDF weights (initialized to 1, can be fitted)
        self.idf = np.ones(vocab_size, dtype=np.float32)
        self.fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    def _hash_ngram(self, ngram: str) -> int:
        """
        Hash an n-gram to a bucket index.

        Uses Python's built-in hash for speed, modulo vocab_size.
        """
        return hash(ngram) % self.vocab_size

    def _extract_ngram_features(self, text: str) -> np.ndarray:
        """
        Extract n-gram hash features with TF weighting.

        Returns:
            Sparse TF vector of size vocab_size
        """
        tf = np.zeros(self.vocab_size, dtype=np.float32)

        # Extract n-grams for each size
        for n in self.n_grams:
            if len(text) < n:
                continue

            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                h = self._hash_ngram(ngram)
                tf[h] += 1.0

        # Normalize by total n-gram count (TF normalization)
        total = tf.sum()
        if total > 0:
            tf = tf / total

        return tf

    def fit(self, texts: list):
        """
        Compute IDF weights from training corpus.

        IDF(t) = log(N / df(t))
        where N = total documents, df(t) = document frequency of term t
        """
        if not self.use_idf:
            return

        N = len(texts)
        doc_freq = np.zeros(self.vocab_size, dtype=np.float32)

        for text in texts:
            seen = set()

            for n in self.n_grams:
                if len(text) < n:
                    continue

                for i in range(len(text) - n + 1):
                    ngram = text[i:i+n]
                    h = self._hash_ngram(ngram)

                    if h not in seen:
                        doc_freq[h] += 1
                        seen.add(h)

        # Compute IDF: log(N / (df + 1))
        # +1 smoothing to avoid division by zero
        self.idf = np.log(N / (doc_freq + 1.0)).astype(np.float32)

        # Clip extreme values
        self.idf = np.clip(self.idf, 0.0, 10.0)

        self.fitted = True

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to vector using n-gram hashing.

        1. Extract n-gram TF features (sparse, vocab_size dims)
        2. Apply IDF weighting (if fitted)
        3. Random projection to dense vector (dim dims)
        4. L2 normalization
        """
        # Extract TF features
        tf = self._extract_ngram_features(text)

        # Apply IDF
        if self.use_idf:
            tf_idf = tf * self.idf
        else:
            tf_idf = tf

        # Random projection to dense vector
        vec = self.projection.T @ tf_idf

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)


class NgramHashMultiscaleEncoder(BaseEncoder):
    """
    Enhanced version: Combines n-gram hashing with multiscale features.

    Uses n-gram hash as primary discriminative feature,
    supplemented with byte-level and structural features.
    """

    def __init__(
        self,
        dim: int = 128,
        ngram_dim_ratio: float = 0.75,
        n_grams: tuple = (2, 3, 4),
        vocab_size: int = 8192,
        seed: int = 42
    ):
        """
        Args:
            dim: Total output dimension
            ngram_dim_ratio: Fraction of dims allocated to n-gram features
            n_grams: N-gram sizes
            vocab_size: Hash table size
            seed: Random seed
        """
        self._dim = dim
        self.ngram_dim = int(dim * ngram_dim_ratio)
        self.aux_dim = dim - self.ngram_dim

        # N-gram hash encoder
        self.ngram_encoder = NgramHashEncoder(
            dim=self.ngram_dim,
            n_grams=n_grams,
            vocab_size=vocab_size,
            seed=seed,
            use_idf=True
        )

        np.random.seed(seed + 1)

        # Byte-level projection for auxiliary features
        self.byte_projection = np.random.randn(256, self.aux_dim).astype(np.float32)
        self.byte_projection /= np.sqrt(self.aux_dim)

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: list):
        """Fit IDF weights on training corpus."""
        self.ngram_encoder.fit(texts)

    def _extract_byte_features(self, text: str) -> np.ndarray:
        """Extract byte-level distribution features."""
        byte_hist = np.zeros(256, dtype=np.float32)

        try:
            text_bytes = text.encode('utf-8', errors='ignore')
            for b in text_bytes:
                byte_hist[b] += 1

            if len(text_bytes) > 0:
                byte_hist /= len(text_bytes)
        except:
            pass

        # Project to lower dimension
        vec = self.byte_projection.T @ byte_hist
        return vec

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text combining n-gram hash and byte features.

        1. N-gram hash features (75% of dims)
        2. Byte-level features (25% of dims)
        3. Concatenate and normalize
        """
        # Extract features
        ngram_vec = self.ngram_encoder.encode(text)
        byte_vec = self._extract_byte_features(text)

        # Ensure correct dimensions
        ngram_vec = ngram_vec[:self.ngram_dim]
        byte_vec = byte_vec[:self.aux_dim]

        # Pad if needed
        if len(ngram_vec) < self.ngram_dim:
            ngram_vec = np.concatenate([
                ngram_vec,
                np.zeros(self.ngram_dim - len(ngram_vec), dtype=np.float32)
            ])

        if len(byte_vec) < self.aux_dim:
            byte_vec = np.concatenate([
                byte_vec,
                np.zeros(self.aux_dim - len(byte_vec), dtype=np.float32)
            ])

        # Concatenate
        vec = np.concatenate([ngram_vec, byte_vec])

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
