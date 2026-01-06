"""MinHash baseline encoder"""

import numpy as np
from .base import BaseEncoder


class MinHashEncoder(BaseEncoder):
    """
    MinHash baseline

    Character n-gram에 대한 MinHash 서명 생성
    """

    def __init__(self, dim: int = 128, ngram_size: int = 3, seed: int = 42):
        """
        Args:
            dim: 출력 벡터 차원 (해시 함수 개수)
            ngram_size: n-gram 크기
            seed: 랜덤 시드
        """
        self._dim = dim
        self.ngram_size = ngram_size
        self.seed = seed

        # Generate hash function seeds
        np.random.seed(seed)
        self.hash_seeds = np.random.randint(0, 2**31, size=dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _hash_with_seed(self, item: str, seed: int) -> int:
        """Hash item with specific seed"""
        return hash((item, seed)) & 0x7FFFFFFF  # Positive 31-bit hash

    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 MinHash 서명 벡터로 인코딩

        1. Character n-gram 집합 생성
        2. 각 해시 함수에 대해 최소 해시값 계산
        3. 정규화하여 실수 벡터로 변환
        """
        # Extract n-gram set
        ngrams = set()
        if len(text) >= self.ngram_size:
            for i in range(len(text) - self.ngram_size + 1):
                ngrams.add(text[i:i + self.ngram_size])
        else:
            ngrams.add(text if text else ' ')

        # Initialize signature
        signature = np.full(self._dim, np.inf, dtype=np.float64)

        # Compute MinHash signature
        for gram in ngrams:
            for i in range(self._dim):
                h = self._hash_with_seed(gram, int(self.hash_seeds[i]))
                signature[i] = min(signature[i], h)

        # Normalize to [0, 1]
        vec = signature.astype(np.float32) / (2**31)

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec
