"""SimHash baseline encoder"""

import numpy as np
from .base import BaseEncoder


class SimHashEncoder(BaseEncoder):
    """
    SimHash baseline

    텍스트의 character n-gram에 대한 SimHash 생성
    """

    def __init__(self, dim: int = 128, ngram_size: int = 3):
        """
        Args:
            dim: 출력 벡터 차원 (해시 비트 수)
            ngram_size: n-gram 크기
        """
        self._dim = dim
        self.ngram_size = ngram_size

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 SimHash 벡터로 인코딩

        1. Character n-gram 추출
        2. 각 n-gram의 해시값을 비트로 변환
        3. 비트별 가중 합산
        4. 부호에 따라 0/1 결정
        5. 실수 벡터로 변환 및 정규화
        """
        # Initialize bit vector
        v = np.zeros(self._dim, dtype=np.float32)

        # Extract n-grams
        ngrams = []
        if len(text) >= self.ngram_size:
            for i in range(len(text) - self.ngram_size + 1):
                ngrams.append(text[i:i + self.ngram_size])
        else:
            # For very short text, use the whole text
            ngrams.append(text)

        if not ngrams:
            ngrams = [' ']  # Fallback for empty text

        # SimHash algorithm
        for gram in ngrams:
            h = hash(gram)
            for i in range(self._dim):
                if (h >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1

        # Convert to binary-like float vector
        vec = (v > 0).astype(np.float32)

        # Convert to {-1, 1} for better cosine similarity
        vec = 2 * vec - 1

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec
