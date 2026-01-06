"""Random Projection baseline encoder"""

import numpy as np
from .base import BaseEncoder


class RandomProjectionEncoder(BaseEncoder):
    """
    Random Projection (RP) baseline

    문자 빈도 벡터를 랜덤 투영으로 저차원 벡터로 변환
    """

    def __init__(self, output_dim: int = 128, input_dim: int = 1024, seed: int = 42):
        """
        Args:
            output_dim: 출력 벡터 차원
            input_dim: 입력 피처 차원 (문자 해시 공간)
            seed: 랜덤 시드
        """
        self._dim = output_dim
        self.input_dim = input_dim

        np.random.seed(seed)
        # Random projection matrix (Gaussian)
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(output_dim)
        self.projection = self.projection.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 벡터로 인코딩

        1. 문자를 해시하여 빈도 벡터 생성
        2. 랜덤 투영으로 차원 축소
        3. L2 정규화
        """
        # Character frequency vector (hashed)
        char_freq = np.zeros(self.input_dim, dtype=np.float32)

        for c in text:
            h = hash(c) % self.input_dim
            char_freq[h] += 1

        # Normalize by text length
        if len(text) > 0:
            char_freq /= len(text)

        # Random projection
        vec = self.projection.T @ char_freq

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
