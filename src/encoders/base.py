"""Base encoder interface"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEncoder(ABC):
    """모든 인코더가 구현해야 할 인터페이스"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """출력 벡터 차원"""
        pass

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 벡터로 인코딩

        Args:
            text: UTF-8 문자열

        Returns:
            L2-normalized float32 벡터
        """
        pass

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """배치 인코딩 (기본: 순차 처리)"""
        return np.stack([self.encode(t) for t in texts])

    def similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 유사도 (cosine similarity)"""
        v1, v2 = self.encode(text1), self.encode(text2)
        return float(np.dot(v1, v2))
