"""TF-IDF + SVD baseline encoder"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .base import BaseEncoder


class TfidfSvdEncoder(BaseEncoder):
    """
    TF-IDF + SVD baseline

    Character n-gram TF-IDF를 SVD로 차원 축소
    학습이 필요한 모델 (train set에서 fit)
    """

    def __init__(self, dim: int = 128, max_features: int = 10000):
        """
        Args:
            dim: 출력 벡터 차원
            max_features: TF-IDF 최대 피처 수
        """
        self._dim = dim
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=max_features,
            lowercase=False,  # Preserve case for structure
            dtype=np.float32
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: list):
        """
        학습 데이터로 TF-IDF와 SVD를 fit

        Args:
            texts: 학습 텍스트 리스트
        """
        print(f"Fitting TF-IDF vectorizer on {len(texts)} texts...")
        tfidf = self.vectorizer.fit_transform(texts)

        print(f"Fitting SVD to reduce {tfidf.shape[1]} features to {self._dim} dimensions...")
        self.svd.fit(tfidf)

        self.fitted = True
        print("✓ TF-IDF+SVD model fitted")

    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 TF-IDF+SVD 벡터로 인코딩

        1. Character n-gram TF-IDF 계산
        2. SVD로 차원 축소
        3. L2 정규화
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # TF-IDF transformation
        tfidf = self.vectorizer.transform([text])

        # SVD transformation
        vec = self.svd.transform(tfidf)[0]

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
