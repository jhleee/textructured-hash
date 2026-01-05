"""Learned Feature Weights encoder"""

import numpy as np
from ..base import BaseEncoder


class LearnedWeightsEncoder(BaseEncoder):
    """
    Learned Feature Weights (P2)

    고정 피처 추출 + 학습된 가중치 행렬
    Contrastive loss로 최적화
    """

    def __init__(self, dim: int = 128, feature_dim: int = 200, seed: int = 42):
        """
        Args:
            dim: 출력 벡터 차원
            feature_dim: 입력 피처 차원
            seed: 랜덤 시드
        """
        self._dim = dim
        self.feature_dim = feature_dim

        np.random.seed(seed)
        # Initialize weight matrix (will be learned)
        self.W = np.random.randn(feature_dim, dim).astype(np.float32) * 0.01
        self.fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    def _extract_features(self, text: str) -> np.ndarray:
        """
        고정된 휴리스틱 피처 추출

        200개의 다양한 구조적 피처
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        text_len = len(text)

        # Length features (0-9)
        features[0] = min(text_len / 1000.0, 1.0)
        features[1] = min(np.log1p(text_len) / 10.0, 1.0)

        if text_len > 0:
            # Character type ratios (10-19)
            features[10] = sum(c.isalpha() for c in text) / text_len
            features[11] = sum(c.isdigit() for c in text) / text_len
            features[12] = sum(c.isspace() for c in text) / text_len
            features[13] = sum(c.isupper() for c in text) / text_len
            features[14] = sum(c.islower() for c in text) / text_len

            # Punctuation (20-29)
            features[20] = text.count('.') / text_len
            features[21] = text.count(',') / text_len
            features[22] = text.count(':') / text_len
            features[23] = text.count(';') / text_len
            features[24] = text.count('!') / text_len
            features[25] = text.count('?') / text_len

            # Special characters (30-49)
            features[30] = text.count('@') / text_len
            features[31] = text.count('#') / text_len
            features[32] = text.count('$') / text_len
            features[33] = text.count('%') / text_len
            features[34] = text.count('&') / text_len
            features[35] = text.count('*') / text_len
            features[36] = text.count('(') / text_len
            features[37] = text.count(')') / text_len
            features[38] = text.count('[') / text_len
            features[39] = text.count(']') / text_len
            features[40] = text.count('{') / text_len
            features[41] = text.count('}') / text_len
            features[42] = text.count('<') / text_len
            features[43] = text.count('>') / text_len
            features[44] = text.count('/') / text_len
            features[45] = text.count('\\') / text_len
            features[46] = text.count('-') / text_len
            features[47] = text.count('_') / text_len
            features[48] = text.count('=') / text_len
            features[49] = text.count('+') / text_len

        # Token-level features (50-69)
        tokens = text.split()
        features[50] = min(len(tokens) / 100.0, 1.0)

        if tokens:
            token_lengths = [len(t) for t in tokens]
            features[51] = np.mean(token_lengths) / 50.0
            features[52] = np.max(token_lengths) / 100.0
            features[53] = np.min(token_lengths) / 20.0
            features[54] = np.std(token_lengths) / 20.0 if len(token_lengths) > 1 else 0

        # Pattern indicators (70-99)
        features[70] = 1.0 if text.startswith('http://') or text.startswith('https://') else 0.0
        features[71] = 1.0 if text.startswith('ftp://') else 0.0
        features[72] = 1.0 if '@' in text and '.' in text else 0.0  # Email-like
        features[73] = 1.0 if text.count('.') >= 3 and text.count('@') == 0 else 0.0  # IP-like
        features[74] = 1.0 if text.startswith('{') or text.startswith('[') else 0.0  # JSON-like
        features[75] = 1.0 if text.startswith('<') and text.endswith('>') else 0.0  # XML-like
        features[76] = 1.0 if text.count(',') >= 2 and text.count('\n') == 0 else 0.0  # CSV-like
        features[77] = 1.0 if text.startswith('/') or text.startswith('C:') else 0.0  # Path-like
        features[78] = 1.0 if any(word in text.lower() for word in ['function', 'const', 'let', 'var']) else 0.0  # Code-like
        features[79] = 1.0 if any(word in text.lower() for word in ['select', 'insert', 'update', 'delete']) else 0.0  # SQL-like

        # Unicode ranges (100-119)
        if text_len > 0:
            # Latin
            features[100] = sum(0x41 <= ord(c) <= 0x5A or 0x61 <= ord(c) <= 0x7A for c in text) / text_len
            # Digits
            features[101] = sum(0x30 <= ord(c) <= 0x39 for c in text) / text_len
            # Hangul
            features[102] = sum(0xAC00 <= ord(c) <= 0xD7AF for c in text) / text_len
            # CJK
            features[103] = sum(0x4E00 <= ord(c) <= 0x9FFF for c in text) / text_len

        # Character diversity (120-129)
        if text_len > 0:
            unique_chars = len(set(text))
            features[120] = unique_chars / text_len
            features[121] = unique_chars / 256.0

        # Bigram features (130-149) - hashed
        if text_len >= 2:
            bigrams = [text[i:i+2] for i in range(text_len-1)]
            for i in range(20):
                features[130 + i] = sum(hash(bg) % 1000 == i * 50 for bg in bigrams) / len(bigrams)

        # Trigram features (150-169) - hashed
        if text_len >= 3:
            trigrams = [text[i:i+3] for i in range(text_len-2)]
            for i in range(20):
                features[150 + i] = sum(hash(tg) % 1000 == i * 50 for tg in trigrams) / len(trigrams)

        # Byte-level features (170-199)
        try:
            text_bytes = text.encode('utf-8', errors='ignore')
            if len(text_bytes) > 0:
                # Byte value statistics
                byte_vals = np.array(list(text_bytes))
                features[170] = np.mean(byte_vals) / 255.0
                features[171] = np.std(byte_vals) / 128.0
                features[172] = np.min(byte_vals) / 255.0
                features[173] = np.max(byte_vals) / 255.0

                # Byte range indicators
                features[180] = sum(b < 32 for b in text_bytes) / len(text_bytes)  # Control chars
                features[181] = sum(32 <= b < 127 for b in text_bytes) / len(text_bytes)  # ASCII printable
                features[182] = sum(b >= 127 for b in text_bytes) / len(text_bytes)  # Non-ASCII
        except:
            pass

        return features

    def fit(self, positive_pairs: list, negative_pairs: list, epochs: int = 10, lr: float = 0.01):
        """
        Contrastive loss로 가중치 학습

        Args:
            positive_pairs: (text1, text2) pairs that should be similar
            negative_pairs: (text1, text2) pairs that should be dissimilar
            epochs: 학습 에폭 수
            lr: 학습률
        """
        print(f"Training LearnedWeights encoder...")
        print(f"  Positive pairs: {len(positive_pairs)}")
        print(f"  Negative pairs: {len(negative_pairs)}")
        print(f"  Epochs: {epochs}, Learning rate: {lr}")

        for epoch in range(epochs):
            total_loss = 0
            n_updates = 0

            # Positive pairs: pull together
            for text1, text2 in positive_pairs:
                f1 = self._extract_features(text1)
                f2 = self._extract_features(text2)

                v1 = self.W.T @ f1
                v2 = self.W.T @ f2

                # Normalize
                v1 = v1 / (np.linalg.norm(v1) + 1e-10)
                v2 = v2 / (np.linalg.norm(v2) + 1e-10)

                # Loss: -cosine_similarity (we want to maximize similarity)
                loss = -np.dot(v1, v2)
                total_loss += loss

                # Gradient update (simplified)
                grad = -2 * np.outer(f1, v2) - 2 * np.outer(f2, v1)
                self.W -= lr * grad
                n_updates += 1

            # Negative pairs: push apart (with margin)
            margin = 0.3
            for text1, text2 in negative_pairs:
                f1 = self._extract_features(text1)
                f2 = self._extract_features(text2)

                v1 = self.W.T @ f1
                v2 = self.W.T @ f2

                # Normalize
                v1 = v1 / (np.linalg.norm(v1) + 1e-10)
                v2 = v2 / (np.linalg.norm(v2) + 1e-10)

                # Loss: max(0, cosine_similarity - margin)
                sim = np.dot(v1, v2)
                if sim > margin:
                    loss = sim - margin
                    total_loss += loss

                    # Gradient update
                    grad = 2 * np.outer(f1, v2) + 2 * np.outer(f2, v1)
                    self.W -= lr * grad
                    n_updates += 1

            avg_loss = total_loss / (n_updates + 1)
            print(f"  Epoch {epoch + 1}/{epochs}: avg_loss = {avg_loss:.4f}")

        self.fitted = True
        print("✓ LearnedWeights model trained")

    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 LearnedWeights 벡터로 인코딩

        1. 피처 추출
        2. 학습된 가중치 행렬로 변환
        3. L2 정규화
        """
        features = self._extract_features(text)

        # Linear transformation
        vec = self.W.T @ features

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
