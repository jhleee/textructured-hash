"""Multi-Scale Character Statistics encoder"""

import numpy as np
import unicodedata
from ..base import BaseEncoder


class MultiScaleEncoder(BaseEncoder):
    """
    Multi-Scale Character Statistics (P1)

    여러 스케일의 문자 통계를 결합:
    - Byte-level: 바이트 분포
    - Character-level: Unicode 카테고리 분포
    - Token-level: 토큰 통계
    - Pattern-level: 구조적 패턴
    """

    def __init__(self, dim: int = 128, seed: int = 42):
        """
        Args:
            dim: 출력 벡터 차원 (4의 배수여야 함)
        """
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")

        self._dim = dim
        self.sub_dim = dim // 4

        np.random.seed(seed)
        # Random projection matrices for each scale
        self.byte_projection = np.random.randn(256, self.sub_dim).astype(np.float32) / np.sqrt(self.sub_dim)
        self.unicode_projection = np.random.randn(32, self.sub_dim).astype(np.float32) / np.sqrt(self.sub_dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _extract_byte_features(self, text: str) -> np.ndarray:
        """
        Scale 1: Byte-level distribution

        UTF-8 바이트 분포를 랜덤 투영
        """
        byte_hist = np.zeros(256, dtype=np.float32)

        try:
            text_bytes = text.encode('utf-8', errors='ignore')
            for b in text_bytes:
                byte_hist[b] += 1

            # Normalize
            if len(text_bytes) > 0:
                byte_hist /= len(text_bytes)
        except:
            pass

        # Random projection
        vec = self.byte_projection.T @ byte_hist
        return vec

    def _extract_unicode_features(self, text: str) -> np.ndarray:
        """
        Scale 2: Unicode category distribution

        Unicode 카테고리별 분포 (문자, 숫자, 구두점 등)
        """
        # Unicode categories
        categories = {
            'L': 0,   # Letter
            'N': 1,   # Number
            'P': 2,   # Punctuation
            'S': 3,   # Symbol
            'Z': 4,   # Separator
            'M': 5,   # Mark
            'C': 6,   # Other
        }

        # Script detection (approximate)
        scripts = {
            'latin': 7,      # A-Za-z
            'digit': 8,      # 0-9
            'hangul': 9,     # 한글
            'cjk': 10,       # 中日
            'hiragana': 11,  # ひらがな
            'katakana': 12,  # カタカナ
            'cyrillic': 13,  # Кириллица
            'arabic': 14,    # العربية
        }

        features = np.zeros(32, dtype=np.float32)

        for char in text:
            # Unicode category
            cat = unicodedata.category(char)[0]
            if cat in categories:
                features[categories[cat]] += 1

            # Script detection
            code = ord(char)
            if 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A:
                features[scripts['latin']] += 1
            elif 0x30 <= code <= 0x39:
                features[scripts['digit']] += 1
            elif 0xAC00 <= code <= 0xD7AF:
                features[scripts['hangul']] += 1
            elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
                features[scripts['cjk']] += 1
            elif 0x3040 <= code <= 0x309F:
                features[scripts['hiragana']] += 1
            elif 0x30A0 <= code <= 0x30FF:
                features[scripts['katakana']] += 1
            elif 0x0400 <= code <= 0x04FF:
                features[scripts['cyrillic']] += 1
            elif 0x0600 <= code <= 0x06FF:
                features[scripts['arabic']] += 1

        # Normalize
        if len(text) > 0:
            features /= len(text)

        # Random projection
        vec = self.unicode_projection.T @ features
        return vec

    def _extract_token_features(self, text: str) -> np.ndarray:
        """
        Scale 3: Token-level statistics

        공백으로 분리된 토큰들의 통계
        """
        features = np.zeros(self.sub_dim, dtype=np.float32)

        # Tokenize by whitespace
        tokens = text.split()

        # Feature 0-3: Token count statistics
        features[0] = min(len(tokens) / 100.0, 1.0)  # Normalized token count

        if tokens:
            token_lengths = [len(t) for t in tokens]
            features[1] = np.mean(token_lengths) / 50.0  # Average token length
            features[2] = np.std(token_lengths) / 20.0 if len(token_lengths) > 1 else 0  # Std
            features[3] = max(token_lengths) / 100.0  # Max token length

        # Feature 4-7: Character class ratios
        if len(text) > 0:
            features[4] = sum(c.isalpha() for c in text) / len(text)
            features[5] = sum(c.isdigit() for c in text) / len(text)
            features[6] = sum(c.isspace() for c in text) / len(text)
            features[7] = sum(c in '.,;:!?' for c in text) / len(text)

        # Feature 8-11: Uppercase/lowercase ratios
        if len(text) > 0:
            features[8] = sum(c.isupper() for c in text) / len(text)
            features[9] = sum(c.islower() for c in text) / len(text)

        # Feature 12-15: Special characters
        if len(text) > 0:
            features[12] = sum(c in '()[]{}' for c in text) / len(text)
            features[13] = sum(c in '<>/' for c in text) / len(text)
            features[14] = sum(c in '@#$%&*' for c in text) / len(text)
            features[15] = sum(c in '+-=|\\' for c in text) / len(text)

        return features

    def _extract_pattern_features(self, text: str) -> np.ndarray:
        """
        Scale 4: Structural pattern features

        반복, 엔트로피, 구조적 패턴
        """
        features = np.zeros(self.sub_dim, dtype=np.float32)

        # Feature 0: Text length (log scale)
        features[0] = min(np.log1p(len(text)) / 10.0, 1.0)

        # Feature 1-2: Character diversity
        if len(text) > 0:
            unique_chars = len(set(text))
            features[1] = unique_chars / len(text)
            features[2] = unique_chars / 256.0

        # Feature 3-4: Entropy
        if len(text) > 0:
            char_counts = {}
            for c in text:
                char_counts[c] = char_counts.get(c, 0) + 1

            entropy = 0
            for count in char_counts.values():
                p = count / len(text)
                entropy -= p * np.log2(p + 1e-10)

            features[3] = entropy / 8.0  # Normalized by max entropy

        # Feature 5-8: Repetition patterns
        if len(text) >= 2:
            # Character bigram repetition
            bigrams = [text[i:i+2] for i in range(len(text)-1)]
            if bigrams:
                features[5] = len(set(bigrams)) / len(bigrams)

        # Feature 9-12: Structure indicators
        features[9] = 1.0 if text.startswith('http') else 0.0
        features[10] = 1.0 if '@' in text else 0.0
        features[11] = 1.0 if any(c in '{}[]' for c in text) else 0.0
        features[12] = 1.0 if any(c in '<>' for c in text) else 0.0

        # Feature 13-15: Numeric patterns
        if len(text) > 0:
            features[13] = sum(c.isdigit() for c in text) / len(text)
            features[14] = 1.0 if '.' in text and any(c.isdigit() for c in text) else 0.0
            features[15] = 1.0 if ',' in text and any(c.isdigit() for c in text) else 0.0

        return features

    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 MultiScale 벡터로 인코딩

        1. 4개 스케일의 피처 추출
        2. 연결하여 전체 벡터 생성
        3. L2 정규화
        """
        # Extract features at each scale
        byte_vec = self._extract_byte_features(text)
        unicode_vec = self._extract_unicode_features(text)
        token_vec = self._extract_token_features(text)
        pattern_vec = self._extract_pattern_features(text)

        # Concatenate
        vec = np.concatenate([byte_vec, unicode_vec, token_vec, pattern_vec])

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
