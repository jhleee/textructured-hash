"""Structure Type Detection Encoder (Experiment 1.2)

Multi-type structure-aware encoding with conditional feature extraction.
"""

import numpy as np
import re
import unicodedata
from ..base import BaseEncoder


class StructureTypeEncoder(BaseEncoder):
    """
    Experiment 1.2: Structure Type Detection + Conditional Encoding

    Hypothesis: Detecting text structure types first, then applying type-specific
    encoding will increase separation between different types.

    Expected improvement:
    - Separation: 1.21 → 2.8+ (type separation effect)
    - Mean Neg Sim: Lower due to type vector differences
    """

    # Structure type patterns (ordered by specificity)
    PATTERNS = {
        'url': r'^https?://|^ftp://|^www\.',
        'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
        'json': r'^\s*[\{\[].*[\}\]]\s*$',
        'xml': r'^\s*<\w+',
        'html': r'<!DOCTYPE|<html|<body|<div|<span|<p>',
        'filepath_win': r'^[A-Z]:\\',
        'filepath_unix': r'^/[a-z]',
        'ipv4': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        'ipv6': r'^[0-9a-fA-F:]+::[0-9a-fA-F:]*',
        'phone': r'^\+?\d[\d\-\s\(\)]{7,}$',
        'hash_md5': r'^[a-fA-F0-9]{32}$',
        'hash_sha': r'^[a-fA-F0-9]{40,64}$',
        'uuid': r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$',
        'base64': r'^[A-Za-z0-9+/]{20,}={0,2}$',
        'date_iso': r'^\d{4}-\d{2}-\d{2}',
        'time': r'\d{1,2}:\d{2}(:\d{2})?',
        'code_function': r'(function|def|class|const|let|var)\s+\w+',
        'code_control': r'(if|for|while|switch|try)\s*\(',
        'sql': r'(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\s+',
        'csv': r'^[\w,]+,[\w,]+',
        'korean': r'[\uAC00-\uD7AF]{3,}',
        'japanese': r'[\u3040-\u309F\u30A0-\u30FF]{3,}',
        'chinese': r'[\u4E00-\u9FFF]{3,}',
    }

    # Type ID mapping
    TYPE_IDS = {
        'url': 0, 'email': 1, 'json': 2, 'xml': 3, 'html': 4,
        'filepath_win': 5, 'filepath_unix': 6, 'ipv4': 7, 'ipv6': 8,
        'phone': 9, 'hash_md5': 10, 'hash_sha': 11, 'uuid': 12,
        'base64': 13, 'date_iso': 14, 'time': 15,
        'code_function': 16, 'code_control': 17, 'sql': 18, 'csv': 19,
        'korean': 20, 'japanese': 21, 'chinese': 22,
        'text': 23  # default
    }

    def __init__(self, dim: int = 128, type_dim: int = 16, seed: int = 42):
        """
        Args:
            dim: Total output vector dimension
            type_dim: Dimensions allocated for type encoding (one-hot style)
            seed: Random seed for reproducibility
        """
        if dim <= type_dim:
            raise ValueError(f"dim ({dim}) must be larger than type_dim ({type_dim})")

        self._dim = dim
        self.type_dim = type_dim
        self.content_dim = dim - type_dim

        np.random.seed(seed)

        # Random projections for content features
        self.byte_projection = np.random.randn(256, self.content_dim // 4).astype(np.float32) / np.sqrt(self.content_dim // 4)
        self.unicode_projection = np.random.randn(32, self.content_dim // 4).astype(np.float32) / np.sqrt(self.content_dim // 4)

    @property
    def dim(self) -> int:
        return self._dim

    def detect_type(self, text: str) -> str:
        """
        Detect structure type using regex patterns.

        Returns type name (e.g., 'url', 'email', 'json', 'text')
        """
        # Try each pattern in order
        for type_name, pattern in self.PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return type_name

        return 'text'  # default fallback

    def _encode_type_vector(self, type_name: str) -> np.ndarray:
        """
        Encode type as a sparse one-hot-like vector.

        Uses distributed representation to allow some similarity between related types.
        """
        type_id = self.TYPE_IDS.get(type_name, self.TYPE_IDS['text'])

        # Create one-hot vector
        type_vec = np.zeros(self.type_dim, dtype=np.float32)
        type_vec[type_id % self.type_dim] = 1.0

        # Add small noise to related types for smoother boundaries
        if type_id + 1 < self.type_dim:
            type_vec[(type_id + 1) % self.type_dim] = 0.1

        # L2 normalize
        norm = np.linalg.norm(type_vec)
        if norm > 1e-10:
            type_vec = type_vec / norm

        return type_vec

    def _encode_content_byte_level(self, text: str) -> np.ndarray:
        """Extract byte-level content features."""
        byte_hist = np.zeros(256, dtype=np.float32)

        try:
            text_bytes = text.encode('utf-8', errors='ignore')
            for b in text_bytes:
                byte_hist[b] += 1

            if len(text_bytes) > 0:
                byte_hist /= len(text_bytes)
        except:
            pass

        # Random projection
        vec = self.byte_projection.T @ byte_hist
        return vec

    def _encode_content_unicode_level(self, text: str) -> np.ndarray:
        """Extract unicode category distribution features."""
        categories = {
            'L': 0, 'N': 1, 'P': 2, 'S': 3, 'Z': 4, 'M': 5, 'C': 6,
        }

        scripts = {
            'latin': 7, 'digit': 8, 'hangul': 9, 'cjk': 10,
            'hiragana': 11, 'katakana': 12, 'cyrillic': 13, 'arabic': 14,
        }

        features = np.zeros(32, dtype=np.float32)

        for char in text:
            cat = unicodedata.category(char)[0]
            if cat in categories:
                features[categories[cat]] += 1

            code = ord(char)
            if 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A:
                features[scripts['latin']] += 1
            elif 0x30 <= code <= 0x39:
                features[scripts['digit']] += 1
            elif 0xAC00 <= code <= 0xD7AF:
                features[scripts['hangul']] += 1
            elif 0x4E00 <= code <= 0x9FFF:
                features[scripts['cjk']] += 1
            elif 0x3040 <= code <= 0x309F:
                features[scripts['hiragana']] += 1
            elif 0x30A0 <= code <= 0x30FF:
                features[scripts['katakana']] += 1
            elif 0x0400 <= code <= 0x04FF:
                features[scripts['cyrillic']] += 1
            elif 0x0600 <= code <= 0x06FF:
                features[scripts['arabic']] += 1

        if len(text) > 0:
            features /= len(text)

        vec = self.unicode_projection.T @ features
        return vec

    def _encode_content_char_stats(self, text: str) -> np.ndarray:
        """Extract character-level statistics."""
        dim = self.content_dim // 4
        features = np.zeros(dim, dtype=np.float32)

        if len(text) == 0:
            return features

        # Basic statistics
        features[0] = min(len(text) / 1000.0, 1.0)  # Length
        features[1] = len(set(text)) / min(len(text), 256)  # Unique char ratio

        # Character class ratios
        features[2] = sum(c.isalpha() for c in text) / len(text)
        features[3] = sum(c.isdigit() for c in text) / len(text)
        features[4] = sum(c.isspace() for c in text) / len(text)
        features[5] = sum(c in '.,;:!?' for c in text) / len(text)
        features[6] = sum(c.isupper() for c in text) / len(text)
        features[7] = sum(c.islower() for c in text) / len(text)

        # Special characters
        if dim > 8:
            features[8] = sum(c in '()[]{}' for c in text) / len(text)
            features[9] = sum(c in '<>/' for c in text) / len(text)
            features[10] = sum(c in '@#$%&*' for c in text) / len(text)
            features[11] = sum(c in '+-=|\\' for c in text) / len(text)

        # Entropy
        if dim > 12:
            char_counts = {}
            for c in text:
                char_counts[c] = char_counts.get(c, 0) + 1

            entropy = 0
            for count in char_counts.values():
                p = count / len(text)
                entropy -= p * np.log2(p + 1e-10)

            features[12] = entropy / 8.0

        return features

    def _encode_content_structural(self, text: str, type_name: str) -> np.ndarray:
        """Extract type-specific structural features."""
        dim = self.content_dim // 4
        features = np.zeros(dim, dtype=np.float32)

        # URL-specific features
        if type_name == 'url':
            features[0] = 1.0 if 'https://' in text else 0.5 if 'http://' in text else 0.0
            features[1] = text.count('/') / max(len(text), 1)
            features[2] = text.count('?') / max(len(text), 1)
            features[3] = text.count('&') / max(len(text), 1)

        # Email-specific features
        elif type_name == 'email':
            features[0] = 1.0
            features[1] = text.count('@') / max(len(text), 1)
            features[2] = text.count('.') / max(len(text), 1)

        # JSON/structured data features
        elif type_name in ['json', 'xml', 'html']:
            features[0] = text.count('{') / max(len(text), 1)
            features[1] = text.count('[') / max(len(text), 1)
            features[2] = text.count('<') / max(len(text), 1)
            features[3] = text.count(':') / max(len(text), 1)

        # Code-specific features
        elif 'code' in type_name:
            features[0] = text.count('(') / max(len(text), 1)
            features[1] = text.count(';') / max(len(text), 1)
            features[2] = text.count('=') / max(len(text), 1)

        return features

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text with structure type awareness.

        1. Detect structure type
        2. Encode type as type_vector (16 dims)
        3. Extract type-aware content features (112 dims)
        4. Concatenate and normalize
        """
        # Detect type
        type_name = self.detect_type(text)

        # Encode type vector
        type_vec = self._encode_type_vector(type_name)

        # Extract content features (4 parts)
        byte_vec = self._encode_content_byte_level(text)
        unicode_vec = self._encode_content_unicode_level(text)
        char_stats_vec = self._encode_content_char_stats(text)
        structural_vec = self._encode_content_structural(text, type_name)

        # Concatenate all parts
        content_vec = np.concatenate([byte_vec, unicode_vec, char_stats_vec, structural_vec])

        # Ensure content_vec is correct size
        if len(content_vec) > self.content_dim:
            content_vec = content_vec[:self.content_dim]
        elif len(content_vec) < self.content_dim:
            padding = np.zeros(self.content_dim - len(content_vec), dtype=np.float32)
            content_vec = np.concatenate([content_vec, padding])

        # Combine type and content
        vec = np.concatenate([type_vec, content_vec])

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
