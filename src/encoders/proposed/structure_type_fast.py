"""Numba-accelerated Structure Type Encoder (Experiment 2.1)

JIT-compiled version for improved encoding speed.
"""

import numpy as np
import re
import unicodedata
from numba import jit
from ..base import BaseEncoder


# JIT-compiled helper functions
@jit(nopython=True)
def extract_byte_hist_fast(text_bytes):
    """Fast byte histogram extraction using Numba."""
    byte_hist = np.zeros(256, dtype=np.float32)

    for b in text_bytes:
        byte_hist[b] += 1.0

    # Normalize
    total = len(text_bytes)
    if total > 0:
        for i in range(256):
            byte_hist[i] /= total

    return byte_hist


@jit(nopython=True)
def extract_char_stats_fast(text_array, text_len):
    """
    Fast character statistics extraction.

    Args:
        text_array: numpy array of character codes (int32)
        text_len: length of text

    Returns:
        Feature vector with character statistics
    """
    features = np.zeros(16, dtype=np.float32)

    if text_len == 0:
        return features

    # Count character classes
    n_alpha = 0
    n_digit = 0
    n_space = 0
    n_punct = 0  # .,;:!?
    n_upper = 0
    n_lower = 0
    n_bracket = 0  # ()[]{}
    n_angle = 0  # <>/
    n_special = 0  # @#$%&*
    n_math = 0  # +-=|\

    unique_chars = 0
    char_seen = np.zeros(65536, dtype=np.uint8)  # Track unique chars

    for i in range(text_len):
        code = text_array[i]

        # Track unique
        if code < 65536 and char_seen[code] == 0:
            char_seen[code] = 1
            unique_chars += 1

        # Alpha
        if (65 <= code <= 90) or (97 <= code <= 122):  # A-Z, a-z
            n_alpha += 1

        # Digit
        if 48 <= code <= 57:  # 0-9
            n_digit += 1

        # Space
        if code == 32 or code == 9 or code == 10 or code == 13:
            n_space += 1

        # Punctuation
        if code in (46, 44, 59, 58, 33, 63):  # . , ; : ! ?
            n_punct += 1

        # Upper/lower
        if 65 <= code <= 90:  # A-Z
            n_upper += 1
        if 97 <= code <= 122:  # a-z
            n_lower += 1

        # Brackets
        if code in (40, 41, 91, 93, 123, 125):  # ( ) [ ] { }
            n_bracket += 1

        # Angle/slash
        if code in (60, 62, 47):  # < > /
            n_angle += 1

        # Special
        if code in (64, 35, 36, 37, 38, 42):  # @ # $ % & *
            n_special += 1

        # Math
        if code in (43, 45, 61, 124, 92):  # + - = | \
            n_math += 1

    # Compute ratios
    features[0] = min(float(text_len) / 1000.0, 1.0)  # Length
    features[1] = float(unique_chars) / min(float(text_len), 256.0)  # Unique ratio
    features[2] = float(n_alpha) / float(text_len)
    features[3] = float(n_digit) / float(text_len)
    features[4] = float(n_space) / float(text_len)
    features[5] = float(n_punct) / float(text_len)
    features[6] = float(n_upper) / float(text_len)
    features[7] = float(n_lower) / float(text_len)
    features[8] = float(n_bracket) / float(text_len)
    features[9] = float(n_angle) / float(text_len)
    features[10] = float(n_special) / float(text_len)
    features[11] = float(n_math) / float(text_len)

    # Simple entropy (approximation)
    if text_len > 0 and unique_chars > 0:
        features[12] = float(unique_chars) / 256.0

    return features


@jit(nopython=True)
def project_vector_fast(sparse_vec, projection_matrix):
    """
    Fast matrix-vector multiplication for projection.

    Args:
        sparse_vec: float32 array
        projection_matrix: float32 array (vocab_size, dim)

    Returns:
        float32 array (dim,)
    """
    # Manual transpose @ multiply for type safety
    dim = projection_matrix.shape[1]
    result = np.zeros(dim, dtype=np.float32)

    for i in range(dim):
        for j in range(len(sparse_vec)):
            result[i] += projection_matrix[j, i] * sparse_vec[j]

    return result


class StructureTypeFastEncoder(BaseEncoder):
    """
    Experiment 2.1: Numba JIT-accelerated Structure Type Encoder

    Hypothesis: JIT compilation of hot loops will achieve 2-5x speed improvement.

    Expected improvement:
    - Speed: 9,340/s → 20,000+/s
    - Quality: Same as StructureTypeEncoder (no degradation)
    """

    # Structure type patterns (same as original)
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

    TYPE_IDS = {
        'url': 0, 'email': 1, 'json': 2, 'xml': 3, 'html': 4,
        'filepath_win': 5, 'filepath_unix': 6, 'ipv4': 7, 'ipv6': 8,
        'phone': 9, 'hash_md5': 10, 'hash_sha': 11, 'uuid': 12,
        'base64': 13, 'date_iso': 14, 'time': 15,
        'code_function': 16, 'code_control': 17, 'sql': 18, 'csv': 19,
        'korean': 20, 'japanese': 21, 'chinese': 22,
        'text': 23
    }

    def __init__(self, dim: int = 128, type_dim: int = 16, seed: int = 42):
        """
        Args:
            dim: Total output vector dimension
            type_dim: Dimensions for type encoding
            seed: Random seed
        """
        if dim <= type_dim:
            raise ValueError(f"dim ({dim}) must be larger than type_dim ({type_dim})")

        self._dim = dim
        self.type_dim = type_dim
        self.content_dim = dim - type_dim

        np.random.seed(seed)

        # Random projections (pre-computed, contiguous arrays for Numba)
        self.byte_projection = np.ascontiguousarray(
            np.random.randn(256, self.content_dim // 2).astype(np.float32) / np.sqrt(self.content_dim // 2)
        )
        self.char_projection = np.ascontiguousarray(
            np.random.randn(16, self.content_dim // 2).astype(np.float32) / np.sqrt(self.content_dim // 2)
        )

    @property
    def dim(self) -> int:
        return self._dim

    def detect_type(self, text: str) -> str:
        """Detect structure type using regex patterns."""
        for type_name, pattern in self.PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return type_name
        return 'text'

    def _encode_type_vector(self, type_name: str) -> np.ndarray:
        """Encode type as sparse one-hot vector."""
        type_id = self.TYPE_IDS.get(type_name, self.TYPE_IDS['text'])

        type_vec = np.zeros(self.type_dim, dtype=np.float32)
        type_vec[type_id % self.type_dim] = 1.0

        # Add small noise to related types
        if type_id + 1 < self.type_dim:
            type_vec[(type_id + 1) % self.type_dim] = 0.1

        # L2 normalize
        norm = np.linalg.norm(type_vec)
        if norm > 1e-10:
            type_vec = type_vec / norm

        return type_vec

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text with JIT-accelerated feature extraction.

        1. Detect type (regex, not JIT-able)
        2. Extract byte features (JIT)
        3. Extract char stats (JIT)
        4. Project and combine
        """
        # Detect type
        type_name = self.detect_type(text)
        type_vec = self._encode_type_vector(type_name)

        # Convert text to bytes and char array for JIT functions
        try:
            text_bytes = np.frombuffer(text.encode('utf-8', errors='ignore'), dtype=np.uint8)
        except:
            text_bytes = np.zeros(0, dtype=np.uint8)

        # Character codes as int32 array
        text_codes = np.array([ord(c) for c in text], dtype=np.int32)

        # Extract features using JIT functions
        byte_hist = extract_byte_hist_fast(text_bytes)
        char_stats = extract_char_stats_fast(text_codes, len(text_codes))

        # Project to lower dimensions
        byte_vec = project_vector_fast(byte_hist, self.byte_projection)
        char_vec = project_vector_fast(char_stats, self.char_projection)

        # Combine content features
        content_vec = np.concatenate([byte_vec, char_vec])

        # Ensure correct size
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
