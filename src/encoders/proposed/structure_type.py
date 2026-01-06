"""Statistical Structure Encoder (Experiment 1.2 - Pattern-Free Version)

Multi-scale statistical encoding without hardcoded patterns.
Removed all regex patterns for better generalization to unseen data formats.
"""

import numpy as np
import unicodedata
from ..base import BaseEncoder


class StructureTypeEncoder(BaseEncoder):
    """
    Experiment 1.2 (Pattern-Free): Statistical Structure Encoding

    REMOVED: Hardcoded regex patterns for URL, email, JSON, etc.

    NEW APPROACH: Pure statistical features for generalization
    - No pattern matching, no type detection
    - Statistical features only: byte distribution, unicode categories, n-grams
    - Better generalization to unknown/unstructured data formats

    Trade-off:
    - Lower accuracy on known structured types
    - Better robustness to format variations
    - No maintenance overhead for pattern updates
    """

    def __init__(self, dim: int = 128, type_dim: int = 16, seed: int = 42):
        """
        Args:
            dim: Total output vector dimension
            type_dim: Dimensions for statistical signature (replaces type encoding)
            seed: Random seed for reproducibility
        """
        if dim <= type_dim:
            raise ValueError(f"dim ({dim}) must be larger than type_dim ({type_dim})")

        self._dim = dim
        self.type_dim = type_dim  # Now used for statistical signature
        self.content_dim = dim - type_dim

        np.random.seed(seed)

        # Random projections for content features
        self.byte_projection = np.random.randn(256, self.content_dim // 4).astype(np.float32) / np.sqrt(self.content_dim // 4)
        self.unicode_projection = np.random.randn(32, self.content_dim // 4).astype(np.float32) / np.sqrt(self.content_dim // 4)

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_statistical_signature(self, text: str) -> np.ndarray:
        """
        Encode statistical signature instead of hardcoded type detection.

        Uses character-level statistics to create a signature vector.
        No pattern matching - purely statistical.
        """
        features = np.zeros(self.type_dim, dtype=np.float32)

        if len(text) == 0:
            return features

        # Statistical features that capture text "type" without hardcoding
        # Feature 0-3: Character class ratios
        features[0] = sum(c.isalpha() for c in text) / len(text)
        features[1] = sum(c.isdigit() for c in text) / len(text)
        features[2] = sum(c.isspace() for c in text) / len(text)
        features[3] = sum(c in '.,;:!?' for c in text) / len(text)

        # Feature 4-7: Special character ratios
        features[4] = sum(c in '()[]{}' for c in text) / len(text)
        features[5] = sum(c in '<>/' for c in text) / len(text)
        features[6] = sum(c in '@#$%&*' for c in text) / len(text)
        features[7] = sum(c in '+-=|\\' for c in text) / len(text)

        # Feature 8-10: Case statistics
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count > 0:
            features[8] = sum(c.isupper() for c in text) / alpha_count
            features[9] = sum(c.islower() for c in text) / alpha_count

        # Feature 10-11: Character diversity
        features[10] = len(set(text)) / min(len(text), 256)
        features[11] = len(set(text)) / 256.0

        # Feature 12-13: Position features
        features[12] = 1.0 if text[0].isupper() else 0.0
        features[13] = 1.0 if text[0].isdigit() else 0.0

        # Feature 14-15: Length features
        features[14] = min(len(text) / 1000.0, 1.0)
        if len(text) >= 2:
            # Bigram diversity
            bigrams = [text[i:i+2] for i in range(len(text)-1)]
            features[15] = len(set(bigrams)) / len(bigrams)

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 1e-10:
            features = features / norm

        return features

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

    def _encode_content_structural(self, text: str) -> np.ndarray:
        """
        Extract statistical structural features (no type-specific logic).

        Replaces hardcoded type-specific features with general statistical patterns.
        """
        dim = self.content_dim // 4
        features = np.zeros(dim, dtype=np.float32)

        if len(text) == 0:
            return features

        # General character frequency features (no type assumptions)
        features[0] = text.count('/') / max(len(text), 1)
        features[1] = text.count('.') / max(len(text), 1)
        features[2] = text.count('@') / max(len(text), 1)
        features[3] = text.count(':') / max(len(text), 1)

        if dim > 4:
            features[4] = text.count('{') / max(len(text), 1)
            features[5] = text.count('[') / max(len(text), 1)
            features[6] = text.count('<') / max(len(text), 1)
            features[7] = text.count('(') / max(len(text), 1)

        if dim > 8:
            features[8] = text.count(';') / max(len(text), 1)
            features[9] = text.count('=') / max(len(text), 1)
            features[10] = text.count('?') / max(len(text), 1)
            features[11] = text.count('&') / max(len(text), 1)

        # Statistical patterns (not type-specific)
        if dim > 12:
            # Character transition entropy
            if len(text) >= 2:
                transitions = {}
                for i in range(len(text)-1):
                    pair = (text[i], text[i+1])
                    transitions[pair] = transitions.get(pair, 0) + 1

                if transitions:
                    total = sum(transitions.values())
                    entropy = 0
                    for count in transitions.values():
                        p = count / total
                        entropy -= p * np.log2(p + 1e-10)
                    features[12] = entropy / 10.0

        return features

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text with statistical features (no pattern matching).

        1. Extract statistical signature (replaces type detection)
        2. Extract multi-scale content features
        3. Concatenate and normalize

        NO hardcoded patterns, NO regex, NO type detection.
        """
        # Statistical signature (replaces type vector)
        stat_sig = self._encode_statistical_signature(text)

        # Extract content features (4 parts)
        byte_vec = self._encode_content_byte_level(text)
        unicode_vec = self._encode_content_unicode_level(text)
        char_stats_vec = self._encode_content_char_stats(text)
        structural_vec = self._encode_content_structural(text)

        # Concatenate all parts
        content_vec = np.concatenate([byte_vec, unicode_vec, char_stats_vec, structural_vec])

        # Ensure content_vec is correct size
        if len(content_vec) > self.content_dim:
            content_vec = content_vec[:self.content_dim]
        elif len(content_vec) < self.content_dim:
            padding = np.zeros(self.content_dim - len(content_vec), dtype=np.float32)
            content_vec = np.concatenate([content_vec, padding])

        # Combine statistical signature and content
        vec = np.concatenate([stat_sig, content_vec])

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
