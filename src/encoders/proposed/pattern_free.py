"""Pattern-Free Statistical Encoder

Pure statistical approach without any hardcoded patterns or regex.
Uses only statistical features that generalize to all text types.
"""

import numpy as np
import unicodedata
from ..base import BaseEncoder


class PatternFreeEncoder(BaseEncoder):
    """
    Pattern-Free Statistical Text Encoder

    Hypothesis: A purely statistical approach without hardcoded patterns
    can generalize better to unseen/unstructured data types.

    Features used (NO regex, NO hardcoded patterns):
    1. Byte-level distribution (UTF-8 bytes)
    2. Unicode category distribution
    3. Character n-gram statistics (bigrams, trigrams)
    4. Character transition probabilities
    5. Statistical moments (entropy, diversity, etc.)
    6. Positional character statistics
    7. Token-level statistics

    Trade-off:
    - Lower accuracy on known structured types (URL, email, etc.)
    - Better generalization to unknown/unstructured data
    - More robust to variations in data format
    """

    def __init__(self, dim: int = 128, seed: int = 42):
        """
        Args:
            dim: Output vector dimension (must be divisible by 4)
            seed: Random seed for reproducibility
        """
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")

        self._dim = dim
        self.sub_dim = dim // 4

        np.random.seed(seed)

        # Random projection matrices for dimensionality reduction
        self.byte_projection = np.random.randn(256, self.sub_dim).astype(np.float32) / np.sqrt(self.sub_dim)
        self.unicode_projection = np.random.randn(48, self.sub_dim).astype(np.float32) / np.sqrt(self.sub_dim)
        self.bigram_projection = np.random.randn(64, self.sub_dim).astype(np.float32) / np.sqrt(self.sub_dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _extract_byte_features(self, text: str) -> np.ndarray:
        """
        Extract byte-level distribution features.

        Uses UTF-8 byte distribution with random projection.
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

        # Random projection to reduce dimensionality
        vec = self.byte_projection.T @ byte_hist
        return vec

    def _extract_unicode_features(self, text: str) -> np.ndarray:
        """
        Extract Unicode category and script distribution features.

        Uses statistical distribution of Unicode categories and scripts.
        """
        # Unicode categories (7 major categories)
        categories = {
            'L': 0,   # Letter
            'N': 1,   # Number
            'P': 2,   # Punctuation
            'S': 3,   # Symbol
            'Z': 4,   # Separator
            'M': 5,   # Mark
            'C': 6,   # Other
        }

        # Script detection based on Unicode ranges (statistical, not pattern-based)
        scripts = {
            'basic_latin': 7,      # 0x0000-0x007F
            'latin_ext': 8,        # 0x0080-0x024F
            'digit': 9,            # 0x0030-0x0039
            'hangul': 10,          # 0xAC00-0xD7AF
            'cjk': 11,             # 0x4E00-0x9FFF
            'hiragana': 12,        # 0x3040-0x309F
            'katakana': 13,        # 0x30A0-0x30FF
            'cyrillic': 14,        # 0x0400-0x04FF
            'arabic': 15,          # 0x0600-0x06FF
            'hebrew': 16,          # 0x0590-0x05FF
            'thai': 17,            # 0x0E00-0x0E7F
            'emoji': 18,           # 0x1F600-0x1F64F
        }

        features = np.zeros(48, dtype=np.float32)

        for char in text:
            # Unicode category
            cat = unicodedata.category(char)[0]
            if cat in categories:
                features[categories[cat]] += 1

            # Script detection (based on code point ranges)
            code = ord(char)

            if 0x0000 <= code <= 0x007F:
                features[scripts['basic_latin']] += 1
            elif 0x0080 <= code <= 0x024F:
                features[scripts['latin_ext']] += 1

            if 0x30 <= code <= 0x39:
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
            elif 0x0590 <= code <= 0x05FF:
                features[scripts['hebrew']] += 1
            elif 0x0E00 <= code <= 0x0E7F:
                features[scripts['thai']] += 1
            elif 0x1F600 <= code <= 0x1F64F:
                features[scripts['emoji']] += 1

            # Additional statistical features
            # Character subcategories (more granular)
            subcat = unicodedata.category(char)
            if subcat == 'Lu':  # Uppercase letter
                features[19] += 1
            elif subcat == 'Ll':  # Lowercase letter
                features[20] += 1
            elif subcat == 'Lt':  # Titlecase letter
                features[21] += 1
            elif subcat == 'Nd':  # Decimal number
                features[22] += 1
            elif subcat == 'Po':  # Other punctuation
                features[23] += 1
            elif subcat == 'Ps':  # Open punctuation
                features[24] += 1
            elif subcat == 'Pe':  # Close punctuation
                features[25] += 1

        # Normalize
        if len(text) > 0:
            features /= len(text)

        # Random projection
        vec = self.unicode_projection.T @ features
        return vec

    def _extract_ngram_features(self, text: str) -> np.ndarray:
        """
        Extract n-gram statistical features.

        Uses character n-gram diversity and transition probabilities.
        """
        features = np.zeros(64, dtype=np.float32)

        if len(text) == 0:
            return features

        # Feature 0-2: Basic text statistics
        features[0] = min(len(text) / 1000.0, 1.0)  # Length (normalized)
        features[1] = len(set(text)) / min(len(text), 256)  # Unique char ratio
        features[2] = len(set(text)) / 256.0  # Vocabulary size

        # Feature 3-5: Character class ratios
        features[3] = sum(c.isalpha() for c in text) / len(text)
        features[4] = sum(c.isdigit() for c in text) / len(text)
        features[5] = sum(c.isspace() for c in text) / len(text)

        # Feature 6-10: Punctuation and special characters
        features[6] = sum(c in '.,;:!?' for c in text) / len(text)
        features[7] = sum(c in '()[]{}' for c in text) / len(text)
        features[8] = sum(c in '<>/' for c in text) / len(text)
        features[9] = sum(c in '@#$%&*' for c in text) / len(text)
        features[10] = sum(c in '+-=|\\' for c in text) / len(text)

        # Feature 11-13: Case statistics
        features[11] = sum(c.isupper() for c in text) / len(text)
        features[12] = sum(c.islower() for c in text) / len(text)
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars > 0:
            features[13] = sum(c.isupper() for c in text) / alpha_chars

        # Feature 14-17: Character entropy
        char_counts = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1

        if char_counts:
            entropy = 0
            for count in char_counts.values():
                p = count / len(text)
                entropy -= p * np.log2(p + 1e-10)
            features[14] = entropy / 8.0  # Normalized

        # Feature 18-22: Bigram statistics
        if len(text) >= 2:
            bigrams = [text[i:i+2] for i in range(len(text)-1)]
            unique_bigrams = len(set(bigrams))
            features[18] = unique_bigrams / len(bigrams)  # Diversity
            features[19] = unique_bigrams / min(len(bigrams), 1024)  # Normalized count

            # Bigram entropy
            bigram_counts = {}
            for bg in bigrams:
                bigram_counts[bg] = bigram_counts.get(bg, 0) + 1

            bigram_entropy = 0
            for count in bigram_counts.values():
                p = count / len(bigrams)
                bigram_entropy -= p * np.log2(p + 1e-10)
            features[20] = bigram_entropy / 12.0

        # Feature 23-26: Trigram statistics
        if len(text) >= 3:
            trigrams = [text[i:i+3] for i in range(len(text)-2)]
            unique_trigrams = len(set(trigrams))
            features[23] = unique_trigrams / len(trigrams)  # Diversity
            features[24] = unique_trigrams / min(len(trigrams), 4096)  # Normalized count

        # Feature 27-30: Character transition probabilities
        if len(text) >= 2:
            transitions = {}
            for i in range(len(text)-1):
                pair = (text[i], text[i+1])
                transitions[pair] = transitions.get(pair, 0) + 1

            if transitions:
                total = sum(transitions.values())
                trans_entropy = 0
                for count in transitions.values():
                    p = count / total
                    trans_entropy -= p * np.log2(p + 1e-10)
                features[27] = trans_entropy / 12.0

        # Feature 31-36: Positional statistics
        if len(text) > 0:
            # First/last character statistics
            features[31] = 1.0 if text[0].isupper() else 0.0
            features[32] = 1.0 if text[0].isdigit() else 0.0
            features[33] = 1.0 if text[0].isspace() else 0.0
            features[34] = 1.0 if text[-1].isalpha() else 0.0
            features[35] = 1.0 if text[-1].isdigit() else 0.0
            features[36] = 1.0 if text[-1] in '.,;:!?' else 0.0

        # Feature 37-42: Token statistics
        tokens = text.split()
        features[37] = min(len(tokens) / 100.0, 1.0)

        if tokens:
            token_lengths = [len(t) for t in tokens]
            features[38] = np.mean(token_lengths) / 50.0
            features[39] = np.std(token_lengths) / 20.0 if len(token_lengths) > 1 else 0
            features[40] = min(token_lengths) / 50.0
            features[41] = max(token_lengths) / 100.0
            features[42] = len(set(tokens)) / len(tokens)  # Unique token ratio

        # Random projection
        vec = self.bigram_projection.T @ features
        return vec

    def _extract_statistical_moments(self, text: str) -> np.ndarray:
        """
        Extract higher-order statistical moments.

        Captures subtle statistical patterns in the text.
        """
        features = np.zeros(self.sub_dim, dtype=np.float32)

        if len(text) == 0:
            return features

        # Convert text to numerical sequence (character codes)
        char_codes = np.array([ord(c) for c in text], dtype=np.float32)

        # Feature 0-4: Moments of character code distribution
        if len(char_codes) > 0:
            features[0] = np.mean(char_codes) / 65536.0  # Mean
            features[1] = np.std(char_codes) / 65536.0   # Std
            if len(char_codes) > 1:
                features[2] = np.percentile(char_codes, 25) / 65536.0  # 25th percentile
                features[3] = np.percentile(char_codes, 50) / 65536.0  # Median
                features[4] = np.percentile(char_codes, 75) / 65536.0  # 75th percentile

        # Feature 5-9: Difference statistics (measures variability)
        if len(char_codes) > 1:
            diffs = np.diff(char_codes)
            features[5] = np.mean(np.abs(diffs)) / 65536.0
            features[6] = np.std(diffs) / 65536.0
            features[7] = np.max(np.abs(diffs)) / 65536.0

        # Feature 10-14: Run-length encoding statistics
        if len(text) > 0:
            runs = []
            current_char = text[0]
            current_run = 1

            for i in range(1, len(text)):
                if text[i] == current_char:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_char = text[i]
                    current_run = 1
            runs.append(current_run)

            if runs:
                features[10] = np.mean(runs) / 10.0
                features[11] = np.max(runs) / 20.0
                features[12] = len([r for r in runs if r > 1]) / len(runs)

        # Feature 13-16: Zipf's law statistics (frequency distribution)
        if len(text) > 0:
            char_counts = {}
            for c in text:
                char_counts[c] = char_counts.get(c, 0) + 1

            freq_counts = sorted(char_counts.values(), reverse=True)
            if len(freq_counts) > 0:
                features[13] = freq_counts[0] / len(text)  # Most frequent char ratio
                if len(freq_counts) > 1:
                    features[14] = freq_counts[1] / len(text)
                if len(freq_counts) > 2:
                    features[15] = freq_counts[2] / len(text)

        return features

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text using pure statistical features.

        No regex patterns, no hardcoded type detection.
        Only statistical properties of the text.

        Returns:
            128-dim normalized feature vector
        """
        # Extract features from 4 different statistical perspectives
        byte_vec = self._extract_byte_features(text)
        unicode_vec = self._extract_unicode_features(text)
        ngram_vec = self._extract_ngram_features(text)
        moment_vec = self._extract_statistical_moments(text)

        # Concatenate all features
        vec = np.concatenate([byte_vec, unicode_vec, ngram_vec, moment_vec])

        # Ensure correct dimension
        if len(vec) > self._dim:
            vec = vec[:self._dim]
        elif len(vec) < self._dim:
            padding = np.zeros(self._dim - len(vec), dtype=np.float32)
            vec = np.concatenate([vec, padding])

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec.astype(np.float32)
