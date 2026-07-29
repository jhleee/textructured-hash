"""Generalization-first text structure encoder.

The encoder keeps a small supervised Fisher branch for known-format accuracy,
but most similarity comes from label-free sketches that survive renderer and
family changes.  Every branch is normalized independently before weighted
concatenation so the supervised branch cannot dominate the cosine score.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..base import BaseEncoder
from .fisher_encoder import FisherStructureEncoder


_FISHER_DIM = 80
_LEXICAL_DIM = 96
_CHAR_GRAM_DIM = 80
_TOKEN_DIM = _LEXICAL_DIM - _CHAR_GRAM_DIM
_SHAPE_DIM = 48
_LAYOUT_DIM = 32
_OUTPUT_DIM = _FISHER_DIM + _LEXICAL_DIM + _SHAPE_DIM + _LAYOUT_DIM

# Default mixture keeps most cosine mass on label-free lexical and shape
# sketches. Visually tabular inputs route more mass to the layout expert.
_BRANCH_WEIGHTS = np.asarray([0.10, 0.38, 0.32, 0.20], dtype=np.float32)
_TABLE_BRANCH_WEIGHTS = np.asarray([0.03, 0.10, 0.07, 0.80], dtype=np.float32)
_DELIMITED_BRANCH_WEIGHTS = np.asarray([0.20, 0.40, 0.15, 0.25], dtype=np.float32)


@njit(cache=True)
def _canonical_byte(value):
    """Canonicalize volatile values while retaining reusable format syntax."""
    if value >= 65 and value <= 90:  # ASCII upper -> lower
        return value + 32
    if value >= 48 and value <= 57:  # all digits share one symbol
        return 48
    if value == 9 or value == 32:    # horizontal whitespace
        return 32
    if value == 13:                  # CRLF and LF are equivalent
        return 10
    if value >= 128:                 # Unicode payload is a structural class
        return 128
    return value


@njit(cache=True)
def _shape_class(value):
    """Map a UTF-8 byte to a small, domain-independent structural alphabet."""
    if (value >= 65 and value <= 90) or (value >= 97 and value <= 122):
        return 0  # letter
    if value >= 48 and value <= 57:
        return 1  # digit
    if value == 32 or value == 9:
        return 2  # horizontal whitespace
    if value == 10 or value == 13:
        return 3  # newline
    if value == 34 or value == 39 or value == 96:
        return 4  # quote
    if value == 40 or value == 41 or value == 91 or value == 93 or value == 123 or value == 125:
        return 5  # bracket
    if value == 44 or value == 58 or value == 59 or value == 124:
        return 6  # field/key separator
    if value == 46 or value == 47 or value == 92 or value == 45 or value == 95:
        return 7  # path/member separator
    if value == 61 or value == 43 or value == 42 or value == 37 or value == 38 or value == 63:
        return 8  # operator
    if value >= 128:
        return 9  # Unicode byte
    return 10     # other punctuation/control


@njit(cache=True)
def _mix_hash(current, value):
    """Small deterministic avalanche mixer; unlike base-256 modulo, all bytes matter."""
    x = np.bitwise_xor(np.uint64(current), np.uint64(int(value) + 0x9E3779B9))
    # NumPy's ufunc has identical wrapping uint64 semantics with and without JIT.
    x = np.multiply(x, np.uint64(0xBF58476D1CE4E5B9))
    x = np.bitwise_xor(x, np.right_shift(x, np.uint64(27)))
    return np.uint64(x)


@njit(cache=True)
def _add_signed(vector, hash_value, amount):
    index = int(hash_value % np.uint64(len(vector)))
    sign = 1.0 if ((hash_value >> np.uint64(17)) & np.uint64(1)) == 0 else -1.0
    vector[index] += sign * amount


@njit(cache=True)
def _add_signed_slice(vector, hash_value, offset, width, amount):
    index = offset + int(hash_value % np.uint64(width))
    sign = 1.0 if ((hash_value >> np.uint64(17)) & np.uint64(1)) == 0 else -1.0
    vector[index] += sign * amount


@njit(cache=True)
def _add_unsigned_slice(vector, hash_value, offset, width, amount):
    index = offset + int(hash_value % np.uint64(width))
    vector[index] += amount


@njit(cache=True)
def _normalize_slice(vector, start, end, weight):
    norm_sq = 0.0
    for index in range(start, end):
        norm_sq += vector[index] * vector[index]
    if norm_sq > 1e-12:
        scale = np.sqrt(weight / norm_sq)
        for index in range(start, end):
            vector[index] *= scale


@njit(cache=True)
def _extract_invariant_features(byte_arr):
    """Extract lexical, run-shape, and layout branches in one bounded scan."""
    lexical = np.zeros(_LEXICAL_DIM, dtype=np.float32)
    shape = np.zeros(_SHAPE_DIM, dtype=np.float32)
    layout = np.zeros(_LAYOUT_DIM, dtype=np.float32)
    n_bytes = len(byte_arr)
    if n_bytes == 0:
        layout[0] = 1.0
        return lexical, shape, layout, 0.0, 0.0

    # Canonical character 2/3/4-grams. Digits, case, horizontal whitespace,
    # CRLF, and Unicode payload bytes are normalized before hashing.
    canonical = np.empty(n_bytes, dtype=np.int32)
    canonical_count = 0
    previous_newline = False
    for index in range(n_bytes):
        value = _canonical_byte(int(byte_arr[index]))
        if value == 10 and previous_newline and byte_arr[index] == 10:
            # A CRLF was canonicalized to two LF bytes; retain one.
            previous_newline = False
            continue
        canonical[canonical_count] = value
        canonical_count += 1
        previous_newline = value == 10 and byte_arr[index] == 13

    gram_total = 0
    for width in range(2, 5):
        if canonical_count < width:
            continue
        seed = np.uint64(0xCBF29CE484222325 + width * 131)
        for start in range(canonical_count - width + 1):
            value_hash = seed
            for offset in range(width):
                value_hash = _mix_hash(value_hash, canonical[start + offset] + offset * 257)
            _add_signed_slice(lexical, value_hash, 0, _CHAR_GRAM_DIM, 1.0)
            gram_total += 1
    if gram_total > 0:
        scale = 1.0 / np.sqrt(float(gram_total))
        for index in range(_CHAR_GRAM_DIM):
            value = lexical[index] * scale
            lexical[index] = np.sign(value) * np.sqrt(abs(value))

    # Exact lowercase word tokens form sparse anchors across radically
    # different renderers (for example field names or protocol vocabulary).
    # Numeric-only values and long opaque identifiers are deliberately omitted.
    token_hash = np.uint64(0x9E3779B185EBCA87)
    token_length = 0
    token_count = 0
    for index in range(canonical_count + 1):
        value = 32 if index == canonical_count else canonical[index]
        is_letter = value >= 97 and value <= 122
        if is_letter:
            token_hash = _mix_hash(token_hash, value)
            token_length += 1
        else:
            if token_length >= 2 and token_length <= 24:
                _add_unsigned_slice(lexical, token_hash, _CHAR_GRAM_DIM, _TOKEN_DIM, 1.0)
                token_count += 1
            token_hash = np.uint64(0x9E3779B185EBCA87)
            token_length = 0
    if token_count > 0:
        for index in range(_CHAR_GRAM_DIM, _LEXICAL_DIM):
            lexical[index] = np.sign(lexical[index]) * np.sqrt(abs(lexical[index]))

    # Keep common literal tokens from being drowned by punctuation grams.
    _normalize_slice(lexical, 0, _CHAR_GRAM_DIM, 0.80)
    _normalize_slice(lexical, _CHAR_GRAM_DIM, _LEXICAL_DIM, 0.20)

    # Run-shape uni/bi/tri-grams preserve token and delimiter order while
    # discarding concrete identifiers and values.
    run_tokens = np.empty(n_bytes, dtype=np.int32)
    run_count = 0
    current_class = _shape_class(int(byte_arr[0]))
    current_length = 1
    for index in range(1, n_bytes + 1):
        next_class = -1 if index == n_bytes else _shape_class(int(byte_arr[index]))
        if next_class == current_class:
            current_length += 1
            continue
        if current_length <= 1:
            length_bucket = 0
        elif current_length <= 2:
            length_bucket = 1
        elif current_length <= 4:
            length_bucket = 2
        elif current_length <= 8:
            length_bucket = 3
        else:
            length_bucket = 4
        run_tokens[run_count] = current_class * 5 + length_bucket
        run_count += 1
        current_class = next_class
        current_length = 1

    run_gram_total = 0
    for width in range(1, 4):
        if run_count < width:
            continue
        seed = np.uint64(0x84222325CBF29CE4 + width * 193)
        for start in range(run_count - width + 1):
            value_hash = seed
            for offset in range(width):
                value_hash = _mix_hash(value_hash, run_tokens[start + offset] + offset * 97)
            _add_signed(shape, value_hash, 1.0)
            run_gram_total += 1
    if run_gram_total > 0:
        scale = 1.0 / np.sqrt(float(run_gram_total))
        for index in range(_SHAPE_DIM):
            value = shape[index] * scale
            shape[index] = np.sign(value) * np.sqrt(abs(value))

    # Generic line/layout profile. Features are grouped one-hot buckets so
    # unlike separator types (CSV/TSV/CLI columns) can still share geometry.
    max_lines = n_bytes + 1
    line_lengths = np.empty(max_lines, dtype=np.float64)
    line_indents = np.empty(max_lines, dtype=np.float64)
    line_fields = np.empty(max_lines, dtype=np.float64)
    line_count = 0
    line_length = 0
    line_indent = 0
    leading = True
    horizontal_run = 0
    line_field_separators = 0
    non_ascii = 0
    brackets = 0
    quotes = 0
    total_field_separators = 0
    total_visual_separators = 0
    total_machine_separators = 0
    total_horizontal_space = 0

    for index in range(n_bytes + 1):
        value = 10 if index == n_bytes else int(byte_arr[index])
        if value >= 128:
            non_ascii += 1
        if value == 40 or value == 41 or value == 91 or value == 93 or value == 123 or value == 125:
            brackets += 1
        if value == 34 or value == 39 or value == 96:
            quotes += 1

        is_horizontal_space = value == 32 or value == 9
        if is_horizontal_space:
            total_horizontal_space += 1
            if leading:
                line_indent += 1
            elif value == 32:
                horizontal_run += 1
        elif value != 13 and value != 10:
            if horizontal_run >= 2:
                line_field_separators += 1
                total_field_separators += 1
                total_visual_separators += 1
            horizontal_run = 0
            leading = False

        if value == 44 or value == 9 or value == 124 or value == 59:
            line_field_separators += 1
            total_field_separators += 1
            if value == 44 or value == 9 or value == 59:
                total_machine_separators += 1

        # Decode box-drawing code points as generic visual separators. This is
        # Unicode normalization, not a detector for a particular table style.
        if value == 0xE2 and index + 2 < n_bytes:
            second = int(byte_arr[index + 1])
            third = int(byte_arr[index + 2])
            codepoint = ((value & 15) << 12) | ((second & 63) << 6) | (third & 63)
            if codepoint >= 0x2500 and codepoint <= 0x257F:
                is_horizontal_rule = (
                    codepoint == 0x2500 or codepoint == 0x2501 or
                    codepoint == 0x2504 or codepoint == 0x2505 or
                    codepoint == 0x2508 or codepoint == 0x2509 or
                    codepoint == 0x254C or codepoint == 0x254D
                )
                if not is_horizontal_rule:
                    line_field_separators += 1
                    total_field_separators += 1
                    total_visual_separators += 1

        if value == 10:
            line_lengths[line_count] = line_length
            line_indents[line_count] = line_indent
            line_fields[line_count] = line_field_separators
            line_count += 1
            line_length = 0
            line_indent = 0
            horizontal_run = 0
            line_field_separators = 0
            leading = True
        elif value != 13 and not (value >= 128 and value < 192):
            # Count Unicode code points rather than UTF-8 continuation bytes.
            line_length += 1

    length_sum = 0.0
    indent_sum = 0.0
    fields_sum = 0.0
    nonempty_lines = 0.0
    for index in range(line_count):
        length_sum += line_lengths[index]
        indent_sum += line_indents[index]
        fields_sum += line_fields[index]
        if line_lengths[index] > 0:
            nonempty_lines += 1.0
    mean_length = length_sum / max(float(line_count), 1.0)
    mean_indent = indent_sum / max(float(line_count), 1.0)
    mean_fields = fields_sum / max(nonempty_lines, 1.0)

    length_var = 0.0
    fields_var = 0.0
    repeated_shape = 0.0
    for index in range(line_count):
        diff = line_lengths[index] - mean_length
        length_var += diff * diff
        field_diff = line_fields[index] - mean_fields
        fields_var += field_diff * field_diff
        if index > 0:
            if abs(line_lengths[index] - line_lengths[index - 1]) <= 4 and abs(line_fields[index] - line_fields[index - 1]) <= 1:
                repeated_shape += 1.0
    length_cv = np.sqrt(length_var / max(float(line_count), 1.0)) / max(mean_length, 1.0)
    fields_cv = np.sqrt(fields_var / max(float(line_count), 1.0)) / max(mean_fields, 1.0)

    # Eight generic quantities, four soft ordinal buckets each.
    quantities = np.empty(8, dtype=np.float64)
    quantities[0] = float(line_count)
    quantities[1] = mean_length
    quantities[2] = length_cv
    quantities[3] = mean_indent
    quantities[4] = mean_fields
    quantities[5] = fields_cv
    quantities[6] = repeated_shape / max(float(line_count - 1), 1.0)
    quantities[7] = non_ascii / max(float(n_bytes), 1.0)

    boundaries = np.empty((8, 3), dtype=np.float64)
    boundaries[0, 0], boundaries[0, 1], boundaries[0, 2] = 1.0, 3.0, 8.0
    boundaries[1, 0], boundaries[1, 1], boundaries[1, 2] = 16.0, 48.0, 96.0
    boundaries[2, 0], boundaries[2, 1], boundaries[2, 2] = 0.08, 0.25, 0.60
    boundaries[3, 0], boundaries[3, 1], boundaries[3, 2] = 0.0, 2.0, 6.0
    boundaries[4, 0], boundaries[4, 1], boundaries[4, 2] = 0.0, 2.0, 5.0
    boundaries[5, 0], boundaries[5, 1], boundaries[5, 2] = 0.10, 0.35, 0.80
    boundaries[6, 0], boundaries[6, 1], boundaries[6, 2] = 0.10, 0.50, 0.90
    boundaries[7, 0], boundaries[7, 1], boundaries[7, 2] = 0.0, 0.05, 0.30
    for quantity_index in range(8):
        bucket_index = 3
        for boundary_index in range(3):
            if quantities[quantity_index] <= boundaries[quantity_index, boundary_index]:
                bucket_index = boundary_index
                break
        layout[quantity_index * 4 + bucket_index] = 1.0

    # Add bounded density information without introducing family detectors.
    layout[5] += min(total_horizontal_space / max(float(n_bytes), 1.0), 0.5)
    layout[9] += min(total_field_separators / max(float(n_bytes), 1.0), 0.25) * 2.0
    layout[13] += min((brackets + quotes) / max(float(n_bytes), 1.0), 0.25) * 2.0
    visual_separators_per_line = total_visual_separators / max(nonempty_lines, 1.0)
    machine_separators_per_line = total_machine_separators / max(nonempty_lines, 1.0)
    visual_table_strength = 1.0 if line_count >= 3 and visual_separators_per_line >= 1.5 else 0.0
    delimited_table_strength = 1.0 if line_count >= 3 and machine_separators_per_line >= 1.5 else 0.0
    return lexical, shape, layout, visual_table_strength, delimited_table_strength


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


class GeneralizedStructureEncoder(BaseEncoder):
    """Hybrid encoder whose similarity is mostly label-free and invariant."""

    def __init__(self, dim: int = _OUTPUT_DIM, seed: int = 42):
        if dim != _OUTPUT_DIM:
            raise ValueError(f"GeneralizedStructureEncoder requires dim={_OUTPUT_DIM}")
        self._dim = dim
        self.seed = seed
        self.fisher = FisherStructureEncoder(dim=_FISHER_DIM, seed=seed)
        # Warm the new numba kernel as part of initialization.
        _extract_invariant_features(np.zeros(1, dtype=np.uint8))

    @property
    def dim(self) -> int:
        return self._dim

    def train(self, train_pairs: list) -> None:
        self.fisher.train(train_pairs, n_components=_FISHER_DIM)

    def encode(self, text: str) -> np.ndarray:
        byte_arr = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
        lexical, shape, layout, table_strength, delimited_strength = _extract_invariant_features(byte_arr)

        # Content-adaptive routing uses generic column evidence only. Aligned
        # spaces/box drawing and repeated machine delimiters get separate
        # mixtures; ordinary text stays on the balanced default mixture.
        if table_strength > 0.5:
            weights = _TABLE_BRANCH_WEIGHTS
        elif delimited_strength > 0.5:
            weights = _DELIMITED_BRANCH_WEIGHTS
        else:
            weights = _BRANCH_WEIGHTS

        branches = (
            _unit(self.fisher.encode(text)),
            _unit(lexical),
            _unit(shape),
            _unit(layout),
        )
        weighted = [branch * np.sqrt(weight) for branch, weight in zip(branches, weights)]
        return _unit(np.concatenate(weighted))

    def encode_int8(self, text: str) -> np.ndarray:
        return np.round(self.encode(text) * 127.0).astype(np.int8)


    def save(self, path: str) -> None:
        """Persist the only learned state: the bounded Fisher branch."""
        self.fisher.save(path)

    @classmethod
    def load(cls, path: str, seed: int = 42) -> "GeneralizedStructureEncoder":
        """Restore a trained encoder from :meth:`save` output."""
        encoder = cls(dim=_OUTPUT_DIM, seed=seed)
        encoder.fisher = FisherStructureEncoder.load(path, seed=seed)
        if encoder.fisher.dim != _FISHER_DIM:
            raise ValueError(
                f"Expected Fisher branch dim {_FISHER_DIM}, got {encoder.fisher.dim}"
            )
        return encoder