"""Int8 Quantized Structure Type Encoder (Experiment 2.2)

Reduces memory footprint by 75% using int8 quantization.
"""

import numpy as np
from .structure_type import StructureTypeEncoder


class QuantizedEncoder(StructureTypeEncoder):
    """
    Experiment 2.2: Int8 Quantization

    Hypothesis: Float32 → Int8 quantization reduces vector size by 75%
    with minimal quality degradation.

    Expected improvement:
    - Vector size: 512 bytes → 128 bytes (75% reduction)
    - Quality loss: AUC-ROC -0.01 ~ -0.02 (acceptable)

    Note: For evaluation compatibility, encode() returns float32.
    Use encode_int8() to get actual int8 vectors for storage.
    """

    def __init__(self, dim: int = 128, type_dim: int = 16, seed: int = 42):
        """
        Args:
            dim: Output vector dimension
            type_dim: Type encoding dimension
            seed: Random seed
        """
        super().__init__(dim=dim, type_dim=type_dim, seed=seed)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text with int8 quantization.

        Returns float32 for compatibility, but values are quantized
        to simulate int8 precision loss.
        """
        # Get float32 vector from parent (L2 normalized)
        vec_f32 = super().encode(text)

        # Scale to [-1, 1] range (should already be there from normalization)
        # Then quantize to 256 levels
        vec_quantized = np.round(vec_f32 * 127.0) / 127.0

        # Clip to valid range
        vec_quantized = np.clip(vec_quantized, -1.0, 1.0)

        return vec_quantized.astype(np.float32)

    def encode_int8(self, text: str) -> np.ndarray:
        """
        Encode to true int8 vector for minimal storage (128 bytes for dim=128).

        This is the actual quantized representation.
        """
        vec_f32 = super().encode(text)

        # Scale normalized vector to int8 range [-127, 127]
        vec_i8 = np.round(vec_f32 * 127.0).astype(np.int8)

        return vec_i8


class QuantizedStructureTypeCompactEncoder(StructureTypeEncoder):
    """
    Experiment 2.2 Enhanced: Int8 Quantization with 256 dimensions

    Achieves exactly 256 bytes with dim=256 and int8 quantization.
    """

    def __init__(self, dim: int = 256, type_dim: int = 32, seed: int = 42):
        """
        Args:
            dim: Output dimension (256 for 256-byte target with int8)
            type_dim: Type encoding dimension
            seed: Random seed
        """
        super().__init__(dim=dim, type_dim=type_dim, seed=seed)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode with int8-precision quantization (returned as float32 for compatibility).

        Actual int8 storage: 256 bytes (exactly at target!)
        """
        vec_f32 = super().encode(text)

        # Quantize to int8 precision
        vec_quantized = np.round(vec_f32 * 127.0) / 127.0
        vec_quantized = np.clip(vec_quantized, -1.0, 1.0)

        return vec_quantized.astype(np.float32)

    def encode_int8(self, text: str) -> np.ndarray:
        """Encode to true int8 for 256-byte storage."""
        vec_f32 = super().encode(text)
        vec_i8 = np.round(vec_f32 * 127.0).astype(np.int8)
        return vec_i8
