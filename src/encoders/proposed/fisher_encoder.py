"""Fisher Discriminant Structure Encoder

Combines fast numba-compiled feature extraction with Fisher Linear Discriminant
Analysis to learn an optimal projection that maximizes between-class separation
while minimizing within-class variance.

Uses numba JIT for feature extraction to achieve >10,000 texts/sec, while the
learned projection matrix ensures different structural types map to well-separated
regions of the embedding space.
"""

import numpy as np
from numba import njit
from scipy import linalg
from ..base import BaseEncoder


# Feature dimension constant
_FEATURE_DIM = 320

# Precomputed lookup tables as module-level arrays for numba
_IS_ALPHA = np.zeros(128, dtype=np.int8)
_IS_DIGIT = np.zeros(128, dtype=np.int8)
_IS_UPPER = np.zeros(128, dtype=np.int8)
_IS_LOWER = np.zeros(128, dtype=np.int8)
_IS_SPACE = np.zeros(128, dtype=np.int8)
_IS_PUNCT = np.zeros(128, dtype=np.int8)
_IS_BRACKET = np.zeros(128, dtype=np.int8)
_IS_MATH = np.zeros(128, dtype=np.int8)
_IS_SPECIAL = np.zeros(128, dtype=np.int8)

for _i in range(128):
    _c = chr(_i)
    if _c.isalpha(): _IS_ALPHA[_i] = 1
    if _c.isdigit(): _IS_DIGIT[_i] = 1
    if _c.isupper(): _IS_UPPER[_i] = 1
    if _c.islower(): _IS_LOWER[_i] = 1
    if _c.isspace(): _IS_SPACE[_i] = 1
    if _c in '.,;:!?': _IS_PUNCT[_i] = 1
    if _c in '()[]{}': _IS_BRACKET[_i] = 1
    if _c in '+-=*/<>|&^~%': _IS_MATH[_i] = 1
    if _c in '@#$_': _IS_SPECIAL[_i] = 1

# Structural character byte values
_STRUCT_CHARS = np.array([
    46, 47, 58, 64, 35, 61, 63, 38, 59, 44,
    123, 125, 91, 93, 60, 62, 40, 41, 34, 39,
    45, 95, 92, 124, 126, 43, 42, 37, 33, 10, 9, 32
], dtype=np.int32)


@njit
def _extract_features_numba(byte_arr, features, struct_chars,
                            is_alpha, is_digit, is_upper, is_lower,
                            is_space, is_punct, is_bracket, is_math, is_special):
    """Numba-compiled feature extraction from byte array."""
    n_bytes = len(byte_arr)
    if n_bytes == 0:
        return features
    
    inv_n = 1.0 / n_bytes
    
    # === Byte histogram (256 bins) ===
    hist = np.zeros(256, dtype=np.float64)
    for i in range(n_bytes):
        hist[byte_arr[i]] += 1.0
    
    # [0:64] Grouped byte histogram (4 bytes per bin)
    for i in range(64):
        features[i] = (hist[i*4] + hist[i*4+1] + hist[i*4+2] + hist[i*4+3]) * inv_n
    
    # === [64:76] Character class ratios ===
    n_alpha = 0.0
    n_digit = 0.0
    n_upper = 0.0
    n_lower = 0.0
    n_space = 0.0
    n_punct = 0.0
    n_bracket = 0.0
    n_math = 0.0
    n_special = 0.0
    n_ascii = 0
    
    for i in range(n_bytes):
        b = byte_arr[i]
        if b < 128:
            n_ascii += 1
            n_alpha += is_alpha[b]
            n_digit += is_digit[b]
            n_upper += is_upper[b]
            n_lower += is_lower[b]
            n_space += is_space[b]
            n_punct += is_punct[b]
            n_bracket += is_bracket[b]
            n_math += is_math[b]
            n_special += is_special[b]
    
    if n_ascii > 0:
        features[64] = n_alpha * inv_n
        features[65] = n_digit * inv_n
        features[66] = n_upper * inv_n
        features[67] = n_lower * inv_n
        features[68] = n_space * inv_n
        features[69] = n_punct * inv_n
        features[70] = n_bracket * inv_n
        features[71] = n_math * inv_n
        features[72] = n_special * inv_n
        features[73] = n_ascii * inv_n  # ASCII ratio
        features[74] = 1.0 - n_ascii * inv_n  # Non-ASCII ratio
        # Digit/alpha ratio
        if n_alpha + n_digit > 0:
            features[75] = n_digit / (n_alpha + n_digit)
    
    # === [76:140] Bigram hash features (64 bins) ===
    if n_bytes >= 2:
        inv_bi = 1.0 / (n_bytes - 1)
        for i in range(n_bytes - 1):
            bigram_val = byte_arr[i] * 256 + byte_arr[i+1]
            features[76 + bigram_val % 64] += inv_bi
    
    # === [140:172] Structural character frequencies ===
    for i in range(32):
        features[140 + i] = hist[struct_chars[i]] * inv_n
    
    # === [172:220] Statistical features ===
    # Byte mean
    byte_sum = 0.0
    for i in range(n_bytes):
        byte_sum += byte_arr[i]
    byte_mean = byte_sum / n_bytes
    features[172] = byte_mean / 255.0
    
    # Byte std
    var_sum = 0.0
    for i in range(n_bytes):
        diff = byte_arr[i] - byte_mean
        var_sum += diff * diff
    byte_std = (var_sum / n_bytes) ** 0.5
    features[173] = byte_std / 128.0
    
    # Length features
    features[174] = min(n_bytes / 100.0, 1.0)
    features[175] = min(n_bytes / 1000.0, 1.0)
    features[176] = min(np.log(n_bytes + 1) / 10.0, 1.0)
    
    # Unique byte count
    n_unique = 0
    for i in range(256):
        if hist[i] > 0:
            n_unique += 1
    features[177] = n_unique / 256.0
    features[178] = n_unique / min(n_bytes, 256)
    
    # Entropy
    entropy = 0.0
    for i in range(256):
        if hist[i] > 0:
            p = hist[i] * inv_n
            entropy -= p * np.log2(p)
    features[179] = entropy / 8.0
    
    # Change rate
    if n_bytes >= 2:
        changes = 0
        for i in range(n_bytes - 1):
            if byte_arr[i] != byte_arr[i+1]:
                changes += 1
        features[180] = changes / (n_bytes - 1.0)
    
    # Space/newline counts
    n_spaces = int(hist[32])
    n_newlines = int(hist[10])
    n_tabs = int(hist[9])
    features[181] = min(n_spaces / 50.0, 1.0)
    features[182] = min((n_spaces + 1.0) / n_bytes, 1.0)
    features[183] = min(n_newlines / 20.0, 1.0)
    features[184] = min(n_tabs / 10.0, 1.0)
    
    # Bracket balance
    n_open = hist[40] + hist[91] + hist[123]  # ( [ {
    n_close = hist[41] + hist[93] + hist[125]  # ) ] }
    features[185] = min(n_open * inv_n, 0.2) * 5.0
    features[186] = min(n_close * inv_n, 0.2) * 5.0
    if n_open == n_close and n_open > 0:
        features[187] = 1.0
    
    # Leading byte indicators
    first = byte_arr[0]
    if first < 128:
        features[188] = float(is_alpha[first])
        features[189] = float(is_digit[first])
    features[190] = 1.0 if first == 47 else 0.0   # /
    features[191] = 1.0 if first == 60 else 0.0   # <
    features[192] = 1.0 if first == 123 else 0.0  # {
    features[193] = 1.0 if first == 91 else 0.0   # [
    features[194] = 1.0 if first == 104 else 0.0  # h
    features[195] = 1.0 if first >= 128 else 0.0  # non-ASCII
    
    # Pattern indicators
    if n_bytes >= 4:
        if byte_arr[0] == 104 and byte_arr[1] == 116 and byte_arr[2] == 116 and byte_arr[3] == 112:
            features[196] = 1.0  # http
    if hist[64] > 0 and hist[46] > 0:
        features[197] = 1.0  # @ and . (email-like)
    if first == 123 or first == 91:
        features[198] = 1.0  # JSON-like
    if first == 60:
        features[199] = 1.0  # XML-like
    if first == 47 or (hist[58] > 0 and hist[92] > 0):
        features[200] = 1.0  # filepath-like
    
    # Code punctuation density
    code_punct = hist[59] + hist[123] + hist[125] + hist[40] + hist[41] + hist[61]
    features[201] = min(code_punct * inv_n, 0.3) / 0.3
    
    # First 4 bytes normalized
    n_pos = min(4, n_bytes)
    for i in range(n_pos):
        features[202 + i] = byte_arr[i] / 255.0
    # Last 4 bytes
    for i in range(n_pos):
        features[206 + i] = byte_arr[n_bytes - n_pos + i] / 255.0
    
    # Quartile means
    q_size = max(n_bytes // 4, 1)
    for q in range(4):
        start = q * q_size
        end = min((q + 1) * q_size, n_bytes) if q < 3 else n_bytes
        if start < n_bytes:
            qsum = 0.0
            qcount = 0
            for i in range(start, min(end, n_bytes)):
                qsum += byte_arr[i]
                qcount += 1
            if qcount > 0:
                features[210 + q] = qsum / (qcount * 255.0)
    
    # Separator regularity
    sep_bytes = [44, 9, 124, 59]  # comma, tab, pipe, semicolon
    for s_idx in range(4):
        sep_byte = sep_bytes[s_idx]
        count = int(hist[sep_byte])
        if count >= 2:
            # Find positions and compute gap regularity
            positions = np.empty(count, dtype=np.int64)
            pos_idx = 0
            for i in range(n_bytes):
                if byte_arr[i] == sep_byte:
                    positions[pos_idx] = i
                    pos_idx += 1
            if pos_idx >= 2:
                gap_sum = 0.0
                gap_sq_sum = 0.0
                n_gaps = pos_idx - 1
                for i in range(n_gaps):
                    gap = positions[i+1] - positions[i]
                    gap_sum += gap
                    gap_sq_sum += gap * gap
                gap_mean = gap_sum / n_gaps
                gap_var = gap_sq_sum / n_gaps - gap_mean * gap_mean
                gap_std = gap_var ** 0.5 if gap_var > 0 else 0.0
                cv = gap_std / (gap_mean + 1e-10)
                features[214 + s_idx] = 1.0 / (1.0 + cv)
    
    # === [220:284] Trigram hash features (64 bins) ===
    if n_bytes >= 3:
        inv_tri = 1.0 / (n_bytes - 2)
        for i in range(n_bytes - 2):
            tri_val = (byte_arr[i] * 65536 + byte_arr[i+1] * 256 + byte_arr[i+2]) % 64
            features[220 + tri_val] += inv_tri
    
    # === [284:316] Quadgram hash features (32 bins) ===
    if n_bytes >= 4:
        inv_quad = 1.0 / (n_bytes - 3)
        for i in range(n_bytes - 3):
            quad_val = (byte_arr[i] * 16777216 + byte_arr[i+1] * 65536 + 
                       byte_arr[i+2] * 256 + byte_arr[i+3]) % 32
            features[284 + quad_val] += inv_quad
    
    # === [316:320] Additional features ===
    # Consecutive digit run max length
    max_digit_run = 0
    cur_digit_run = 0
    for i in range(n_bytes):
        b = byte_arr[i]
        if b >= 48 and b <= 57:  # '0'-'9'
            cur_digit_run += 1
            if cur_digit_run > max_digit_run:
                max_digit_run = cur_digit_run
        else:
            cur_digit_run = 0
    features[316] = min(max_digit_run / 20.0, 1.0)
    
    # Max alpha run
    max_alpha_run = 0
    cur_alpha_run = 0
    for i in range(n_bytes):
        b = byte_arr[i]
        if b < 128 and is_alpha[b]:
            cur_alpha_run += 1
            if cur_alpha_run > max_alpha_run:
                max_alpha_run = cur_alpha_run
        else:
            cur_alpha_run = 0
    features[317] = min(max_alpha_run / 30.0, 1.0)
    
    # Ratio of bytes in range 0x80-0xBF (UTF-8 continuation bytes)
    n_continuation = 0
    for i in range(n_bytes):
        if byte_arr[i] >= 128 and byte_arr[i] < 192:
            n_continuation += 1
    features[318] = n_continuation * inv_n
    
    # Has equal sign (key=value patterns)
    features[319] = 1.0 if hist[61] > 0 else 0.0
    
    return features


class FisherStructureEncoder(BaseEncoder):
    """
    Fisher Discriminant-based text structure encoder.
    
    Training: learns projection W via Fisher LDA on category-grouped features.
    Inference: numba-compiled feature extraction + matrix multiply + L2 norm.
    """

    def __init__(self, dim: int = 256, feature_dim: int = None, seed: int = 42):
        self._dim = dim
        self.feature_dim = feature_dim if feature_dim is not None else _FEATURE_DIM
        self.seed = seed
        
        np.random.seed(seed)
        self.W = np.random.randn(self.feature_dim, dim).astype(np.float32) * 0.01
        self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.feature_std = np.ones(self.feature_dim, dtype=np.float32)
        self.trained = False
        
        # Trigger numba compilation on init
        dummy = np.zeros(1, dtype=np.uint8)
        dummy_feat = np.zeros(self.feature_dim, dtype=np.float64)
        _extract_features_numba(dummy, dummy_feat, _STRUCT_CHARS,
                               _IS_ALPHA, _IS_DIGIT, _IS_UPPER, _IS_LOWER,
                               _IS_SPACE, _IS_PUNCT, _IS_BRACKET, _IS_MATH, _IS_SPECIAL)

    @property
    def dim(self) -> int:
        return self._dim

    def _extract_features(self, text: str) -> np.ndarray:
        """Extract features using numba-compiled function."""
        text_bytes = text.encode('utf-8', errors='ignore')
        byte_arr = np.frombuffer(text_bytes, dtype=np.uint8)
        
        if len(byte_arr) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        features = np.zeros(self.feature_dim, dtype=np.float64)
        _extract_features_numba(byte_arr, features, _STRUCT_CHARS,
                               _IS_ALPHA, _IS_DIGIT, _IS_UPPER, _IS_LOWER,
                               _IS_SPACE, _IS_PUNCT, _IS_BRACKET, _IS_MATH, _IS_SPECIAL)
        return features.astype(np.float32)

    def train(self, train_pairs: list, n_components: int = None):
        """
        Train the Fisher projection matrix from labeled training pairs.
        """
        if n_components is None:
            n_components = self._dim
            
        print("Fisher Encoder Training...")
        print("  Step 1: Extracting features from training data...")
        
        # Collect texts by category
        category_texts = {}
        for pair in train_pairs:
            cat1 = pair['category1']
            cat2 = pair['category2']
            if cat1 not in category_texts:
                category_texts[cat1] = []
            if cat2 not in category_texts:
                category_texts[cat2] = []
            category_texts[cat1].append(pair['text1'])
            category_texts[cat2].append(pair['text2'])
        
        # Deduplicate in a stable order so training does not depend on
        # Python's per-process hash randomization.
        for cat in category_texts:
            category_texts[cat] = sorted(set(category_texts[cat]))

        print(f"  Found {len(category_texts)} categories")

        # Extract features for all texts
        all_features = []
        all_labels = []
        cat_to_idx = {cat: i for i, cat in enumerate(sorted(category_texts.keys()))}

        for cat in sorted(category_texts):
            texts = category_texts[cat]
            cat_idx = cat_to_idx[cat]
            for text in texts:
                feat = self._extract_features(text)
                all_features.append(feat)
                all_labels.append(cat_idx)
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels)
        n_samples, n_features = X.shape
        n_classes = len(category_texts)
        
        print(f"  Extracted features: {n_samples} samples x {n_features} features")
        print(f"  Classes: {n_classes}")
        
        # Step 2: Feature normalization
        print("  Step 2: Normalizing features...")
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        X = (X - self.feature_mean) / self.feature_std
        
        # Step 3: Compute scatter matrices
        print("  Step 3: Computing scatter matrices...")
        overall_mean = X.mean(axis=0)
        
        Sw = np.zeros((n_features, n_features), dtype=np.float64)
        Sb = np.zeros((n_features, n_features), dtype=np.float64)
        
        for c in range(n_classes):
            Xc = X[y == c]
            nc = len(Xc)
            if nc == 0:
                continue
            mean_c = Xc.mean(axis=0)
            
            diff = Xc - mean_c
            Sw += diff.T @ diff
            
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            Sb += nc * (mean_diff @ mean_diff.T)
        
        # Step 4: Solve generalized eigenvalue problem
        print("  Step 4: Solving Fisher LDA eigenvalue problem...")
        
        reg = 1e-4 * np.trace(Sw) / n_features
        Sw += reg * np.eye(n_features)
        
        try:
            eigenvalues, eigenvectors = linalg.eigh(Sb, Sw)
            
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            W = eigenvectors[:, :n_components].astype(np.float32)
            
            print(f"  Top eigenvalues: {eigenvalues[:5]}")
            print(f"  Projection matrix shape: {W.shape}")
            
        except Exception as e:
            print(f"  Warning: eigenvalue decomposition failed ({e}), using PCA fallback")
            U, s, Vt = linalg.svd(X, full_matrices=False)
            W = Vt[:n_components].T.astype(np.float32)
        
        # Step 5: Fine-tune with contrastive objective
        print("  Step 5: Contrastive fine-tuning...")
        W = self._contrastive_finetune(W, train_pairs, n_epochs=5)
        
        self.W = W
        self.trained = True
        print("  Training complete!")
        
    def _contrastive_finetune(self, W, train_pairs, n_epochs=5):
        """Fine-tune W using contrastive loss on training pairs."""
        import random
        random.seed(self.seed)
        
        sampled_pairs = random.sample(train_pairs, min(3000, len(train_pairs)))
        
        feats1 = []
        feats2 = []
        labels = []
        for pair in sampled_pairs:
            f1 = self._extract_features(pair['text1'])
            f2 = self._extract_features(pair['text2'])
            f1 = (f1 - self.feature_mean) / self.feature_std
            f2 = (f2 - self.feature_mean) / self.feature_std
            feats1.append(f1)
            feats2.append(f2)
            labels.append(pair['label'])
        
        F1 = np.array(feats1, dtype=np.float64)
        F2 = np.array(feats2, dtype=np.float64)
        labels = np.array(labels)
        
        W = W.copy().astype(np.float64)
        lr = 0.001
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_updates = 0
            
            indices = list(range(len(F1)))
            random.shuffle(indices)
            
            for idx in indices:
                f1 = F1[idx]
                f2 = F2[idx]
                label = labels[idx]
                
                v1 = W.T @ f1
                v2 = W.T @ f2
                
                n1 = np.linalg.norm(v1) + 1e-10
                n2 = np.linalg.norm(v2) + 1e-10
                v1_norm = v1 / n1
                v2_norm = v2 / n2
                
                sim = np.dot(v1_norm, v2_norm)
                
                if label == 1.0:
                    if sim < 0.95:
                        grad = np.outer(f1, (v2_norm - sim * v1_norm) / n1) + \
                               np.outer(f2, (v1_norm - sim * v2_norm) / n2)
                        W += lr * grad
                        total_loss += (1 - sim)
                        n_updates += 1
                else:
                    margin = 0.1
                    if sim > margin:
                        grad = np.outer(f1, (v2_norm - sim * v1_norm) / n1) + \
                               np.outer(f2, (v1_norm - sim * v2_norm) / n2)
                        W -= lr * grad
                        total_loss += (sim - margin)
                        n_updates += 1
            
            if n_updates > 0:
                avg_loss = total_loss / n_updates
                print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, updates={n_updates}")
            
            lr *= 0.8
        
        return W.astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to 256-dim L2-normalized float32 vector.
        """
        features = self._extract_features(text)
        
        # Normalize
        features = (features - self.feature_mean) / self.feature_std
        
        # Project
        vec = self.W.T @ features
        
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        
        return vec.astype(np.float32)

    def encode_int8(self, text: str) -> np.ndarray:
        """
        Encode to true int8 vector for minimal storage (256 bytes for dim=256).

        Scales the L2-normalized float32 vector to int8 range [-127, 127].
        This is the actual quantized representation for storage.
        """
        vec_f32 = self.encode(text)
        vec_i8 = np.round(vec_f32 * 127.0).astype(np.int8)
        return vec_i8

    def save(self, path: str) -> None:
        """
        Save trained model state (W, feature_mean, feature_std) to a .npz file.

        Args:
            path: File path to save (should end in .npz)
        """
        np.savez(
            path,
            W=self.W,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            dim=np.array([self._dim]),
            feature_dim=np.array([self.feature_dim]),
            trained=np.array([self.trained]),
        )

    @classmethod
    def load(cls, path: str, seed: int = 42) -> 'FisherStructureEncoder':
        """
        Load a trained model from a .npz file.

        Args:
            path: File path to load from
            seed: Random seed (used only for numba warmup)

        Returns:
            FisherStructureEncoder instance with restored trained state
        """
        data = np.load(path)
        dim = int(data['dim'][0])
        feature_dim = int(data['feature_dim'][0])

        encoder = cls(dim=dim, feature_dim=feature_dim, seed=seed)
        encoder.W = data['W']
        encoder.feature_mean = data['feature_mean']
        encoder.feature_std = data['feature_std']
        encoder.trained = bool(data['trained'][0])

        return encoder
