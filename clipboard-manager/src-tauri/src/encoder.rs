use std::collections::HashMap;

/// Multi-Scale Character Statistics Encoder
///
/// Port of the Python MultiScaleEncoder that achieved AUC-ROC 0.955
/// Encodes text using multiple scales of statistical features
pub struct MultiScaleEncoder {
    dim: usize,
    sub_dim: usize,
    byte_projection: Vec<Vec<f32>>,
    unicode_projection: Vec<Vec<f32>>,
}

impl MultiScaleEncoder {
    pub fn new(dim: usize, seed: u64) -> Self {
        assert_eq!(dim % 4, 0, "dim must be divisible by 4");

        let sub_dim = dim / 4;

        // Generate random projection matrices
        let byte_projection = Self::random_projection(256, sub_dim, seed);
        let unicode_projection = Self::random_projection(32, sub_dim, seed + 1);

        Self {
            dim,
            sub_dim,
            byte_projection,
            unicode_projection,
        }
    }

    fn random_projection(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f32>> {
        use std::f32::consts::PI;

        let mut rng_state = seed;
        let scale = 1.0 / (cols as f32).sqrt();

        let mut matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                // Simple LCG random number generator
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u1 = (rng_state as f32 / u64::MAX as f32).clamp(0.0, 1.0);

                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u2 = (rng_state as f32 / u64::MAX as f32).clamp(0.0, 1.0);

                // Box-Muller transform for normal distribution
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * PI * u2;
                let value = r * theta.cos() * scale;

                row.push(value);
            }
            matrix.push(row);
        }

        matrix
    }

    pub fn encode(&self, text: &str) -> Vec<f32> {
        // Extract features at each scale
        let byte_vec = self.extract_byte_features(text);
        let unicode_vec = self.extract_unicode_features(text);
        let token_vec = self.extract_token_features(text);
        let pattern_vec = self.extract_pattern_features(text);

        // Concatenate all features
        let mut vec = Vec::with_capacity(self.dim);
        vec.extend_from_slice(&byte_vec);
        vec.extend_from_slice(&unicode_vec);
        vec.extend_from_slice(&token_vec);
        vec.extend_from_slice(&pattern_vec);

        // L2 normalization
        Self::normalize(&mut vec);

        vec
    }

    fn extract_byte_features(&self, text: &str) -> Vec<f32> {
        let mut byte_hist = vec![0.0f32; 256];

        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return self.project(&byte_hist, &self.byte_projection);
        }

        for &b in bytes {
            byte_hist[b as usize] += 1.0;
        }

        // Normalize
        let total = bytes.len() as f32;
        for val in &mut byte_hist {
            *val /= total;
        }

        self.project(&byte_hist, &self.byte_projection)
    }

    fn extract_unicode_features(&self, text: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; 32];

        if text.is_empty() {
            return self.project(&features, &self.unicode_projection);
        }

        for ch in text.chars() {
            let code = ch as u32;

            // Unicode categories
            if ch.is_alphabetic() { features[0] += 1.0; }
            if ch.is_numeric() { features[1] += 1.0; }
            if ch.is_ascii_punctuation() { features[2] += 1.0; }
            if ch.is_whitespace() { features[4] += 1.0; }

            // Script detection
            if (0x41..=0x5A).contains(&code) || (0x61..=0x7A).contains(&code) {
                features[7] += 1.0;  // Latin
            } else if (0x30..=0x39).contains(&code) {
                features[8] += 1.0;  // Digit
            } else if (0xAC00..=0xD7AF).contains(&code) {
                features[9] += 1.0;  // Hangul
            } else if (0x4E00..=0x9FFF).contains(&code) || (0x3400..=0x4DBF).contains(&code) {
                features[10] += 1.0;  // CJK
            } else if (0x3040..=0x309F).contains(&code) {
                features[11] += 1.0;  // Hiragana
            } else if (0x30A0..=0x30FF).contains(&code) {
                features[12] += 1.0;  // Katakana
            } else if (0x0400..=0x04FF).contains(&code) {
                features[13] += 1.0;  // Cyrillic
            } else if (0x0600..=0x06FF).contains(&code) {
                features[14] += 1.0;  // Arabic
            }
        }

        // Normalize
        let total = text.chars().count() as f32;
        for val in &mut features {
            *val /= total;
        }

        self.project(&features, &self.unicode_projection)
    }

    fn extract_token_features(&self, text: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; self.sub_dim];

        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Token count statistics
        features[0] = (tokens.len() as f32 / 100.0).min(1.0);

        if !tokens.is_empty() {
            let lengths: Vec<usize> = tokens.iter().map(|t| t.len()).collect();
            let mean = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
            let max = *lengths.iter().max().unwrap() as f32;

            features[1] = mean / 50.0;
            features[3] = max / 100.0;

            if lengths.len() > 1 {
                let variance = lengths.iter()
                    .map(|&l| (l as f32 - mean).powi(2))
                    .sum::<f32>() / (lengths.len() - 1) as f32;
                features[2] = variance.sqrt() / 20.0;
            }
        }

        // Character class ratios
        if !text.is_empty() {
            let len = text.len() as f32;
            features[4] = text.chars().filter(|c| c.is_alphabetic()).count() as f32 / len;
            features[5] = text.chars().filter(|c| c.is_numeric()).count() as f32 / len;
            features[6] = text.chars().filter(|c| c.is_whitespace()).count() as f32 / len;
            features[7] = text.chars().filter(|c| ".,;:!?".contains(*c)).count() as f32 / len;
            features[8] = text.chars().filter(|c| c.is_uppercase()).count() as f32 / len;
            features[9] = text.chars().filter(|c| c.is_lowercase()).count() as f32 / len;
            features[12] = text.chars().filter(|c| "()[]{}".contains(*c)).count() as f32 / len;
            features[13] = text.chars().filter(|c| "<>/".contains(*c)).count() as f32 / len;
            features[14] = text.chars().filter(|c| "@#$%&*".contains(*c)).count() as f32 / len;
            features[15] = text.chars().filter(|c| "+-=|\\".contains(*c)).count() as f32 / len;
        }

        features
    }

    fn extract_pattern_features(&self, text: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; self.sub_dim];

        // Text length (log scale)
        features[0] = ((text.len() as f32 + 1.0).ln() / 10.0).min(1.0);

        if !text.is_empty() {
            let len = text.len() as f32;

            // Character diversity
            let unique_chars: std::collections::HashSet<_> = text.chars().collect();
            features[1] = unique_chars.len() as f32 / len;
            features[2] = unique_chars.len() as f32 / 256.0;

            // Entropy
            let mut char_counts = HashMap::new();
            for ch in text.chars() {
                *char_counts.entry(ch).or_insert(0) += 1;
            }

            let mut entropy = 0.0f32;
            for &count in char_counts.values() {
                let p = count as f32 / len;
                entropy -= p * p.log2();
            }
            features[3] = entropy / 8.0;

            // N-gram diversity
            if text.len() >= 2 {
                let bigrams: Vec<_> = text.chars()
                    .collect::<Vec<_>>()
                    .windows(2)
                    .map(|w| (w[0], w[1]))
                    .collect();
                let unique_bigrams: std::collections::HashSet<_> = bigrams.iter().collect();
                features[5] = unique_bigrams.len() as f32 / bigrams.len() as f32;
            }

            if text.len() >= 3 {
                let trigrams: Vec<_> = text.chars()
                    .collect::<Vec<_>>()
                    .windows(3)
                    .map(|w| (w[0], w[1], w[2]))
                    .collect();
                let unique_trigrams: std::collections::HashSet<_> = trigrams.iter().collect();
                features[6] = unique_trigrams.len() as f32 / trigrams.len() as f32;
            }

            // Character position statistics
            let chars: Vec<char> = text.chars().collect();
            if !chars.is_empty() {
                features[9] = if chars[0].is_uppercase() { 1.0 } else { 0.0 };
                features[10] = if chars[0].is_numeric() { 1.0 } else { 0.0 };
                features[11] = if chars[chars.len() - 1].is_alphabetic() { 1.0 } else { 0.0 };
                features[12] = if ".,;:!?".contains(chars[chars.len() - 1]) { 1.0 } else { 0.0 };
            }

            // Numeric and special character statistics
            features[13] = text.chars().filter(|c| c.is_numeric()).count() as f32 / len;
            features[14] = text.chars().filter(|c| ".,;:!?".contains(*c)).count() as f32 / len;
            features[15] = text.chars().filter(|c| "()[]{}".contains(*c)).count() as f32 / len;
        }

        features
    }

    fn project(&self, vec: &[f32], matrix: &[Vec<f32>]) -> Vec<f32> {
        let cols = matrix[0].len();
        let mut result = vec![0.0f32; cols];

        for (i, row) in matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[j] += vec[i] * val;
            }
        }

        result
    }

    fn normalize(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for val in vec {
                *val /= norm;
            }
        }
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_basic() {
        let encoder = MultiScaleEncoder::new(128, 42);

        let vec1 = encoder.encode("hello world");
        let vec2 = encoder.encode("hello world");

        assert_eq!(vec1.len(), 128);
        assert_eq!(vec2.len(), 128);

        let sim = cosine_similarity(&vec1, &vec2);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_similar_texts() {
        let encoder = MultiScaleEncoder::new(128, 42);

        let vec1 = encoder.encode("010-1234-5678");
        let vec2 = encoder.encode("010-9876-5432");

        let sim = cosine_similarity(&vec1, &vec2);
        assert!(sim > 0.7, "Phone numbers should be similar, got {}", sim);
    }

    #[test]
    fn test_different_texts() {
        let encoder = MultiScaleEncoder::new(128, 42);

        let vec1 = encoder.encode("010-1234-5678");
        let vec2 = encoder.encode("hello@example.com");

        let sim = cosine_similarity(&vec1, &vec2);
        assert!(sim < 0.6, "Phone and email should be different, got {}", sim);
    }
}
