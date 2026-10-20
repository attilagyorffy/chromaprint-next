use crate::types::{ChromaVector, NUM_BANDS};

const BUFFER_SIZE: usize = 8;

/// Temporal FIR filter for chroma vectors.
/// Applies a symmetric filter (default: 5-tap [0.25, 0.75, 1.0, 0.75, 0.25])
/// across consecutive chroma frames.
pub struct ChromaFilter {
    coefficients: &'static [f64],
    buffer: [[f64; NUM_BANDS]; BUFFER_SIZE],
    buffer_offset: usize,
    buffer_size: usize,
}

impl ChromaFilter {
    pub fn new(coefficients: &'static [f64]) -> Self {
        Self {
            coefficients,
            buffer: [[0.0; NUM_BANDS]; BUFFER_SIZE],
            buffer_offset: 0,
            buffer_size: 0,
        }
    }

    pub fn reset(&mut self) {
        self.buffer = [[0.0; NUM_BANDS]; BUFFER_SIZE];
        self.buffer_offset = 0;
        self.buffer_size = 0;
    }

    /// Feed a chroma vector into the filter.
    /// Returns Some(filtered_vector) when enough frames have been buffered.
    pub fn consume(&mut self, features: &ChromaVector) -> Option<ChromaVector> {
        profile_stage!(ChromaFilter);
        let idx = (self.buffer_offset + self.buffer_size) % BUFFER_SIZE;
        self.buffer[idx] = *features;
        self.buffer_size += 1;

        if self.buffer_size < self.coefficients.len() {
            return None;
        }

        let mut result = [0.0f64; NUM_BANDS];
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            let buf_idx = (self.buffer_offset + i) % BUFFER_SIZE;
            for (r, &b) in result.iter_mut().zip(self.buffer[buf_idx].iter()) {
                *r += b * coeff;
            }
        }

        self.buffer_offset = (self.buffer_offset + 1) % BUFFER_SIZE;
        self.buffer_size -= 1;

        Some(result)
    }
}
