use crate::types::{ChromaVector, NUM_BANDS};

const NORM_THRESHOLD: f64 = 0.01;

/// Normalize a chroma vector using Euclidean (L2) norm.
/// If the norm is below the threshold, the vector is zeroed out.
#[inline]
pub fn normalize(features: &mut ChromaVector) {
    profile_stage!(Normalize);
    let mut sum_sq: f64 = 0.0;
    for &f in features.iter() {
        sum_sq += f * f;
    }

    if sum_sq > 0.0 {
        let norm = sum_sq.sqrt();
        if norm < NORM_THRESHOLD {
            *features = [0.0; NUM_BANDS];
        } else {
            let inv_norm = 1.0 / norm;
            for f in features.iter_mut() {
                *f *= inv_norm;
            }
        }
    } else {
        *features = [0.0; NUM_BANDS];
    }
}
