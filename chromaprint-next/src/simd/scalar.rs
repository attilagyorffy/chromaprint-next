//! Scalar fallbacks. All functions are safe; mismatched lengths simply
//! process the common prefix (the public wrappers in `simd` assert equality).

/// Scalar fallback: i16 × i16 dot product with i32 accumulator (wrapping).
#[inline]
pub fn dot_product_i16(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(0i32, |acc, (&x, &y)| acc.wrapping_add(x as i32 * y as i32))
}

/// Scalar fallback: mix interleaved stereo i16 pairs to mono by averaging.
/// `input` has `output.len() * 2` elements (L0,R0,L1,R1,...).
/// output[i] = (input[2*i] + input[2*i+1]) / 2 (truncation toward zero)
#[inline]
pub fn stereo_to_mono(input: &[i16], output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len() * 2);
    for (o, lr) in output.iter_mut().zip(input.chunks_exact(2)) {
        *o = ((lr[0] as i32 + lr[1] as i32) / 2) as i16;
    }
}

/// Scalar fallback: convert i16 samples to f32 and multiply by window coefficients.
/// output[i] = input[i] as f32 * window[i]
#[inline]
pub fn window_apply(input: &[i16], window: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), window.len());
    debug_assert_eq!(input.len(), output.len());
    for ((o, &x), &w) in output.iter_mut().zip(input.iter()).zip(window.iter()) {
        *o = x as f32 * w;
    }
}
