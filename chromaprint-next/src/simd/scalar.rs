//! Scalar fallbacks. All functions are safe; mismatched lengths simply
//! process the common prefix (the public wrappers in `simd` assert equality).

/// Scalar twiddle pass of a real FFT for pairs `first..=twiddles_re.len()`;
/// see `simd::real_fft_power_pairs`. SIMD versions call it for their tail.
#[inline]
pub fn real_fft_power_pairs(
    x: &[f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    power: &mut [f32],
    first: usize,
) {
    let n = x.len() / 2;
    for k in first..=twiddles_re.len() {
        let (tw_re, tw_im) = (twiddles_re[k - 1], twiddles_im[k - 1]);
        let (out_re, out_im) = (x[2 * k], x[2 * k + 1]);
        let (rev_re, rev_im) = (x[2 * (n - k)], x[2 * (n - k) + 1]);
        let (sum_re, sum_im) = (out_re + rev_re, out_im + rev_im);
        let (diff_re, diff_im) = (out_re - rev_re, out_im - rev_im);
        let twiddled_re = sum_im * tw_re + diff_re * tw_im;
        let twiddled_im = sum_im * tw_im - diff_re * tw_re;
        let half_sum_re = 0.5 * sum_re;
        let half_diff_im = 0.5 * diff_im;
        let (re, im) = (half_sum_re + twiddled_re, half_diff_im + twiddled_im);
        power[k] = re * re + im * im;
        let (re, im) = (half_sum_re - twiddled_re, twiddled_im - half_diff_im);
        power[n - k] = re * re + im * im;
    }
}

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
