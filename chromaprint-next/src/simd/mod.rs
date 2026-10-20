#[cfg(not(target_arch = "aarch64"))]
mod scalar;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse;

/// Whether SSE2 may be used. On x86_64 SSE2 is baseline, so this is a
/// compile-time constant; on 32-bit x86 without static SSE2 it falls back to
/// runtime detection.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn sse2_available() -> bool {
    cfg!(target_feature = "sse2") || is_x86_feature_detected!("sse2")
}

/// Compute the dot product of two i16 slices, returning an i32 accumulator
/// (wrapping on overflow).
///
/// Dispatches to NEON on aarch64, SSE2 on x86/x86_64, or scalar fallback.
/// Integer SIMD produces identical results to scalar — no rounding concerns.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn dot_product_i16(a: &[i16], b: &[i16]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_product_i16: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_product_i16(a, b)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if sse2_available() {
            // SAFETY: SSE2 is statically enabled or was detected at runtime,
            // and the slice lengths were asserted equal above.
            unsafe { sse::dot_product_i16(a, b) }
        } else {
            scalar::dot_product_i16(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::dot_product_i16(a, b)
    }
}

/// Mix interleaved stereo i16 pairs to mono by averaging.
///
/// `input` has `output.len() * 2` elements (L0,R0,L1,R1,...).
/// output[i] = (input[2*i] + input[2*i+1]) / 2 (truncation toward zero).
///
/// # Panics
/// Panics if `input.len() != output.len() * 2`.
#[inline]
pub fn stereo_to_mono(input: &[i16], output: &mut [i16]) {
    assert_eq!(
        input.len(),
        output.len().checked_mul(2).expect("output length overflow"),
        "stereo_to_mono: input must have exactly twice the output length"
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::stereo_to_mono(input, output)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if sse2_available() {
            // SAFETY: SSE2 is available (see `sse2_available`) and
            // `input.len() == output.len() * 2` was asserted above.
            unsafe { sse::stereo_to_mono(input, output) }
        } else {
            scalar::stereo_to_mono(input, output)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::stereo_to_mono(input, output)
    }
}

/// Convert i16 samples to f32 and multiply by window coefficients.
///
/// output[i] = input[i] as f32 * window[i]
/// All three slices must have the same length.
/// i16→f32 conversion is exact (f32 has 24-bit mantissa), so SIMD
/// produces bit-identical results to scalar.
///
/// # Panics
/// Panics if the three slices do not all have the same length.
#[inline]
pub fn window_apply(input: &[i16], window: &[f32], output: &mut [f32]) {
    assert!(
        input.len() == window.len() && input.len() == output.len(),
        "window_apply: length mismatch (input={}, window={}, output={})",
        input.len(),
        window.len(),
        output.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::window_apply(input, window, output)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if sse2_available() {
            // SAFETY: SSE2 is available (see `sse2_available`) and all
            // slice lengths were asserted equal above.
            unsafe { sse::window_apply(input, window, output) }
        } else {
            scalar::window_apply(input, window, output)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::window_apply(input, window, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift generator.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn i16(&mut self) -> i16 {
            match self.next() % 8 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => 0,
                3 => -1,
                _ => (self.next() >> 20) as i16,
            }
        }
    }

    fn gen_i16(rng: &mut Rng, n: usize) -> Vec<i16> {
        (0..n).map(|_| rng.i16()).collect()
    }

    #[test]
    fn dot_product_matches_reference() {
        // Covers every resampler filter length from 8 kHz (16 taps) to 192 kHz
        // (349 taps) input, plus every tail length of the SIMD loops.
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for len in 0..=360 {
            for _ in 0..4 {
                let a = gen_i16(&mut rng, len);
                let b = gen_i16(&mut rng, len);
                let expected = a
                    .iter()
                    .zip(&b)
                    .fold(0i32, |acc, (&x, &y)| acc.wrapping_add(x as i32 * y as i32));
                assert_eq!(dot_product_i16(&a, &b), expected, "len={len}");
            }
        }
    }

    #[test]
    fn stereo_to_mono_matches_reference() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for len in 0..=67 {
            for _ in 0..4 {
                let input = gen_i16(&mut rng, len * 2);
                let expected: Vec<i16> = (0..len)
                    .map(|i| ((input[2 * i] as i32 + input[2 * i + 1] as i32) / 2) as i16)
                    .collect();
                let mut out = vec![0x5A5Ai16; len];
                stereo_to_mono(&input, &mut out);
                assert_eq!(out, expected, "len={len}");
            }
        }
    }

    #[test]
    fn stereo_to_mono_truncates_toward_zero() {
        let input = [
            1i16,
            0,
            -1,
            0,
            3,
            0,
            -3,
            0,
            -1,
            -2,
            1,
            2,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            -3,
            2,
        ];
        let mut out = vec![0i16; input.len() / 2];
        stereo_to_mono(&input, &mut out);
        assert_eq!(out, [0, 0, 1, -1, -1, 1, i16::MIN, i16::MAX, 0]);
    }

    #[test]
    fn window_apply_matches_reference() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        for len in 0..=67 {
            let input = gen_i16(&mut rng, len);
            let window: Vec<f32> = (0..len)
                .map(|_| (rng.next() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let expected: Vec<f32> = input
                .iter()
                .zip(&window)
                .map(|(&x, &w)| x as f32 * w)
                .collect();
            let mut out = vec![f32::NAN; len];
            window_apply(&input, &window, &mut out);
            for (i, (g, e)) in out.iter().zip(&expected).enumerate() {
                assert_eq!(g.to_bits(), e.to_bits(), "len={len} i={i}");
            }
        }
    }

    #[test]
    #[should_panic]
    fn dot_product_length_mismatch_panics() {
        dot_product_i16(&[1, 2, 3], &[1, 2]);
    }

    #[test]
    #[should_panic]
    fn stereo_to_mono_length_mismatch_panics() {
        let mut out = [0i16; 3];
        stereo_to_mono(&[1, 2, 3, 4], &mut out);
    }

    #[test]
    #[should_panic]
    fn window_apply_input_window_mismatch_panics() {
        let mut out = [0f32; 3];
        window_apply(&[1, 2, 3], &[1.0, 2.0], &mut out);
    }

    #[test]
    #[should_panic]
    fn window_apply_output_mismatch_panics() {
        let mut out = [0f32; 2];
        window_apply(&[1, 2, 3], &[1.0, 2.0, 3.0], &mut out);
    }
}
