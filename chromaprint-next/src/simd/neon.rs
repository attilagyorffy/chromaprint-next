use std::arch::aarch64::*;

/// NEON-accelerated i16 × i16 dot product with i32 accumulator.
///
/// Uses `vmlal_s16` (signed multiply-accumulate long) to process
/// 4 i16 pairs per instruction, with two accumulator chains for
/// instruction-level parallelism.
#[inline]
pub fn dot_product_i16(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);

        let chunks = len / 8;

        for i in 0..chunks {
            let offset = i * 8;
            let a_lo = vld1_s16(a_ptr.add(offset));
            let b_lo = vld1_s16(b_ptr.add(offset));
            let a_hi = vld1_s16(a_ptr.add(offset + 4));
            let b_hi = vld1_s16(b_ptr.add(offset + 4));

            acc0 = vmlal_s16(acc0, a_lo, b_lo);
            acc1 = vmlal_s16(acc1, a_hi, b_hi);
        }

        acc0 = vaddq_s32(acc0, acc1);

        // Handle 4-element remainder chunk
        let mut offset = chunks * 8;
        if len - offset >= 4 {
            let a_v = vld1_s16(a_ptr.add(offset));
            let b_v = vld1_s16(b_ptr.add(offset));
            acc0 = vmlal_s16(acc0, a_v, b_v);
            offset += 4;
        }

        // Horizontal sum
        let mut result = vaddvq_s32(acc0);

        // Scalar tail (0-3 elements)
        for i in 0..(len - offset) {
            result = result.wrapping_add(*a_ptr.add(offset + i) as i32 * *b_ptr.add(offset + i) as i32);
        }

        result
    }
}

/// NEON-accelerated stereo-to-mono mixing.
///
/// Deinterleaves stereo pairs with `vld2q_s16`, widens to i32, averages
/// with truncation toward zero (matching Rust's integer division), and
/// narrows back to i16.
#[inline]
pub fn stereo_to_mono(input: &[i16], output: &mut [i16]) {
    assert_eq!(input.len(), output.len() * 2);
    let len = output.len();

    unsafe {
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();
        let chunks = len / 8;

        for i in 0..chunks {
            let offset = i * 8;
            // Load 8 interleaved stereo pairs, deinterleaving L and R
            let pair = vld2q_s16(in_ptr.add(offset * 2));
            let left = pair.0;
            let right = pair.1;

            // Widen to i32 and add
            let left_lo = vmovl_s16(vget_low_s16(left));
            let left_hi = vmovl_s16(vget_high_s16(left));
            let right_lo = vmovl_s16(vget_low_s16(right));
            let right_hi = vmovl_s16(vget_high_s16(right));
            let sum_lo = vaddq_s32(left_lo, right_lo);
            let sum_hi = vaddq_s32(left_hi, right_hi);

            // Truncation-toward-zero divide by 2:
            // (sum - (sum >> 31)) >> 1
            // This adds 1 to negative values before arithmetic shift,
            // converting round-toward-negative-infinity into truncation.
            let sign_lo = vshrq_n_s32::<31>(sum_lo);
            let sign_hi = vshrq_n_s32::<31>(sum_hi);
            let adj_lo = vsubq_s32(sum_lo, sign_lo);
            let adj_hi = vsubq_s32(sum_hi, sign_hi);
            let avg_lo = vshrq_n_s32::<1>(adj_lo);
            let avg_hi = vshrq_n_s32::<1>(adj_hi);

            // Narrow back to i16
            let result = vcombine_s16(vmovn_s32(avg_lo), vmovn_s32(avg_hi));
            vst1q_s16(out_ptr.add(offset), result);
        }

        // Scalar tail
        for i in (chunks * 8)..len {
            *out_ptr.add(i) =
                ((*in_ptr.add(i * 2) as i32 + *in_ptr.add(i * 2 + 1) as i32) / 2) as i16;
        }
    }
}

/// NEON-accelerated FFT window application.
///
/// Converts i16 samples to f32 and multiplies by pre-computed window
/// coefficients. Processes 8 elements per iteration (two sets of 4 for ILP).
#[inline]
pub fn window_apply(input: &[i16], window: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), window.len());
    assert_eq!(input.len(), output.len());
    let len = input.len();

    unsafe {
        let in_ptr = input.as_ptr();
        let win_ptr = window.as_ptr();
        let out_ptr = output.as_mut_ptr();
        let chunks = len / 8;

        for i in 0..chunks {
            let offset = i * 8;
            // Load 2 × 4 i16 values
            let i16_lo = vld1_s16(in_ptr.add(offset));
            let i16_hi = vld1_s16(in_ptr.add(offset + 4));
            // Sign-extend i16 → i32
            let i32_lo = vmovl_s16(i16_lo);
            let i32_hi = vmovl_s16(i16_hi);
            // Convert i32 → f32
            let f32_lo = vcvtq_f32_s32(i32_lo);
            let f32_hi = vcvtq_f32_s32(i32_hi);
            // Load window coefficients
            let win_lo = vld1q_f32(win_ptr.add(offset));
            let win_hi = vld1q_f32(win_ptr.add(offset + 4));
            // Multiply
            let res_lo = vmulq_f32(f32_lo, win_lo);
            let res_hi = vmulq_f32(f32_hi, win_hi);
            // Store
            vst1q_f32(out_ptr.add(offset), res_lo);
            vst1q_f32(out_ptr.add(offset + 4), res_hi);
        }

        // Scalar tail (0-7 elements)
        for i in (chunks * 8)..len {
            *out_ptr.add(i) = *in_ptr.add(i) as f32 * *win_ptr.add(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_dot_product_matches_scalar() {
        for len in [1, 3, 4, 7, 8, 15, 16, 30, 59, 80, 88] {
            let a: Vec<i16> = (0..len).map(|i| (i * 7 - 100) as i16).collect();
            let b: Vec<i16> = (0..len).map(|i| (i * 3 + 50) as i16).collect();

            let scalar: i32 = a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum();
            let neon = dot_product_i16(&a, &b);
            assert_eq!(scalar, neon, "mismatch for len={len}");
        }
    }

    #[test]
    fn test_neon_stereo_to_mono_matches_scalar() {
        for len in [1, 3, 4, 7, 8, 15, 16, 30, 59, 80, 88] {
            let input: Vec<i16> = (0..len * 2).map(|i| (i as i32 * 7 - 100) as i16).collect();
            let mut scalar_out = vec![0i16; len];
            let mut neon_out = vec![0i16; len];

            for i in 0..len {
                scalar_out[i] =
                    ((input[i * 2] as i32 + input[i * 2 + 1] as i32) / 2) as i16;
            }

            stereo_to_mono(&input, &mut neon_out);
            assert_eq!(scalar_out, neon_out, "mismatch for len={len}");
        }
    }

    #[test]
    fn test_neon_stereo_to_mono_truncation_toward_zero() {
        // Test cases where L+R is odd and negative — the rounding direction matters.
        // Rust/C integer division truncates toward zero: (-3)/2 = -1
        // Arithmetic right shift rounds toward -inf: (-3)>>1 = -2
        let cases: Vec<(i16, i16, i16)> = vec![
            (-2, -1, -1),  // sum=-3, (-3)/2 = -1
            (-1, -2, -1),  // sum=-3, (-3)/2 = -1
            (1, 2, 1),     // sum=3, 3/2 = 1
            (-1, 0, 0),    // sum=-1, (-1)/2 = 0
            (0, -1, 0),    // sum=-1, (-1)/2 = 0
            (0, 1, 0),     // sum=1, 1/2 = 0
            (-32768, -32767, -32767), // extreme negative
            (32767, 32766, 32766),    // extreme positive
        ];

        let input: Vec<i16> = cases.iter().flat_map(|&(l, r, _)| [l, r]).collect();
        let expected: Vec<i16> = cases.iter().map(|&(_, _, e)| e).collect();
        let mut actual = vec![0i16; cases.len()];

        stereo_to_mono(&input, &mut actual);
        assert_eq!(expected, actual, "truncation-toward-zero mismatch");
    }

    #[test]
    fn test_neon_window_apply_matches_scalar() {
        for len in [1, 3, 4, 7, 8, 15, 16, 100, 4096] {
            let input: Vec<i16> = (0..len).map(|i| ((i as i32 * 13 - 500) % 32768) as i16).collect();
            let win: Vec<f32> = (0..len).map(|i| i as f32 * 0.001).collect();
            let mut scalar_out = vec![0.0f32; len];
            let mut neon_out = vec![0.0f32; len];

            for i in 0..len {
                scalar_out[i] = input[i] as f32 * win[i];
            }

            window_apply(&input, &win, &mut neon_out);
            assert_eq!(scalar_out, neon_out, "mismatch for len={len}");
        }
    }
}
