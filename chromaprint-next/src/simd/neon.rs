use std::arch::aarch64::*;

/// NEON-accelerated i16 × i16 dot product with i32 accumulator.
///
/// Loads 16 i16 pairs per iteration as two 128-bit vectors and feeds four
/// independent `smlal`/`smlal2` accumulator chains, so the multiply-accumulate
/// latency is hidden behind the other chains. Integer addition is associative
/// (wrapping), so the result is identical to the scalar sum.
#[inline]
pub fn dot_product_i16(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);

        let mut acc4 = vdupq_n_s32(0);
        let mut acc5 = vdupq_n_s32(0);
        let mut acc6 = vdupq_n_s32(0);
        let mut acc7 = vdupq_n_s32(0);

        let mut offset = 0;
        while offset + 32 <= len {
            let a0 = vld1q_s16(a_ptr.add(offset));
            let b0 = vld1q_s16(b_ptr.add(offset));
            let a1 = vld1q_s16(a_ptr.add(offset + 8));
            let b1 = vld1q_s16(b_ptr.add(offset + 8));
            let a2 = vld1q_s16(a_ptr.add(offset + 16));
            let b2 = vld1q_s16(b_ptr.add(offset + 16));
            let a3 = vld1q_s16(a_ptr.add(offset + 24));
            let b3 = vld1q_s16(b_ptr.add(offset + 24));

            acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
            acc1 = vmlal_high_s16(acc1, a0, b0);
            acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
            acc3 = vmlal_high_s16(acc3, a1, b1);
            acc4 = vmlal_s16(acc4, vget_low_s16(a2), vget_low_s16(b2));
            acc5 = vmlal_high_s16(acc5, a2, b2);
            acc6 = vmlal_s16(acc6, vget_low_s16(a3), vget_low_s16(b3));
            acc7 = vmlal_high_s16(acc7, a3, b3);
            offset += 32;
        }

        if len - offset >= 16 {
            let a0 = vld1q_s16(a_ptr.add(offset));
            let b0 = vld1q_s16(b_ptr.add(offset));
            let a1 = vld1q_s16(a_ptr.add(offset + 8));
            let b1 = vld1q_s16(b_ptr.add(offset + 8));
            acc4 = vmlal_s16(acc4, vget_low_s16(a0), vget_low_s16(b0));
            acc5 = vmlal_high_s16(acc5, a0, b0);
            acc6 = vmlal_s16(acc6, vget_low_s16(a1), vget_low_s16(b1));
            acc7 = vmlal_high_s16(acc7, a1, b1);
            offset += 16;
        }
        acc0 = vaddq_s32(acc0, acc4);
        acc1 = vaddq_s32(acc1, acc5);
        acc2 = vaddq_s32(acc2, acc6);
        acc3 = vaddq_s32(acc3, acc7);

        // 8-element remainder chunk
        if len - offset >= 8 {
            let a0 = vld1q_s16(a_ptr.add(offset));
            let b0 = vld1q_s16(b_ptr.add(offset));
            acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
            acc1 = vmlal_high_s16(acc1, a0, b0);
            offset += 8;
        }

        // 4-element remainder chunk
        if len - offset >= 4 {
            let a_v = vld1_s16(a_ptr.add(offset));
            let b_v = vld1_s16(b_ptr.add(offset));
            acc2 = vmlal_s16(acc2, a_v, b_v);
            offset += 4;
        }

        // Horizontal sum
        let mut result = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));

        // Scalar tail (0-3 elements)
        for i in 0..(len - offset) {
            result = result.wrapping_add(*a_ptr.add(offset + i) as i32 * *b_ptr.add(offset + i) as i32);
        }

        result
    }
}

/// NEON-accelerated batch of four i16 × i16 dot products.
///
/// Each product keeps two `smlal`/`smlal2` chains, giving eight independent
/// chains in total. The four results are reduced together with two levels of
/// pairwise adds instead of one `addv` per result.
#[inline]
pub fn dot_product_i16_x4(a: [&[i16]; 4], b: [&[i16]; 4]) -> [i32; 4] {
    let len = a[0].len();
    debug_assert!(a.iter().chain(b.iter()).all(|s| s.len() == len));

    unsafe {
        let ap = a.map(<[i16]>::as_ptr);
        let bp = b.map(<[i16]>::as_ptr);

        let mut lo = [vdupq_n_s32(0); 4];
        let mut hi = [vdupq_n_s32(0); 4];

        let mut offset = 0;
        while offset + 8 <= len {
            for k in 0..4 {
                let av = vld1q_s16(ap[k].add(offset));
                let bv = vld1q_s16(bp[k].add(offset));
                lo[k] = vmlal_s16(lo[k], vget_low_s16(av), vget_low_s16(bv));
                hi[k] = vmlal_high_s16(hi[k], av, bv);
            }
            offset += 8;
        }

        // 4-element remainder chunk
        if len - offset >= 4 {
            for k in 0..4 {
                let av = vld1_s16(ap[k].add(offset));
                let bv = vld1_s16(bp[k].add(offset));
                lo[k] = vmlal_s16(lo[k], av, bv);
            }
            offset += 4;
        }

        // Lane k of the result is the horizontal sum of product k.
        let s = [0, 1, 2, 3].map(|k| vaddq_s32(lo[k], hi[k]));
        let sums = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
        let mut result = [0i32; 4];
        vst1q_s32(result.as_mut_ptr(), sums);

        // Scalar tail (0-3 elements)
        for (k, r) in result.iter_mut().enumerate() {
            for i in offset..len {
                *r = r.wrapping_add(*ap[k].add(i) as i32 * *bp[k].add(i) as i32);
            }
        }

        result
    }
}

/// Reverse the four lanes of a vector.
#[inline(always)]
unsafe fn reverse_f32x4(v: float32x4_t) -> float32x4_t {
    let pairs_swapped = vrev64q_f32(v);
    vextq_f32::<2>(pairs_swapped, pairs_swapped)
}

/// NEON twiddle pass of a real FFT, four bin pairs per iteration; see
/// `simd::real_fft_power_pairs`.
///
/// Bins `k..k + 4` load directly; their partners `n - k - 3..=n - k` load
/// as one block and are lane-reversed. Only `vaddq`/`vsubq`/`vmulq` are used,
/// in the scalar order, never `vfmaq`, so results are bit-identical.
#[inline]
pub fn real_fft_power_pairs(x: &[f32], twiddles_re: &[f32], twiddles_im: &[f32], power: &mut [f32]) {
    let n = x.len() / 2;
    let pairs = twiddles_re.len();
    debug_assert!(2 * (pairs + 1) <= n && power.len() > n && twiddles_im.len() == pairs);

    let mut k = 1;
    // SAFETY: k + 3 <= pairs < n / 2, so bins k..k+4 and n-k-3..=n-k are
    // within the n bins of x, twiddles k-1..k+3 within the twiddle slices,
    // and power indices within power.len() > n (asserted by the caller).
    unsafe {
        let x_ptr = x.as_ptr();
        let p_ptr = power.as_mut_ptr();
        let half = vdupq_n_f32(0.5);
        while k + 3 <= pairs {
            let tw_re = vld1q_f32(twiddles_re.as_ptr().add(k - 1));
            let tw_im = vld1q_f32(twiddles_im.as_ptr().add(k - 1));
            let out = vld2q_f32(x_ptr.add(2 * k));
            let rev = vld2q_f32(x_ptr.add(2 * (n - k - 3)));
            let (out_re, out_im) = (out.0, out.1);
            let (rev_re, rev_im) = (reverse_f32x4(rev.0), reverse_f32x4(rev.1));

            let sum_re = vaddq_f32(out_re, rev_re);
            let sum_im = vaddq_f32(out_im, rev_im);
            let diff_re = vsubq_f32(out_re, rev_re);
            let diff_im = vsubq_f32(out_im, rev_im);
            let twiddled_re = vaddq_f32(vmulq_f32(sum_im, tw_re), vmulq_f32(diff_re, tw_im));
            let twiddled_im = vsubq_f32(vmulq_f32(sum_im, tw_im), vmulq_f32(diff_re, tw_re));
            let half_sum_re = vmulq_f32(half, sum_re);
            let half_diff_im = vmulq_f32(half, diff_im);

            let re = vaddq_f32(half_sum_re, twiddled_re);
            let im = vaddq_f32(half_diff_im, twiddled_im);
            vst1q_f32(p_ptr.add(k), vaddq_f32(vmulq_f32(re, re), vmulq_f32(im, im)));

            let re = vsubq_f32(half_sum_re, twiddled_re);
            let im = vsubq_f32(twiddled_im, half_diff_im);
            let rev_power = vaddq_f32(vmulq_f32(re, re), vmulq_f32(im, im));
            vst1q_f32(p_ptr.add(n - k - 3), reverse_f32x4(rev_power));

            k += 4;
        }
    }

    super::scalar::real_fft_power_pairs(x, twiddles_re, twiddles_im, power, k);
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
