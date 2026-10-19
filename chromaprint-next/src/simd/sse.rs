#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// SSE2-accelerated i16 × i16 dot product with i32 accumulator.
///
/// Uses `_mm_madd_epi16` which multiplies 8 i16 pairs and pairwise-adds
/// the products into 4 i32 results — ideal for this pattern.
///
/// # Safety
/// The CPU must support SSE2 and `a.len() == b.len()`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn dot_product_i16(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc0 = _mm_setzero_si128();
    let mut acc1 = _mm_setzero_si128();

    let chunks = len / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let a0 = _mm_loadu_si128(a_ptr.add(offset) as *const __m128i);
        let b0 = _mm_loadu_si128(b_ptr.add(offset) as *const __m128i);
        let a1 = _mm_loadu_si128(a_ptr.add(offset + 8) as *const __m128i);
        let b1 = _mm_loadu_si128(b_ptr.add(offset + 8) as *const __m128i);

        acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(a0, b0));
        acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(a1, b1));
    }

    acc0 = _mm_add_epi32(acc0, acc1);

    // Handle 8-element remainder chunk
    let mut offset = chunks * 16;
    if len - offset >= 8 {
        let a_v = _mm_loadu_si128(a_ptr.add(offset) as *const __m128i);
        let b_v = _mm_loadu_si128(b_ptr.add(offset) as *const __m128i);
        acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(a_v, b_v));
        offset += 8;
    }

    // Horizontal sum of 4 i32 lanes
    let hi64 = _mm_unpackhi_epi64(acc0, acc0);
    acc0 = _mm_add_epi32(acc0, hi64);
    let hi32 = _mm_shuffle_epi32(acc0, 1);
    let sum = _mm_add_epi32(acc0, hi32);
    let mut result = _mm_cvtsi128_si32(sum);

    // Scalar tail (0-7 elements)
    for i in 0..(len - offset) {
        result = result.wrapping_add(*a_ptr.add(offset + i) as i32 * *b_ptr.add(offset + i) as i32);
    }

    result
}

/// SSE2-accelerated stereo-to-mono mixing.
///
/// Uses `_mm_madd_epi16` with a vector of all 1s to sum adjacent L+R pairs
/// into i32, then divides by 2 with truncation toward zero and packs back to i16.
///
/// # Safety
/// The CPU must support SSE2 and `input.len() == output.len() * 2`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn stereo_to_mono(input: &[i16], output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len() * 2);
    let len = output.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let chunks = len / 8;
    let ones = _mm_set1_epi16(1);

    for i in 0..chunks {
        let offset = i * 8;
        // Load 16 i16 values (8 stereo pairs)
        let v0 = _mm_loadu_si128(in_ptr.add(offset * 2) as *const __m128i);
        let v1 = _mm_loadu_si128(in_ptr.add(offset * 2 + 8) as *const __m128i);

        // _mm_madd_epi16 with ones: sums adjacent i16 pairs into i32
        // [L0,R0,L1,R1,L2,R2,L3,R3] * [1,1,1,1,1,1,1,1] → [L0+R0, L1+R1, L2+R2, L3+R3]
        let sum0 = _mm_madd_epi16(v0, ones);
        let sum1 = _mm_madd_epi16(v1, ones);

        // Truncation-toward-zero divide by 2: (sum - (sum >> 31)) >> 1
        let sign0 = _mm_srai_epi32(sum0, 31);
        let sign1 = _mm_srai_epi32(sum1, 31);
        let adj0 = _mm_sub_epi32(sum0, sign0);
        let adj1 = _mm_sub_epi32(sum1, sign1);
        let avg0 = _mm_srai_epi32(adj0, 1);
        let avg1 = _mm_srai_epi32(adj1, 1);

        // Pack 8 i32 values back to 8 i16 values (with signed saturation)
        let result = _mm_packs_epi32(avg0, avg1);
        _mm_storeu_si128(out_ptr.add(offset) as *mut __m128i, result);
    }

    // Scalar tail
    for i in (chunks * 8)..len {
        *out_ptr.add(i) =
            ((*in_ptr.add(i * 2) as i32 + *in_ptr.add(i * 2 + 1) as i32) / 2) as i16;
    }
}

/// SSE2-accelerated FFT window application.
///
/// Converts i16 samples to f32 and multiplies by pre-computed window
/// coefficients. Loads 4 i16 values, sign-extends to i32 via
/// `_mm_srai_epi16`+`_mm_unpacklo_epi16`, converts to f32, and multiplies.
///
/// # Safety
/// The CPU must support SSE2 and all three slices must have equal length.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn window_apply(input: &[i16], window: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), window.len());
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();
    let in_ptr = input.as_ptr();
    let win_ptr = window.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        // Load 4 i16 values (64 bits) into lower half of __m128i
        let i16_vals = _mm_loadl_epi64(in_ptr.add(offset) as *const __m128i);
        // Sign-extend i16 → i32: unpack low i16 with sign extension
        let signs = _mm_srai_epi16(i16_vals, 15);
        let i32_vals = _mm_unpacklo_epi16(i16_vals, signs);
        // Convert i32 → f32
        let f32_vals = _mm_cvtepi32_ps(i32_vals);
        // Load 4 window f32 values and multiply
        let win_vals = _mm_loadu_ps(win_ptr.add(offset));
        let result = _mm_mul_ps(f32_vals, win_vals);
        _mm_storeu_ps(out_ptr.add(offset), result);
    }

    // Scalar tail (0-3 elements)
    for i in (chunks * 4)..len {
        *out_ptr.add(i) = *in_ptr.add(i) as f32 * *win_ptr.add(i);
    }
}
