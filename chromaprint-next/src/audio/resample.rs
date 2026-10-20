// Polyphase FIR resampler — Rust port of av_resample from FFmpeg/libav.
// Original C code: Copyright (c) 2004 Michael Niedermayer <michaelni@gmx.at>
// Licensed under LGPL-2.1-or-later. See LICENSE-LGPL-2.1 in the repository root.
//
// Uses Kaiser-windowed sinc interpolation with fixed-point i16 arithmetic.

const FILTER_SHIFT: i32 = 15;
const WINDOW_TYPE: i32 = 9; // Kaiser beta = 9

/// 0th order modified Bessel function of the first kind.
fn bessel(x: f64) -> f64 {
    let mut v: f64 = 1.0;
    let mut last_v: f64 = 0.0;
    let mut t: f64 = 1.0;
    let x = x * x / 4.0;
    let mut i = 1;
    while v != last_v {
        last_v = v;
        t *= x / (i * i) as f64;
        v += t;
        i += 1;
    }
    v
}

/// Build a polyphase filter bank.
fn build_filter(
    filter: &mut [i16],
    factor: f64,
    tap_count: usize,
    phase_count: usize,
    scale: i32,
    window_type: i32,
) {
    let center = (tap_count - 1) / 2;
    let mut tab = vec![0.0f64; tap_count];

    for ph in 0..phase_count {
        let mut norm: f64 = 0.0;
        for (i, slot) in tab.iter_mut().enumerate() {
            let x = std::f64::consts::PI
                * ((i as f64 - center as f64) - ph as f64 / phase_count as f64)
                * factor;
            let mut y = if x == 0.0 { 1.0 } else { x.sin() / x };

            // Kaiser window
            let w = 2.0 * x / (factor * tap_count as f64 * std::f64::consts::PI);
            y *= bessel(window_type as f64 * (1.0 - w * w).max(0.0).sqrt());

            *slot = y;
            norm += y;
        }

        // Normalize
        for (dst, &t) in filter[ph * tap_count..(ph + 1) * tap_count]
            .iter_mut()
            .zip(tab.iter())
        {
            let val = (t * scale as f64 / norm).round() as i32;
            *dst = val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

pub struct AvResampleContext {
    filter_bank: Vec<i16>,
    filter_length: usize,
    dst_incr: i32,
    index: i32,
    frac: i32,
    src_incr: i32,
    phase_shift: i32,
    phase_mask: i32,
}

impl AvResampleContext {
    pub fn new(
        out_rate: i32,
        in_rate: i32,
        filter_size: i32,
        phase_shift: i32,
        _linear: bool,
        cutoff: f64,
    ) -> Self {
        let factor = (out_rate as f64 * cutoff / in_rate as f64).min(1.0);
        let phase_count = 1i32 << phase_shift;

        let filter_length = ((filter_size as f64 / factor).ceil() as usize).max(1);
        let mut filter_bank = vec![0i16; filter_length * (phase_count as usize + 1)];

        build_filter(
            &mut filter_bank,
            factor,
            filter_length,
            phase_count as usize,
            1 << FILTER_SHIFT,
            WINDOW_TYPE,
        );

        // Copy wraparound for linear interpolation
        let pc = phase_count as usize;
        filter_bank[filter_length * pc] = filter_bank[filter_length - 1];
        for i in 1..filter_length {
            filter_bank[filter_length * pc + i] = filter_bank[i - 1];
        }

        AvResampleContext {
            filter_bank,
            filter_length,
            dst_incr: in_rate * phase_count,
            index: -phase_count * ((filter_length as i32 - 1) / 2),
            frac: 0,
            src_incr: out_rate,
            phase_shift,
            phase_mask: phase_count - 1,
        }
    }

    /// Resample audio data. Returns number of output samples written.
    /// `consumed` is set to the number of input samples consumed.
    pub fn resample(
        &mut self,
        dst: &mut [i16],
        src: &[i16],
        consumed: &mut usize,
        dst_size: usize,
    ) -> usize {
        profile_stage!(Resample);
        let src_size = src.len() as i32;
        let filter_length = self.filter_length;
        let filter_bank = &self.filter_bank;
        let phase_mask = self.phase_mask;
        let phase_shift = self.phase_shift;
        let src_incr = self.src_incr;
        let mut index = self.index;
        let mut frac = self.frac;
        let dst_incr_frac = self.dst_incr % src_incr;
        let dst_incr = self.dst_incr / src_incr;
        let mut dst_index = 0usize;

        while dst_index < dst_size {
            let filter_offset = filter_length * ((index & phase_mask) as usize);
            let sample_index = index >> phase_shift;

            if sample_index < 0 {
                let mut val: i32 = 0;
                for i in 0..filter_length as i32 {
                    let si = ((sample_index + i).abs() % src_size) as usize;
                    val += src[si] as i32 * filter_bank[filter_offset + i as usize] as i32;
                }
                val = (val + (1 << (FILTER_SHIFT - 1))) >> FILTER_SHIFT;
                dst[dst_index] = if (val as u32).wrapping_add(32768) > 65535 {
                    ((val >> 31) ^ 32767) as i16
                } else {
                    val as i16
                };
            } else if sample_index + filter_length as i32 > src_size {
                break;
            } else {
                // Hot path: the check above guarantees si..si+filter_length is within
                // src bounds, and filter_offset..filter_offset+filter_length is within
                // filter_bank bounds (phase < phase_count, bank has phase_count+1 phases).
                let si = sample_index as usize;
                let mut val: i32 = crate::simd::dot_product_i16(
                    &src[si..si + filter_length],
                    &filter_bank[filter_offset..filter_offset + filter_length],
                );
                val = (val + (1 << (FILTER_SHIFT - 1))) >> FILTER_SHIFT;
                dst[dst_index] = if (val as u32).wrapping_add(32768) > 65535 {
                    ((val >> 31) ^ 32767) as i16
                } else {
                    val as i16
                };
            }

            frac += dst_incr_frac;
            index += dst_incr;
            if frac >= src_incr {
                frac -= src_incr;
                index += 1;
            }
            dst_index += 1;
        }

        *consumed = (index.max(0) >> phase_shift) as usize;
        if index >= 0 {
            index &= phase_mask;
        }

        self.frac = frac;
        self.index = index;
        self.dst_incr = dst_incr_frac + src_incr * dst_incr;

        dst_index
    }
}
