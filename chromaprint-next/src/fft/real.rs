//! Power spectrum of a real frame.
//!
//! This is realfft's even-length forward transform (`RealToComplexEven`): a
//! half-length complex FFT of the frame read as interleaved pairs, followed by
//! a twiddle pass that separates the two interleaved real spectra. Only the
//! power spectrum is used downstream, so the twiddle pass writes `re² + im²`
//! directly instead of complex bins, and runs vectorized. Every bin goes
//! through the same f32 operations in the same order as in realfft, so the
//! result is bit-identical to realfft followed by a power-spectrum loop.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct RealPowerFft {
    len: usize,
    fft: Arc<dyn Fft<f32>>,
    /// realfft's twiddles for bins `1..len / 4`, pre-scaled by 0.5, split into
    /// real and imaginary parts for SIMD loads.
    twiddles_re: Vec<f32>,
    twiddles_im: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl RealPowerFft {
    /// Plan a transform of `len` real samples. `len` must be a multiple of 4
    /// (frame sizes are powers of two).
    pub fn new(len: usize) -> Self {
        assert!(len >= 4 && len % 4 == 0, "RealPowerFft: length {len} is not a multiple of 4");
        let fft = FftPlanner::<f32>::new().plan_fft_forward(len / 2);
        // As realfft's `compute_twiddle(i, len) * 0.5` for i in 1..len / 4.
        let (twiddles_re, twiddles_im) = (1..len / 4)
            .map(|i| {
                let angle = -2f64 * std::f64::consts::PI / len as f64 * i as f64;
                (angle.cos() as f32 * 0.5, angle.sin() as f32 * 0.5)
            })
            .unzip();
        let scratch = vec![Complex::default(); fft.get_outofplace_scratch_len()];
        Self {
            len,
            fft,
            twiddles_re,
            twiddles_im,
            spectrum: vec![Complex::default(); len / 2],
            scratch,
        }
    }

    /// Write the `len / 2 + 1` bin power spectrum of `input` to `power`.
    /// `input` is used as scratch space and left unspecified.
    pub fn process(&mut self, input: &mut [f32], power: &mut [f32]) {
        let half = self.len / 2;
        assert_eq!(input.len(), self.len);
        assert_eq!(power.len(), half + 1);

        {
            profile_stage!(Fft);
            // SAFETY: Complex<f32> is #[repr(C)] { re, im }, so `len` f32
            // values are exactly `len / 2` complex values with the same
            // alignment requirement.
            let pairs = unsafe {
                std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut Complex<f32>, half)
            };
            self.fft
                .process_outofplace_with_scratch(pairs, &mut self.spectrum, &mut self.scratch);
        }

        profile_stage!(PowerSpectrum);
        let x = &self.spectrum;
        // Bins 0 and len/2 are the sum and difference of bin 0's parts, with
        // zero imaginary part; the centre bin is conjugated. Squared as
        // `re * re + im * im` like every other bin.
        let first = Complex::new(x[0].re + x[0].im, 0.0);
        let last = Complex::new(x[0].re - x[0].im, 0.0);
        let centre = Complex::new(x[half / 2].re, -x[half / 2].im);
        power[0] = first.re * first.re + first.im * first.im;
        power[half] = last.re * last.re + last.im * last.im;
        power[half / 2] = centre.re * centre.re + centre.im * centre.im;

        // SAFETY: as above, `half` complex values are `2 * half` f32 values.
        let interleaved =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, 2 * half) };
        crate::simd::real_fft_power_pairs(interleaved, &self.twiddles_re, &self.twiddles_im, power);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use realfft::RealFftPlanner;

    /// realfft's forward transform followed by the power-spectrum loop the
    /// pipeline used before the two were fused.
    fn reference_power(frame: &[f32]) -> Vec<f32> {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(frame.len());
        let mut input = frame.to_vec();
        let mut output = fft.make_output_vec();
        fft.process(&mut input, &mut output).unwrap();
        output.iter().map(|c| c.re * c.re + c.im * c.im).collect()
    }

    fn frames(len: usize) -> Vec<Vec<f32>> {
        let mut state = 0xBB67_AE85_84CA_A73Bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut frames = vec![
            vec![0.0; len],
            (0..len).map(|i| if i == 0 { 32767.0 } else { 0.0 }).collect(),
            vec![-32768.0; len],
            (0..len).map(|i| if i % 2 == 0 { 32767.0 } else { -32768.0 }).collect(),
        ];
        // Windowed-sample range, plus small and large magnitudes, so that
        // rounding differs between any two operation orders.
        for scale in [1.0f32, 32768.0, 1e-3, 1e6] {
            for _ in 0..8 {
                frames.push((0..len).map(|_| (next() as i32 as f32 / i32::MAX as f32) * scale).collect());
            }
        }
        frames
    }

    #[test]
    fn power_matches_realfft_bit_for_bit() {
        // 4096 is the frame size of every algorithm; the others cover every
        // tail length of the four-bin SIMD loop and the smallest transforms.
        for len in [4, 8, 12, 16, 20, 24, 28, 64, 1024, 4096, 8192] {
            let mut fft = RealPowerFft::new(len);
            for (n, frame) in frames(len).iter().enumerate() {
                let mut input = frame.clone();
                let mut power = vec![f32::NAN; len / 2 + 1];
                fft.process(&mut input, &mut power);
                let want = reference_power(frame);
                for (bin, (g, w)) in power.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), w.to_bits(), "len={len} frame={n} bin={bin}: {g} != {w}");
                }
            }
        }
    }
}
