pub mod window;

use realfft::RealFftPlanner;
use realfft::RealToComplex;
use std::sync::Arc;

/// Round up to the next power of two.
fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

/// FFT processor that handles frame slicing, windowing, and FFT computation.
pub struct FftProcessor {
    frame_size: usize,
    increment: usize,
    window: Vec<f32>,
    // Ring buffer for incoming samples (power-of-2 sized for fast masking)
    ring_buffer: Vec<i16>,
    ring_mask: usize,
    ring_write: usize,
    ring_read: usize,
    ring_count: usize,
    // FFT state
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_scratch: Vec<realfft::num_complex::Complex<f32>>,
    fft_output: Vec<realfft::num_complex::Complex<f32>>,
    // Power spectrum output
    power_spectrum: Vec<f32>,
}

impl FftProcessor {
    pub fn new(frame_size: usize, frame_overlap: usize) -> Self {
        let increment = frame_size - frame_overlap;
        let window = window::hamming(frame_size);

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(frame_size);

        let fft_input = vec![0.0f32; frame_size];
        let fft_output = fft.make_output_vec();
        let fft_scratch = fft.make_scratch_vec();
        let power_spectrum = vec![0.0f32; frame_size / 2 + 1];

        // Power-of-2 ring buffer for bitmask indexing
        let ring_size = next_power_of_two(frame_size * 4);

        Self {
            frame_size,
            increment,
            window,
            ring_buffer: vec![0i16; ring_size],
            ring_mask: ring_size - 1,
            ring_write: 0,
            ring_read: 0,
            ring_count: 0,
            fft,
            fft_input,
            fft_scratch,
            fft_output,
            power_spectrum,
        }
    }

    pub fn reset(&mut self) {
        self.ring_write = 0;
        self.ring_read = 0;
        self.ring_count = 0;
    }

    /// Feed samples into the FFT processor.
    /// Calls `callback` for each complete FFT frame's power spectrum.
    pub fn consume<F>(&mut self, samples: &[i16], mut callback: F)
    where
        F: FnMut(&[f32]),
    {
        let ring_mask = self.ring_mask;
        let frame_size = self.frame_size;
        let increment = self.increment;
        let mut pos = 0;

        while pos < samples.len() {
            // Bulk-copy as many samples as possible into the ring buffer
            let space_before_wrap = (ring_mask + 1) - self.ring_write;
            let space_before_full = (ring_mask + 1) - self.ring_count;
            let can_write = space_before_wrap
                .min(space_before_full)
                .min(samples.len() - pos);
            // Limit to what gets us to exactly one frame, so we don't overshoot
            let need_for_frame = frame_size.saturating_sub(self.ring_count);
            let to_copy = can_write.min(need_for_frame.max(1));

            self.ring_buffer[self.ring_write..self.ring_write + to_copy]
                .copy_from_slice(&samples[pos..pos + to_copy]);
            self.ring_write = (self.ring_write + to_copy) & ring_mask;
            self.ring_count += to_copy;
            pos += to_copy;

            // Process frames as soon as we have enough data
            while self.ring_count >= frame_size {
                self.process_frame();
                callback(&self.power_spectrum);
                self.ring_read = (self.ring_read + increment) & ring_mask;
                self.ring_count -= increment;
            }
        }
    }

    fn process_frame(&mut self) {
        let ring_mask = self.ring_mask;
        let ring_read = self.ring_read;

        // A frame that wraps around the ring end (a quarter of all frames with a
        // ring four frames long) is windowed as two contiguous pieces.
        {
            profile_stage!(FftWindow);
            let first = self.frame_size.min(ring_mask + 1 - ring_read);
            crate::simd::window_apply(
                &self.ring_buffer[ring_read..ring_read + first],
                &self.window[..first],
                &mut self.fft_input[..first],
            );
            let rest = self.frame_size - first;
            crate::simd::window_apply(
                &self.ring_buffer[..rest],
                &self.window[first..],
                &mut self.fft_input[first..],
            );
        }

        // Compute real FFT
        {
            profile_stage!(Fft);
            self.fft
                .process_with_scratch(&mut self.fft_input, &mut self.fft_output, &mut self.fft_scratch)
                .expect("FFT computation failed");
        }

        // Compute power spectrum: |X[k]|^2 = re^2 + im^2
        profile_stage!(PowerSpectrum);
        for (i, c) in self.fft_output.iter().enumerate() {
            self.power_spectrum[i] = c.re * c.re + c.im * c.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames cut straight from the input, windowed per element, transformed and
    /// squared one by one: what the ring buffer must reproduce bit for bit.
    fn reference_spectra(samples: &[i16], frame_size: usize, increment: usize) -> Vec<Vec<f32>> {
        let window = window::hamming(frame_size);
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(frame_size);
        let mut output = fft.make_output_vec();
        (0..)
            .map(|k| k * increment)
            .take_while(|start| start + frame_size <= samples.len())
            .map(|start| {
                let mut input: Vec<f32> = samples[start..start + frame_size]
                    .iter()
                    .zip(&window)
                    .map(|(&s, &w)| s as f32 * w)
                    .collect();
                fft.process(&mut input, &mut output).unwrap();
                output.iter().map(|c| c.re * c.re + c.im * c.im).collect()
            })
            .collect()
    }

    #[test]
    fn spectra_match_reference_across_ring_wraparound() {
        let mut state = 0x6A09_E667_F3BC_C909u64;
        let samples: Vec<i16> = (0..40_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as i16
            })
            .collect();
        // 40,000 samples give 27 frames, and the read position passes the
        // 16,384-sample ring end twice, so wrapped frames are covered.
        let expected = reference_spectra(&samples, 4096, 4096 - 2731);
        assert_eq!(expected.len(), 27);

        // Uneven chunks so frames complete at every offset within a chunk.
        let mut fft = FftProcessor::new(4096, 2731);
        let mut got = Vec::new();
        let mut pos = 0;
        for chunk in (1..).map(|i| (i * 977) % 5000 + 1) {
            let end = (pos + chunk).min(samples.len());
            fft.consume(&samples[pos..end], |s| got.push(s.to_vec()));
            pos = end;
            if pos == samples.len() {
                break;
            }
        }

        assert_eq!(got.len(), expected.len());
        for (n, (g, e)) in got.iter().zip(&expected).enumerate() {
            let g: Vec<u32> = g.iter().map(|v| v.to_bits()).collect();
            let e: Vec<u32> = e.iter().map(|v| v.to_bits()).collect();
            assert!(g == e, "frame {n} differs");
        }
    }
}
