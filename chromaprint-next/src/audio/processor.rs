use crate::audio::resample::AvResampleContext;
use crate::error::{Error, Result};

const MIN_SAMPLE_RATE: u32 = 1000;
const MAX_BUFFER_SIZE: usize = 1024 * 32;

// Resampler configuration matching the C implementation
const RESAMPLE_FILTER_LENGTH: i32 = 16;
const RESAMPLE_PHASE_SHIFT: i32 = 8;
const RESAMPLE_LINEAR: bool = false;
const RESAMPLE_CUTOFF: f64 = 0.8;

/// Audio preprocessor that handles channel mixing to mono and resampling.
pub struct AudioProcessor {
    target_sample_rate: u32,
    num_channels: u16,
    buffer: Vec<i16>,
    buffer_offset: usize,
    resample_buffer: Vec<i16>,
    resample_ctx: Option<AvResampleContext>,
    /// Samples of an incomplete frame at the end of the last `consume` call,
    /// completed by the next one, as C chromaprint does since v1.6.2.
    leftover: Vec<i16>,
}

impl AudioProcessor {
    pub fn new(target_sample_rate: u32) -> Self {
        Self {
            target_sample_rate,
            num_channels: 0,
            buffer: vec![0i16; MAX_BUFFER_SIZE],
            buffer_offset: 0,
            resample_buffer: vec![0i16; MAX_BUFFER_SIZE],
            resample_ctx: None,
            leftover: Vec::new(),
        }
    }

    /// Reset the processor for a new audio stream.
    pub fn reset(&mut self, sample_rate: u32, num_channels: u16) -> Result<()> {
        if num_channels == 0 {
            return Err(Error::InvalidChannelCount(num_channels));
        }
        if sample_rate <= MIN_SAMPLE_RATE {
            return Err(Error::InvalidSampleRate(sample_rate));
        }

        self.num_channels = num_channels;
        self.buffer_offset = 0;
        self.leftover.clear();
        self.leftover.reserve(num_channels as usize);

        if sample_rate != self.target_sample_rate {
            self.resample_ctx = Some(AvResampleContext::new(
                self.target_sample_rate as i32,
                sample_rate as i32,
                RESAMPLE_FILTER_LENGTH,
                RESAMPLE_PHASE_SHIFT,
                RESAMPLE_LINEAR,
                RESAMPLE_CUTOFF,
            ));
        } else {
            self.resample_ctx = None;
        }

        Ok(())
    }

    /// Process interleaved multi-channel i16 samples.
    /// Mixes to mono, optionally resamples, and calls the callback with mono samples.
    ///
    /// `input` need not hold whole frames: an incomplete frame at the end is
    /// kept and completed by the next call. One still incomplete at `flush` is
    /// dropped, and `reset` discards it.
    pub fn consume<F>(&mut self, mut input: &[i16], mut callback: F)
    where
        F: FnMut(&[i16]),
    {
        let num_channels = self.num_channels as usize;
        if !self.leftover.is_empty() {
            let missing = (num_channels - self.leftover.len()).min(input.len());
            self.leftover.extend_from_slice(&input[..missing]);
            input = &input[missing..];
            if self.leftover.len() < num_channels {
                return;
            }
            let frame = std::mem::take(&mut self.leftover);
            self.consume_frames(&frame, &mut callback);
            self.leftover = frame;
            self.leftover.clear();
        }
        let (frames, partial) = input.split_at(input.len() - input.len() % num_channels);
        self.consume_frames(frames, &mut callback);
        self.leftover.extend_from_slice(partial);
    }

    /// `consume` for input holding whole frames only.
    fn consume_frames<F>(&mut self, input: &[i16], callback: &mut F)
    where
        F: FnMut(&[i16]),
    {
        let num_channels = self.num_channels as usize;
        debug_assert!(input.len() % num_channels == 0);
        let num_frames = input.len() / num_channels;

        let mut pos = 0;
        let mut remaining = num_frames;

        while remaining > 0 {
            let space = self.buffer.len() - self.buffer_offset;
            let to_load = remaining.min(space);

            match num_channels {
                1 => {
                    profile_stage!(ChannelMix);
                    self.buffer[self.buffer_offset..self.buffer_offset + to_load]
                        .copy_from_slice(&input[pos..pos + to_load]);
                }
                2 => {
                    profile_stage!(ChannelMix);
                    crate::simd::stereo_to_mono(
                        &input[pos * 2..(pos + to_load) * 2],
                        &mut self.buffer[self.buffer_offset..self.buffer_offset + to_load],
                    );
                }
                _ => {
                    profile_stage!(ChannelMix);
                    for i in 0..to_load {
                        let idx = (pos + i) * num_channels;
                        let mut sum: i32 = 0;
                        for ch in 0..num_channels {
                            sum += input[idx + ch] as i32;
                        }
                        self.buffer[self.buffer_offset + i] = (sum / num_channels as i32) as i16;
                    }
                }
            }

            self.buffer_offset += to_load;
            pos += to_load;
            remaining -= to_load;

            if self.buffer_offset == self.buffer.len() {
                self.resample_and_emit(callback);
                if self.buffer_offset == self.buffer.len() {
                    // Resample didn't consume anything — shouldn't happen, but prevent infinite loop
                    return;
                }
            }
        }
    }

    /// Flush any remaining buffered samples.
    pub fn flush<F>(&mut self, mut callback: F)
    where
        F: FnMut(&[i16]),
    {
        if self.buffer_offset > 0 {
            self.resample_and_emit(&mut callback);
        }
    }

    fn resample_and_emit<F>(&mut self, callback: &mut F)
    where
        F: FnMut(&[i16]),
    {
        if let Some(ref mut ctx) = self.resample_ctx {
            let mut consumed: usize = 0;
            let length = ctx.resample(
                &mut self.resample_buffer,
                &self.buffer[..self.buffer_offset],
                &mut consumed,
                MAX_BUFFER_SIZE,
            );
            if length > MAX_BUFFER_SIZE {
                // Shouldn't happen, but match C behavior of clamping
                callback(&self.resample_buffer[..MAX_BUFFER_SIZE]);
            } else {
                callback(&self.resample_buffer[..length]);
            }
            let remaining = self.buffer_offset as isize - consumed as isize;
            if remaining > 0 {
                self.buffer.copy_within(consumed..self.buffer_offset, 0);
                self.buffer_offset = remaining as usize;
            } else {
                self.buffer_offset = 0;
            }
        } else {
            callback(&self.buffer[..self.buffer_offset]);
            self.buffer_offset = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every mono sample `AudioProcessor` emits for `input` fed in `chunks`
    /// (cycled), followed by a flush.
    fn mono_output(input: &[i16], sample_rate: u32, num_channels: u16, chunks: &[usize]) -> Vec<i16> {
        let mut processor = AudioProcessor::new(11025);
        processor.reset(sample_rate, num_channels).unwrap();
        let mut out = Vec::new();
        let mut pos = 0;
        for &size in chunks.iter().cycle() {
            if pos == input.len() {
                break;
            }
            let end = (pos + size).min(input.len());
            processor.consume(&input[pos..end], |s| out.extend_from_slice(s));
            pos = end;
        }
        processor.flush(|s| out.extend_from_slice(s));
        out
    }

    #[test]
    fn output_does_not_depend_on_chunking() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for num_channels in 1..=6u16 {
            for sample_rate in [11025, 44100, 48000] {
                // Not a whole number of frames, so the stream ends mid-frame.
                let input: Vec<i16> = (0..100_000 + 3).map(|_| next() as i16).collect();
                let whole = mono_output(&input, sample_rate, num_channels, &[input.len()]);
                assert!(!whole.is_empty());
                for _ in 0..8 {
                    let chunks: Vec<usize> = (0..16).map(|_| 1 + (next() % 5000) as usize).collect();
                    let chunked = mono_output(&input, sample_rate, num_channels, &chunks);
                    assert!(
                        chunked == whole,
                        "{num_channels} ch at {sample_rate} Hz differs for chunks {chunks:?}"
                    );
                }
            }
        }
    }
}
