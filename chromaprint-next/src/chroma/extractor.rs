use crate::types::{ChromaVector, NUM_BANDS};

// Use f64 to match the C implementation which uses double throughout.
const BASE_FREQ: f64 = 440.0 / 16.0; // 27.5 Hz (A0)

/// Maps FFT power spectrum bins to 12-band chroma features.
pub struct ChromaExtractor {
    notes: Vec<u8>,
    notes_frac: Vec<f64>,
    min_index: usize,
    max_index: usize,
    interpolate: bool,
}

impl ChromaExtractor {
    pub fn new(min_freq: u32, max_freq: u32, frame_size: usize, sample_rate: u32, interpolate: bool) -> Self {
        let mut extractor = Self {
            notes: vec![0; frame_size],
            notes_frac: vec![0.0; frame_size],
            min_index: 0,
            max_index: 0,
            interpolate,
        };
        extractor.prepare_notes(min_freq, max_freq, frame_size, sample_rate);
        extractor
    }

    fn prepare_notes(&mut self, min_freq: u32, max_freq: u32, frame_size: usize, sample_rate: u32) {
        self.min_index = freq_to_index(min_freq as f64, frame_size, sample_rate).max(1);
        self.max_index = freq_to_index(max_freq as f64, frame_size, sample_rate).min(frame_size / 2);

        for i in self.min_index..self.max_index {
            let freq = index_to_freq(i, frame_size, sample_rate);
            let octave = freq_to_octave(freq);
            let note = NUM_BANDS as f64 * (octave - octave.floor());
            // C uses (char)note — truncation toward zero, same as i8 cast.
            // Since note is in [0, 12), casting to i8 then u8 matches C behavior.
            self.notes[i] = (note as i8) as u8;
            self.notes_frac[i] = note - (note as i8 as f64);
        }
    }

    /// Extract chroma features from an FFT power spectrum frame.
    #[inline]
    pub fn extract(&self, frame: &[f32]) -> ChromaVector {
        profile_stage!(ChromaExtract);
        let mut features = [0.0f64; NUM_BANDS];

        #[allow(clippy::needless_range_loop)] // index shared across notes/notes_frac/frame
        for i in self.min_index..self.max_index {
            let note = self.notes[i] as usize;
            let energy = frame[i] as f64;

            if self.interpolate {
                let frac = self.notes_frac[i];
                let mut note2 = note;
                let mut a: f64 = 1.0;

                if frac < 0.5 {
                    note2 = (note + NUM_BANDS - 1) % NUM_BANDS;
                    a = 0.5 + frac;
                } else if frac > 0.5 {
                    note2 = (note + 1) % NUM_BANDS;
                    a = 1.5 - frac;
                }

                features[note] += energy * a;
                features[note2] += energy * (1.0 - a);
            } else {
                features[note] += energy;
            }
        }

        features
    }
}

#[inline]
fn index_to_freq(i: usize, frame_size: usize, sample_rate: u32) -> f64 {
    i as f64 * sample_rate as f64 / frame_size as f64
}

#[inline]
fn freq_to_index(freq: f64, frame_size: usize, sample_rate: u32) -> usize {
    (frame_size as f64 * freq / sample_rate as f64).round() as usize
}

#[inline]
fn freq_to_octave(freq: f64) -> f64 {
    (freq / BASE_FREQ).ln() / std::f64::consts::LN_2
}
