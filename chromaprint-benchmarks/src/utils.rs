use std::fs;
use std::path::PathBuf;

/// Load a raw 16-bit little-endian PCM audio file from the C test data directory.
pub fn load_raw_audio(filename: &str) -> Vec<i16> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("..");
    path.push("chromaprint");
    path.push("tests");
    path.push("data");
    path.push(filename);

    let bytes =
        fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    assert!(
        bytes.len() % 2 == 0,
        "Audio file has odd number of bytes: {}",
        bytes.len()
    );

    let (chunks, _) = bytes.as_chunks::<2>();
    chunks.iter().map(|c| i16::from_le_bytes(*c)).collect()
}

/// Generate synthetic mono PCM audio at the given sample rate for the given duration.
///
/// Uses a mix of sine waves at different frequencies to produce non-trivial audio
/// that exercises the full fingerprinting pipeline (resampling, FFT, chroma, classifiers).
pub fn generate_synthetic_audio(duration_secs: f64, sample_rate: u32) -> Vec<i16> {
    let num_samples = (duration_secs * sample_rate as f64) as usize;
    let mut data = Vec::with_capacity(num_samples);
    // C4, E4, G4, C5, E5 — a major chord with octave
    let frequencies = [261.63, 329.63, 392.0, 523.25, 659.25];

    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let mut sample: f64 = 0.0;
        for (j, &freq) in frequencies.iter().enumerate() {
            // Vary amplitude over time to create texture
            let amp = 0.15 * (1.0 + 0.5 * (t * (0.5 + j as f64 * 0.1)).sin());
            sample += amp * (2.0 * std::f64::consts::PI * freq * t).sin();
        }
        // Add slight deterministic variation (not true noise, but reproducible).
        // The multiplier is an arbitrary value kept as-is so benchmark data stays comparable.
        #[allow(clippy::approx_constant)]
        let jitter = (i as f64 * 1.618_033_988_75).sin() * 0.02;
        sample += jitter;
        let clamped = (sample * 16384.0).clamp(-32768.0, 32767.0) as i16;
        data.push(clamped);
    }
    data
}
