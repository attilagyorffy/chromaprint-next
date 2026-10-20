//! Ultra-high performance audio fingerprinting library.
//!
//! Rust port of [Chromaprint](https://acoustid.org/chromaprint), producing compact
//! 32-bit integer fingerprint arrays for audio identification via AcoustID.
//!
//! # Example
//!
//! ```no_run
//! use chromaprint::{Fingerprinter, Algorithm};
//!
//! let mut fp = Fingerprinter::new(Algorithm::default());
//! fp.start(44100, 2).unwrap();
//! // fp.feed(&pcm_samples).unwrap();
//! fp.finish().unwrap();
//! let raw = fp.fingerprint();
//! let encoded = fp.encode();
//! ```

/// Time the rest of the enclosing block against a `profile::Stage`.
/// Expands to nothing unless the `profiling` feature is enabled.
macro_rules! profile_stage {
    ($stage:ident) => {
        #[cfg(feature = "profiling")]
        let _profile_span = $crate::profile::Span::new($crate::profile::Stage::$stage);
    };
}

pub mod audio;
pub mod chroma;
pub mod codec;
pub mod config;
pub mod error;
pub mod fft;
pub mod fingerprint;
pub mod pipeline;
#[cfg(feature = "profiling")]
pub mod profile;
pub mod simd;
pub mod types;

pub use error::{Error, Result};
pub use types::{Algorithm, Fingerprint};

use pipeline::Pipeline;

/// Main entry point for audio fingerprinting.
///
/// Provides a streaming push-based API: create, start, feed samples, finish,
/// then retrieve the fingerprint.
pub struct Fingerprinter {
    algorithm: Algorithm,
    pipeline: Pipeline,
    started: bool,
}

impl Fingerprinter {
    /// Create a new fingerprinter for the given algorithm.
    pub fn new(algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            pipeline: Pipeline::new(algorithm),
            started: false,
        }
    }

    /// Start processing a new audio stream.
    pub fn start(&mut self, sample_rate: u32, num_channels: u16) -> Result<()> {
        self.pipeline.start(sample_rate, num_channels)?;
        self.started = true;
        Ok(())
    }

    /// Feed interleaved PCM i16 samples into the fingerprinter.
    pub fn feed(&mut self, samples: &[i16]) -> Result<()> {
        if !self.started {
            return Err(Error::NotStarted);
        }
        self.pipeline.feed(samples);
        Ok(())
    }

    /// Finish processing and flush remaining data.
    pub fn finish(&mut self) -> Result<()> {
        if !self.started {
            return Err(Error::NotStarted);
        }
        self.pipeline.finish();
        Ok(())
    }

    /// Get the raw fingerprint (array of 32-bit sub-fingerprints).
    pub fn fingerprint(&self) -> &[u32] {
        self.pipeline.fingerprint()
    }

    /// Get the compressed base64-encoded fingerprint string.
    pub fn encode(&self) -> String {
        fingerprint::compressor::compress(self.pipeline.fingerprint(), self.algorithm)
    }

    /// Get the SimHash of the fingerprint.
    pub fn hash(&self) -> u32 {
        fingerprint::simhash::simhash(self.pipeline.fingerprint())
    }

    /// Get the algorithm used by this fingerprinter.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }
}

/// Convenience function: fingerprint audio in one shot.
pub fn fingerprint_audio(
    samples: &[i16],
    sample_rate: u32,
    num_channels: u16,
    algorithm: Algorithm,
) -> Result<Fingerprint> {
    let mut fp = Fingerprinter::new(algorithm);
    fp.start(sample_rate, num_channels)?;
    fp.feed(samples)?;
    fp.finish()?;
    Ok(Fingerprint::new(fp.fingerprint().to_vec(), algorithm))
}

/// Encode a raw fingerprint to a compressed base64 string.
pub fn encode_fingerprint(raw: &[u32], algorithm: Algorithm) -> String {
    fingerprint::compressor::compress(raw, algorithm)
}

/// Decode a compressed base64 fingerprint string to raw sub-fingerprints.
pub fn decode_fingerprint(encoded: &str) -> Result<(Vec<u32>, Algorithm)> {
    fingerprint::decompressor::decompress(encoded)
}

/// Compute the SimHash of a raw fingerprint.
pub fn hash_fingerprint(raw: &[u32]) -> u32 {
    fingerprint::simhash::simhash(raw)
}

/// Batch fingerprint multiple audio inputs in parallel (requires "parallel" feature).
#[cfg(feature = "parallel")]
pub fn fingerprint_batch(
    inputs: &[(&[i16], u32, u16)],
    algorithm: Algorithm,
) -> Vec<Result<Fingerprint>> {
    use rayon::prelude::*;

    inputs
        .par_iter()
        .map(|&(samples, sample_rate, num_channels)| {
            fingerprint_audio(samples, sample_rate, num_channels, algorithm)
        })
        .collect()
}
