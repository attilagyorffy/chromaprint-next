//! Opt-in per-stage timing for the fingerprinting pipeline.
//!
//! Enabled by the `profiling` feature. Without it this module does not exist
//! and the instrumentation points compile to nothing, so regular builds are
//! unaffected.
//!
//! Each [`Stage`] times a leaf operation that does not call back into a later
//! stage, so the recorded durations are exclusive and can be summed. Time spent
//! between stages (buffer copies, callback glue) is not attributed to any stage.
//!
//! Statistics are accumulated per thread. Call [`reset`] before the work to be
//! measured and [`take`] after it, on the same thread.
//!
//! ```no_run
//! # #[cfg(feature = "profiling")] {
//! use chromaprint::{profile, Algorithm, Fingerprinter};
//!
//! profile::reset();
//! let mut fp = Fingerprinter::new(Algorithm::default());
//! fp.start(44100, 2).unwrap();
//! // fp.feed(&pcm_samples).unwrap();
//! fp.finish().unwrap();
//! let report = profile::take();
//! for (stage, stats) in report.iter() {
//!     println!("{:<16} {:>8} calls {:>10} ns", stage.name(), stats.calls, stats.nanos);
//! }
//! # }
//! ```

use std::cell::Cell;
use std::time::Instant;

/// An instrumented pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Pipeline construction and `start` (FFT planning, resampler filter bank).
    Setup,
    /// Interleaved input to mono (copy, `simd::stereo_to_mono`, or N-channel average).
    ChannelMix,
    /// Polyphase resampling to the algorithm's sample rate.
    Resample,
    /// Hamming window applied to one frame.
    FftWindow,
    /// Half-length complex FFT of one frame.
    Fft,
    /// Real-FFT twiddle pass fused with `re² + im²`.
    PowerSpectrum,
    /// Power spectrum to 12 chroma bands.
    ChromaExtract,
    /// FIR filter over the chroma history.
    ChromaFilter,
    /// L2 normalization of one filtered chroma vector.
    Normalize,
    /// Appending one row to the rolling integral image.
    IntegralImage,
    /// Running the 16 classifiers for one sub-fingerprint.
    Classify,
    /// Fingerprint compression and base64 encoding.
    Compress,
}

impl Stage {
    /// All stages, in pipeline order.
    pub const ALL: [Stage; 12] = [
        Stage::Setup,
        Stage::ChannelMix,
        Stage::Resample,
        Stage::FftWindow,
        Stage::Fft,
        Stage::PowerSpectrum,
        Stage::ChromaExtract,
        Stage::ChromaFilter,
        Stage::Normalize,
        Stage::IntegralImage,
        Stage::Classify,
        Stage::Compress,
    ];

    /// Short, stable name for reports.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Setup => "setup",
            Stage::ChannelMix => "channel_mix",
            Stage::Resample => "resample",
            Stage::FftWindow => "fft_window",
            Stage::Fft => "fft",
            Stage::PowerSpectrum => "power_spectrum",
            Stage::ChromaExtract => "chroma_extract",
            Stage::ChromaFilter => "chroma_filter",
            Stage::Normalize => "normalize",
            Stage::IntegralImage => "integral_image",
            Stage::Classify => "classify",
            Stage::Compress => "compress",
        }
    }
}

const NUM_STAGES: usize = Stage::ALL.len();

/// Accumulated timing for one stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageStats {
    /// Number of times the stage ran.
    pub calls: u64,
    /// Total wall-clock time spent in the stage, in nanoseconds.
    pub nanos: u64,
}

/// Snapshot of all stage statistics on the current thread.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Report {
    stats: [StageStats; NUM_STAGES],
}

impl Report {
    /// Statistics for one stage.
    pub fn get(&self, stage: Stage) -> StageStats {
        self.stats[stage as usize]
    }

    /// Stages and their statistics, in pipeline order.
    pub fn iter(&self) -> impl Iterator<Item = (Stage, StageStats)> + '_ {
        Stage::ALL.iter().map(move |&s| (s, self.get(s)))
    }

    /// Sum of all stage times, in nanoseconds.
    pub fn total_nanos(&self) -> u64 {
        self.stats.iter().map(|s| s.nanos).sum()
    }
}

thread_local! {
    static STATS: Cell<Report> = const { Cell::new(Report { stats: [StageStats { calls: 0, nanos: 0 }; NUM_STAGES] }) };
}

/// Clear the current thread's statistics.
pub fn reset() {
    STATS.with(|s| s.set(Report::default()));
}

/// Return the current thread's statistics and clear them.
pub fn take() -> Report {
    STATS.with(|s| s.replace(Report::default()))
}

/// Records the time from construction to drop against a stage.
pub(crate) struct Span {
    stage: Stage,
    start: Instant,
}

impl Span {
    #[inline]
    pub(crate) fn new(stage: Stage) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        let nanos = self.start.elapsed().as_nanos() as u64;
        STATS.with(|s| {
            let mut report = s.get();
            let entry = &mut report.stats[self.stage as usize];
            entry.calls += 1;
            entry.nanos += nanos;
            s.set(report);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_records_calls_and_time_per_stage() {
        reset();
        for _ in 0..3 {
            let _span = Span::new(Stage::Fft);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let report = take();
        assert_eq!(report.get(Stage::Fft).calls, 3);
        assert!(report.get(Stage::Fft).nanos >= 3_000_000);
        assert_eq!(report.get(Stage::Resample), StageStats::default());
        assert_eq!(take(), Report::default(), "take() must clear the stats");
    }

    #[test]
    fn pipeline_records_every_stage_it_runs() {
        let samples: Vec<i16> = (0..44100 * 2 * 5)
            .map(|i| ((i as f64 * 0.05).sin() * 8000.0) as i16)
            .collect();
        reset();
        let mut fp = crate::Fingerprinter::new(crate::Algorithm::Test2);
        fp.start(44100, 2).unwrap();
        fp.feed(&samples).unwrap();
        fp.finish().unwrap();
        let _ = fp.encode();
        let report = take();
        for stage in Stage::ALL {
            assert!(report.get(stage).calls > 0, "{} was not recorded", stage.name());
        }
        assert_eq!(
            report.get(Stage::Classify).calls as usize,
            fp.fingerprint().len(),
            "one classify span per sub-fingerprint"
        );
    }
}
