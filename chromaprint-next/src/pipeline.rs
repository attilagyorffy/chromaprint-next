use crate::audio::processor::AudioProcessor;
use crate::audio::silence_remover::SilenceRemover;
use crate::chroma::extractor::ChromaExtractor;
use crate::chroma::filter::ChromaFilter;
use crate::chroma::normalizer;
use crate::config::{self, Config};
use crate::error::Result;
use crate::fft::FftProcessor;
use crate::fingerprint::calculator::FingerprintCalculator;
use crate::types::Algorithm;

/// The core fingerprinting pipeline.
/// Chains: AudioProcessor -> [SilenceRemover] -> FFT -> Chroma -> ChromaFilter -> Normalizer -> FingerprintCalculator
pub struct Pipeline {
    _config: &'static Config,
    audio_processor: AudioProcessor,
    silence_remover: Option<SilenceRemover>,
    fft: FftProcessor,
    chroma: ChromaExtractor,
    chroma_filter: ChromaFilter,
    calculator: FingerprintCalculator,
}

impl Pipeline {
    pub fn new(algorithm: Algorithm) -> Self {
        profile_stage!(Setup);
        let cfg = config::get_config(algorithm);

        let audio_processor = AudioProcessor::new(cfg.sample_rate);
        let silence_remover = if cfg.remove_silence {
            Some(SilenceRemover::new(cfg.silence_threshold))
        } else {
            None
        };
        let fft = FftProcessor::new(cfg.frame_size, cfg.frame_overlap);
        let chroma = ChromaExtractor::new(
            config::MIN_FREQ,
            config::MAX_FREQ,
            cfg.frame_size,
            cfg.sample_rate,
            cfg.interpolate,
        );
        let chroma_filter = ChromaFilter::new(cfg.filter_coefficients);
        let calculator = FingerprintCalculator::new(cfg.classifiers);

        Self {
            _config: cfg,
            audio_processor,
            silence_remover,
            fft,
            chroma,
            chroma_filter,
            calculator,
        }
    }

    pub fn start(&mut self, sample_rate: u32, num_channels: u16) -> Result<()> {
        profile_stage!(Setup);
        self.audio_processor.reset(sample_rate, num_channels)?;
        if let Some(ref mut sr) = self.silence_remover {
            sr.reset();
        }
        self.fft.reset();
        self.chroma_filter.reset();
        self.calculator.reset();
        Ok(())
    }

    pub fn feed(&mut self, samples: &[i16]) {
        let silence_remover = &mut self.silence_remover;
        let fft = &mut self.fft;
        let chroma = &self.chroma;
        let chroma_filter = &mut self.chroma_filter;
        let calculator = &mut self.calculator;

        self.audio_processor.consume(samples, |mono_samples| {
            let mut process_mono = |samples: &[i16]| {
                fft.consume(samples, |power_spectrum| {
                    let features = chroma.extract(power_spectrum);
                    if let Some(mut filtered) = chroma_filter.consume(&features) {
                        normalizer::normalize(&mut filtered);
                        calculator.consume(&filtered);
                    }
                });
            };

            if let Some(ref mut sr) = silence_remover {
                sr.consume(mono_samples, &mut process_mono);
            } else {
                process_mono(mono_samples);
            }
        });
    }

    pub fn finish(&mut self) {
        let silence_remover = &mut self.silence_remover;
        let fft = &mut self.fft;
        let chroma = &self.chroma;
        let chroma_filter = &mut self.chroma_filter;
        let calculator = &mut self.calculator;

        self.audio_processor.flush(|mono_samples| {
            let mut process_mono = |samples: &[i16]| {
                fft.consume(samples, |power_spectrum| {
                    let features = chroma.extract(power_spectrum);
                    if let Some(mut filtered) = chroma_filter.consume(&features) {
                        normalizer::normalize(&mut filtered);
                        calculator.consume(&filtered);
                    }
                });
            };

            if let Some(ref mut sr) = silence_remover {
                sr.consume(mono_samples, &mut process_mono);
            } else {
                process_mono(mono_samples);
            }
        });
    }

    pub fn fingerprint(&self) -> &[u32] {
        self.calculator.fingerprint()
    }
}
