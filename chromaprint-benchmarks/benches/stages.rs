use criterion::{criterion_group, criterion_main, Criterion};

use chromaprint::audio::processor::AudioProcessor;
use chromaprint::audio::resample::AvResampleContext;
use chromaprint::chroma::extractor::ChromaExtractor;
use chromaprint::chroma::filter::ChromaFilter;
use chromaprint::chroma::normalizer;
use chromaprint::config;
use chromaprint::fft::FftProcessor;
use chromaprint::fingerprint::calculator::FingerprintCalculator;
use chromaprint::types::Algorithm;

use chromaprint_benchmarks::utils::generate_synthetic_audio;

/// Capture intermediate data by running each pipeline stage once.
struct StageData {
    /// Input: 30s synthetic audio at 44100 Hz mono
    input_audio: Vec<i16>,
    /// Input: the same 30s audio as interleaved stereo (L = input, R = inverted input)
    input_audio_stereo: Vec<i16>,
    /// After AudioProcessor: resampled to 11025 Hz mono
    resampled: Vec<i16>,
    /// After FftProcessor: power spectrum frames
    power_spectra: Vec<Vec<f32>>,
    /// After ChromaExtractor: chroma vectors
    chroma_vectors: Vec<[f64; 12]>,
}

fn prepare_stage_data() -> StageData {
    let cfg = config::get_config(Algorithm::Test2);
    let input_audio = generate_synthetic_audio(30.0, 44100);
    let input_audio_stereo: Vec<i16> = input_audio
        .iter()
        .flat_map(|&s| [s, s.wrapping_neg()])
        .collect();

    // Stage 1: AudioProcessor → resampled mono
    let mut resampled = Vec::new();
    let mut audio_proc = AudioProcessor::new(cfg.sample_rate);
    audio_proc.reset(44100, 1).unwrap();
    audio_proc.consume(&input_audio, |samples| {
        resampled.extend_from_slice(samples);
    });
    audio_proc.flush(|samples| {
        resampled.extend_from_slice(samples);
    });

    // Stage 2: FftProcessor → power spectra
    let mut power_spectra: Vec<Vec<f32>> = Vec::new();
    let mut fft = FftProcessor::new(cfg.frame_size, cfg.frame_overlap);
    fft.consume(&resampled, |spectrum| {
        power_spectra.push(spectrum.to_vec());
    });

    // Stage 3: ChromaExtractor → chroma vectors
    let chroma = ChromaExtractor::new(
        config::MIN_FREQ,
        config::MAX_FREQ,
        cfg.frame_size,
        cfg.sample_rate,
        cfg.interpolate,
    );
    let chroma_vectors: Vec<[f64; 12]> = power_spectra
        .iter()
        .map(|spectrum| chroma.extract(spectrum))
        .collect();

    StageData {
        input_audio,
        input_audio_stereo,
        resampled,
        power_spectra,
        chroma_vectors,
    }
}

fn stage_benchmarks(c: &mut Criterion) {
    let data = prepare_stage_data();
    let cfg = config::get_config(Algorithm::Test2);

    let mut group = c.benchmark_group("pipeline_stages");

    // Stage 1: Audio processing (channel mixing + resampling)
    group.bench_function("1_audio_processing", |b| {
        b.iter(|| {
            let mut proc = AudioProcessor::new(cfg.sample_rate);
            proc.reset(44100, 1).unwrap();
            proc.consume(&data.input_audio, |_| {});
            proc.flush(|_| {});
        });
    });

    // Stage 1b: Audio processing from stereo input (stereo-to-mono mixing + resampling)
    group.bench_function("1b_audio_processing_stereo", |b| {
        b.iter(|| {
            let mut proc = AudioProcessor::new(cfg.sample_rate);
            proc.reset(44100, 2).unwrap();
            proc.consume(&data.input_audio_stereo, |_| {});
            proc.flush(|_| {});
        });
    });

    // Stage 2: FFT (windowing + FFT + power spectrum)
    group.bench_function("2_fft", |b| {
        b.iter(|| {
            let mut fft = FftProcessor::new(cfg.frame_size, cfg.frame_overlap);
            fft.consume(&data.resampled, |_| {});
        });
    });

    // Stage 3: Chroma extraction (frequency → 12-band chroma)
    group.bench_function("3_chroma_extraction", |b| {
        let chroma = ChromaExtractor::new(
            config::MIN_FREQ,
            config::MAX_FREQ,
            cfg.frame_size,
            cfg.sample_rate,
            cfg.interpolate,
        );
        b.iter(|| {
            for spectrum in &data.power_spectra {
                std::hint::black_box(chroma.extract(spectrum));
            }
        });
    });

    // Stage 4: Chroma filter + normalization + fingerprint classification
    group.bench_function("4_filter_and_classify", |b| {
        b.iter(|| {
            let mut filter = ChromaFilter::new(cfg.filter_coefficients);
            let mut calc = FingerprintCalculator::new(cfg.classifiers);
            for features in &data.chroma_vectors {
                if let Some(mut filtered) = filter.consume(features) {
                    normalizer::normalize(&mut filtered);
                    calc.consume(&filtered);
                }
            }
            std::hint::black_box(calc.fingerprint().len());
        });
    });

    // Full pipeline for comparison
    group.bench_function("5_full_pipeline", |b| {
        b.iter(|| {
            let mut fp = chromaprint::Fingerprinter::new(Algorithm::Test2);
            fp.start(44100, 1).unwrap();
            fp.feed(&data.input_audio).unwrap();
            fp.finish().unwrap();
            std::hint::black_box(fp.fingerprint().len());
        });
    });

    group.finish();
}

/// Per-stream construction cost, paid once per fingerprint whatever the audio
/// length, so it dominates short clips.
fn setup_benchmarks(c: &mut Criterion) {
    let cfg = config::get_config(Algorithm::Test2);
    let mut group = c.benchmark_group("setup");

    group.bench_function("fingerprinter_new_start", |b| {
        b.iter(|| {
            let mut fp = chromaprint::Fingerprinter::new(Algorithm::Test2);
            fp.start(44100, 2).unwrap();
            fp
        });
    });
    group.bench_function("resampler_44100", |b| {
        b.iter(|| AvResampleContext::new(cfg.sample_rate as i32, 44100, 16, 8, false, 0.8));
    });
    group.bench_function("fft_processor", |b| {
        b.iter(|| FftProcessor::new(cfg.frame_size, cfg.frame_overlap));
    });
    group.bench_function("chroma_extractor", |b| {
        b.iter(|| {
            ChromaExtractor::new(
                config::MIN_FREQ,
                config::MAX_FREQ,
                cfg.frame_size,
                cfg.sample_rate,
                cfg.interpolate,
            )
        });
    });

    group.finish();
}

criterion_group!(benches, stage_benchmarks, setup_benchmarks);
criterion_main!(benches);
