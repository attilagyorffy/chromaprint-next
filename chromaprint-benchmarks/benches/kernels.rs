use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use chromaprint::audio::resample::AvResampleContext;
use chromaprint::simd;

use chromaprint_benchmarks::utils::generate_synthetic_audio;

/// Resampler filter lengths for 44.1 kHz and 48 kHz input to 11025 Hz.
const FILTER_LENGTHS: [usize; 2] = [80, 88];

fn kernel_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernels");

    // The resampler's inner loop: one call per output sample.
    let audio = generate_synthetic_audio(1.0, 44100);
    for len in FILTER_LENGTHS {
        let taps: Vec<i16> = (0..len).map(|i| (i as i16 * 397) ^ 0x1234).collect();
        // 1024 overlapping windows, as consecutive resampler outputs read them.
        group.bench_function(format!("dot_product_i16_{len}x1024"), |b| {
            b.iter(|| {
                let mut acc = 0i32;
                for start in 0..1024 {
                    let window = std::hint::black_box(&audio[start * 4..start * 4 + len]);
                    acc = acc.wrapping_add(simd::dot_product_i16(window, &taps));
                }
                acc
            });
        });
        // The same 1024 windows, four per call.
        group.bench_function(format!("dot_product_i16_x4_{len}x1024"), |b| {
            b.iter(|| {
                let mut acc = 0i32;
                for start in (0..1024).step_by(4) {
                    let windows = [0, 1, 2, 3].map(|k| {
                        let s = (start + k) * 4;
                        std::hint::black_box(&audio[s..s + len])
                    });
                    for v in simd::dot_product_i16_x4(windows, [taps.as_slice(); 4]) {
                        acc = acc.wrapping_add(v);
                    }
                }
                acc
            });
        });
    }

    // Windowing one FFT frame: i16 samples to f32 times the Hamming window.
    let frame: Vec<i16> = audio[..4096].to_vec();
    let window: Vec<f32> = (0..4096)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / 4095.0).cos())
        .collect();
    let mut windowed = vec![0.0f32; 4096];
    group.bench_function("window_apply_4096", |b| {
        b.iter(|| {
            simd::window_apply(std::hint::black_box(&frame), &window, &mut windowed);
            windowed[17]
        });
    });

    // The FFT of one frame: realfft's 4096-point real transform, and the
    // 2048-point complex transform it runs before its post-processing pass.
    let real_fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(4096);
    let mut real_out = real_fft.make_output_vec();
    let mut real_scratch = real_fft.make_scratch_vec();
    let frame_f32: Vec<f32> = frame.iter().map(|&s| s as f32).collect();
    group.bench_function("realfft_forward_4096", |b| {
        b.iter(|| {
            // realfft uses its input as scratch, so restore it each time.
            windowed.copy_from_slice(&frame_f32);
            real_fft
                .process_with_scratch(&mut windowed, &mut real_out, &mut real_scratch)
                .unwrap();
            real_out[17]
        });
    });
    let complex_fft = rustfft::FftPlanner::<f32>::new().plan_fft_forward(2048);
    let mut complex_in: Vec<rustfft::num_complex::Complex<f32>> = frame[..2048]
        .iter()
        .map(|&s| rustfft::num_complex::Complex::new(s as f32, -(s as f32)))
        .collect();
    let mut complex_out = complex_in.clone();
    let mut complex_scratch =
        vec![rustfft::num_complex::Complex::default(); complex_fft.get_outofplace_scratch_len()];
    group.bench_function("rustfft_complex_2048", |b| {
        let original = complex_in.clone();
        b.iter(|| {
            complex_in.copy_from_slice(&original);
            complex_fft.process_outofplace_with_scratch(
                &mut complex_in,
                &mut complex_out,
                &mut complex_scratch,
            );
            complex_out[17]
        });
    });

    // Resampling alone, without channel mixing or filter bank construction.
    let input = generate_synthetic_audio(30.0, 44100);
    for in_rate in [44100, 48000] {
        group.bench_function(format!("resample_{in_rate}_30s"), |b| {
            const BUF: usize = 32 * 1024;
            let mut out = vec![0i16; BUF];
            b.iter_batched_ref(
                || AvResampleContext::new(11025, in_rate, 16, 8, false, 0.8),
                |ctx| {
                    let mut total = 0;
                    for chunk in input.chunks(BUF) {
                        let mut consumed = 0;
                        total += ctx.resample(&mut out, chunk, &mut consumed, BUF);
                    }
                    total
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, kernel_benchmarks);
criterion_main!(benches);
