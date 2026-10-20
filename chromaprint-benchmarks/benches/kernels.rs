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
