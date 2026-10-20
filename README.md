# chromaprint-next

A pure-Rust, drop-in replacement for [Chromaprint](https://github.com/acoustid/chromaprint), the audio fingerprinting library used by [AcoustID](https://acoustid.org).

Produces **bit-identical fingerprints** to the C reference implementation across all five algorithm variants, [verified](chromaprint-next/docs/implementation.md#how-each-stage-matches-the-reference) by running both implementations side-by-side on identical inputs and asserting equality of raw fingerprints, encoded strings, and hashes.

**About 2x faster than C** — chromaprint-next fingerprints 120 s of audio in 8.9 ms against the C reference library's 17.2 ms (596 vs 308 Melem/s), and short clips up to 4x faster, even though the C version uses Apple's hardware-optimised Accelerate/vDSP for FFT. This comes from SIMD kernels in the resampler, FFT post-processing and windowing, batched resampler outputs, shared filter banks, and register accumulation in chroma extraction, each verified bit-exact against the code it replaced. See [benchmarks](chromaprint-next/docs/benchmarks.md) for details.

| Audio length | C reference | chromaprint-next | rusty-chromaprint |
|--------------|------------|-----------------|-------------------|
| 2 sec | 1.36 ms | **0.32 ms** (4.2x faster) | 1.33 ms |
| 10 sec | 2.14 ms | **0.76 ms** (2.8x faster) | 3.20 ms |
| 30 sec | 4.87 ms | **2.24 ms** (2.2x faster) | 9.57 ms |
| 60 sec | 8.99 ms | **4.43 ms** (2.0x faster) | 19.07 ms |
| 120 sec | 17.21 ms | **8.88 ms** (1.9x faster) | 37.91 ms |

Test2 algorithm, 44.1 kHz mono, single thread, Apple M1 Max.

## Repository structure

```
chromaprint/              C reference implementation (git submodule)
chromaprint-next/         Rust library crate
chromaprint-test-suite/   Cross-implementation comparison tests
chromaprint-benchmarks/   Criterion benchmarks comparing all three implementations
rusty-chromaprint/        Alternative Rust port by darksv (git submodule)
```

## Why another Rust port?

[rusty-chromaprint](https://github.com/darksv/rusty-chromaprint) is an existing pure-Rust port that works well for the default algorithm. However, it uses a different resampler, doesn't reproduce certain C bugs needed for database compatibility, and has incomplete presets for 3 of the 5 algorithm variants. chromaprint-next aims for strict bit-identical output across all configurations. See the [detailed comparison](chromaprint-next/docs/comparison-with-rusty-chromaprint.md).

## Getting started

```
git clone --recursive <repo-url>
cargo build -p chromaprint-next
```

## Running tests

The test suite compares both Rust implementations (chromaprint-next and rusty-chromaprint) against the C reference library, which is built automatically via CMake. It feeds identical PCM audio through all three implementations and asserts equality of raw fingerprints, compressed encodings, and hashes.

```
cargo test -p chromaprint-test-suite
```

Some rusty-chromaprint tests are `#[ignore]`d due to known incompatibilities (resampler differences, incomplete presets). To run them:

```
cargo test -p chromaprint-test-suite -- --ignored
```

## Benchmarks

The `chromaprint-benchmarks` crate compares throughput of all three implementations using the Test2 (default) algorithm at various audio durations, and benchmarks chromaprint-next's pipeline stages, per-stream setup and SIMD kernels.

```
cargo bench -p chromaprint-benchmarks
```

For a per-stage time breakdown of a real stream, enable the opt-in `profiling` feature:

```
cargo run --release -p chromaprint-next --features profiling --example profile_stages
```

See [chromaprint-next/docs/benchmarks.md](chromaprint-next/docs/benchmarks.md) for results and analysis.

## Documentation

- [Implementation details](chromaprint-next/docs/implementation.md) — how the Rust port matches the C reference, performance characteristics, and known bug compatibility
- [Benchmarks](chromaprint-next/docs/benchmarks.md) — performance comparison across all three implementations
- [Comparison with rusty-chromaprint](chromaprint-next/docs/comparison-with-rusty-chromaprint.md) — detailed compatibility analysis

## License

Most of this project is licensed under the [MIT License](LICENSE-MIT). The resampler module (`chromaprint-next/src/audio/resample.rs`) is a port of FFmpeg's `av_resample` and is licensed under [LGPL-2.1-or-later](LICENSE-LGPL-2.1).

The combined work is therefore licensed as `MIT AND LGPL-2.1-or-later`.
