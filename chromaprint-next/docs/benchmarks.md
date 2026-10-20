# Benchmarks

Performance comparison of the three chromaprint implementations using the Test2 (default) algorithm on mono 44100 Hz input.

Benchmarks are run with [Criterion.rs](https://github.com/bheisler/criterion.rs) via the `chromaprint-benchmarks` crate. The shortest data point ("2 sec" in the bench names) is the C test suite's stereo file read as mono, 176,400 samples; longer durations use synthetic multi-frequency audio that exercises the full pipeline (resampling, FFT, chroma extraction, classification).

## Results

Measured on Apple M1 Max, macOS 26.6, Rust 1.96.0 (release profile), single thread. The C reference library uses Accelerate/vDSP for FFT. Times are criterion medians, averaged over two runs.

| Duration | C reference | chromaprint-next | rusty-chromaprint |
|----------|------------|-----------------|-------------------|
| 2 sec | 1.36 ms | 0.32 ms | 1.33 ms |
| 10 sec | 2.14 ms | 0.76 ms | 3.20 ms |
| 30 sec | 4.87 ms | 2.24 ms | 9.57 ms |
| 60 sec | 8.99 ms | 4.43 ms | 19.07 ms |
| 120 sec | 17.21 ms | 8.88 ms | 37.91 ms |

### Throughput (samples/sec)

| Duration | C reference | chromaprint-next | rusty-chromaprint |
|----------|------------|-----------------|-------------------|
| 2 sec | 129 Melem/s | 545 Melem/s | 133 Melem/s |
| 10 sec | 206 Melem/s | 579 Melem/s | 138 Melem/s |
| 30 sec | 271 Melem/s | 591 Melem/s | 138 Melem/s |
| 60 sec | 294 Melem/s | 597 Melem/s | 139 Melem/s |
| 120 sec | 308 Melem/s | 596 Melem/s | 140 Melem/s |

### Relative performance (vs C reference)

| Duration | chromaprint-next | rusty-chromaprint |
|----------|-----------------|-------------------|
| 2 sec | **0.24x (4.2x faster)** | 0.97x |
| 10 sec | **0.36x (2.8x faster)** | 1.49x |
| 30 sec | **0.46x (2.2x faster)** | 1.96x |
| 60 sec | **0.49x (2.0x faster)** | 2.12x |
| 120 sec | **0.52x (1.9x faster)** | 2.20x |

Short clips gain the most because per-stream setup is now 37 µs (see below), where C rebuilds its resampler filter bank for every stream.

### Before and after the hot-path optimisation pass

The same benchmark on the previous release of chromaprint-next, in the same session:

| Duration | before | after | change |
|----------|--------|-------|--------|
| 2 sec | 1.24 ms | 0.32 ms | −74% |
| 10 sec | 2.16 ms | 0.76 ms | −65% |
| 30 sec | 5.25 ms | 2.24 ms | −57% |
| 60 sec | 9.95 ms | 4.43 ms | −55% |
| 120 sec | 19.05 ms | 8.88 ms | −53% |

Before, chromaprint-next was 1–11% slower than C from 10 seconds up.

## Pipeline stage breakdown

Per-stage benchmarks for chromaprint-next (`--bench stages`, 30 seconds of audio at 44100 Hz):

| Stage | Time | % of total |
|-------|------|-----------|
| Audio processing (resampling 44100→11025 Hz) | 0.98 ms | 43% |
| FFT (windowing, 4096-point real FFT, power spectrum) | 1.08 ms | 48% |
| Chroma extraction (spectrum → 12-band features) | 0.10 ms | 4% |
| Filter + normalize + classify | 0.06 ms | 3% |
| **Full pipeline** (including setup) | **2.27 ms** | |

Per-stream setup (`setup` group): `Fingerprinter::new` + `start` takes 37 µs, of which 28 µs is FFT planning. The resampler filter bank (0.63 ms to build for 44.1 kHz input) is built once per process and shared.

The opt-in `profiling` feature times each leaf stage inside a real stream (120 s of the stereo test file, repeated):

```
cargo run --release -p chromaprint-next --features profiling --example profile_stages
```

| Stage | ns/call | % of wall time |
|-------|---------|---------------|
| resample | 22,614 (per 32 Ki-sample block) | 38.1% |
| fft (2048-point complex FFT) | 3,326 | 33.4% |
| power_spectrum (real-FFT twiddle pass + re² + im²) | 682 | 6.9% |
| channel_mix (stereo to mono) | 3,723 (per block) | 6.3% |
| chroma_extract | 425 | 4.3% |
| fft_window | 395 | 4.0% |
| classify | 277 | 2.7% |
| everything else | | 3.3% |

Wall time is 9.6 ms per 120 s stereo stream, about 12,500x realtime. Per-call times under ~50 ns are dominated by the timer itself.

## Optimisations applied

All optimisations preserve bit-identical output. Each one has a test comparing it with the code it replaced (or the crate it reimplements), bit for bit, plus the C comparison tests.

1. **SIMD resampler dot product** — The polyphase FIR convolution uses NEON `smlal`/`smlal2` with eight accumulator chains on aarch64 and SSE2 `_mm_madd_epi16` on x86_64. Integer arithmetic makes any accumulation order exact.

2. **Four resampler outputs per pass** — While four consecutive input windows are in bounds, they are convolved together and reduced with two levels of pairwise adds instead of one horizontal sum each. One bound check covers all four windows.

3. **Shared filter at integer rate ratios** — For 22.05, 44.1 and 88.2 kHz input the filter phase never changes, so the filter is loaded once per four outputs. This is decided once per call and selects a separate copy of the loop.

4. **Shared resampler filter banks** — Filter banks depend only on the rate pair, so a process-wide cache (up to 16 banks) hands every stream with the same rates the same immutable bank. This removed 95% of per-stream setup.

5. **Register accumulation in chroma extraction** — Consecutive FFT bins that map to the same note are summed in a register instead of through memory, in the same order, so the f64 sums are identical (11x faster).

6. **Fused real FFT and power spectrum** — `fft::real` runs rustfft's half-length complex FFT and replaces realfft's scalar twiddle pass and the separate power-spectrum loop with one NEON pass. It uses realfft's operations in realfft's order, with no fused multiply-add.

7. **SIMD windowing for wrapped frames** — A quarter of frames cross the end of the sample ring buffer; they are now windowed as two contiguous SIMD calls instead of one masked index at a time.

8. **SIMD stereo downmix and windowing** — NEON/SSE2 versions of `stereo_to_mono` and `window_apply`.

## Compatibility guard

The C comparison tests do not detect every last-bit floating-point difference: replacing one multiply-add in the FFT post-processing with a fused multiply-add still passed them. Changes to floating-point code are therefore guarded by bit-exact tests against the previous Rust implementation or the `realfft` crate, not only by comparison with C.

## Observations

- **chromaprint-next is about 2x faster than the C reference** on full-length tracks and 4x faster on short clips, despite the C library using Apple's Accelerate/vDSP for FFT.

- **chromaprint-next is 4–5x faster than rusty-chromaprint.** rusty-chromaprint uses `rubato`, a general-purpose resampler with higher per-sample overhead, instead of the fixed-point `av_resample` port.

- **Resampling and the complex FFT now account for 72% of time.** The FFT is left to rustfft: a different FFT algorithm would change rounding, which the C comparison tests cannot be relied on to catch.

- **Not covered by this benchmark:** 48 kHz input takes the general resampler path (no shared filter), and x86_64 runs the SSE2/scalar kernels; neither was measured here.

## Running the benchmarks

```
cargo bench -p chromaprint-benchmarks
```

Criterion generates HTML reports with violin plots and detailed statistics at `target/criterion/fingerprint_test2/report/index.html`.

Per-stage pipeline breakdown and per-stream setup:

```
cargo bench -p chromaprint-benchmarks --bench stages
```

SIMD kernels, the resampler on its own (44.1 and 48 kHz), and the FFT:

```
cargo bench -p chromaprint-benchmarks --bench kernels
```

To compare against a saved baseline:

```
cargo bench -p chromaprint-benchmarks -- --save-baseline v1
# ... make changes ...
cargo bench -p chromaprint-benchmarks -- --baseline v1
```

To run a specific data size:

```
cargo bench -p chromaprint-benchmarks -- "30sec"
```
