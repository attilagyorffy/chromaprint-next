# chromaprint-next

A pure-Rust, drop-in replacement for [Chromaprint](https://github.com/acoustid/chromaprint), the audio fingerprinting library used by [AcoustID](https://acoustid.org).

Produces **bit-identical fingerprints** to the C reference implementation across all five algorithm variants.

**Faster than C** — outperforms the C reference library by ~4% (269 vs 258 Melem/s at 120s), even though the C version uses Apple's hardware-optimised Accelerate/vDSP for FFT.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
chromaprint-next = "0.1"
```

### One-shot fingerprinting

```rust
use chromaprint::{fingerprint_audio, Algorithm};

let samples: Vec<i16> = todo!("load PCM audio");
let result = fingerprint_audio(&samples, 44100, 2, Algorithm::default()).unwrap();

let encoded = result.encode();   // compressed base64 string
let hash = result.hash();        // SimHash for quick comparison
```

### Streaming API

```rust
use chromaprint::{Fingerprinter, Algorithm};

let mut fp = Fingerprinter::new(Algorithm::default());
fp.start(44100, 2).unwrap();

// Feed audio in chunks
// fp.feed(&chunk).unwrap();

fp.finish().unwrap();

let raw = fp.fingerprint();    // &[u32] sub-fingerprints
let encoded = fp.encode();     // compressed base64 string
let hash = fp.hash();          // SimHash
```

### Decode existing fingerprints

```rust
use chromaprint::decode_fingerprint;

let (raw, algorithm) = decode_fingerprint("AQAAEwkjrUmS...").unwrap();
```

## Features

- `parallel` — Enables batch fingerprinting via [rayon](https://github.com/rayon-rs/rayon)

```toml
[dependencies]
chromaprint-next = { version = "0.1", features = ["parallel"] }
```

## License

Most of this crate is licensed under the [MIT License](LICENSE-MIT). The resampler module (`src/audio/resample.rs`) is a port of FFmpeg's `av_resample` and is licensed under [LGPL-2.1-or-later](LICENSE-LGPL-2.1).

The combined work is therefore licensed as `MIT AND LGPL-2.1-or-later`.
