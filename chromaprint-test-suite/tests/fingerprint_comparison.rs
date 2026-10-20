use chromaprint_test_suite::ffi::{
    CChromaprint, c_decode_fingerprint, c_encode_fingerprint, c_hash_fingerprint,
};
use chromaprint_test_suite::utils::{compare_fingerprints, load_raw_audio};
use chromaprint::{Algorithm, Fingerprinter};

const ALGORITHM_TEST2: i32 = 1;

/// Helper: run both C and Rust implementations on the same input and return their results.
fn fingerprint_both(
    data: &[i16],
    sample_rate: i32,
    num_channels: i32,
    algorithm_c: i32,
    algorithm_rs: Algorithm,
) -> (Vec<u32>, String, u32, Vec<u32>, String, u32) {
    // C implementation
    let c_ctx = CChromaprint::new(algorithm_c);
    c_ctx.start(sample_rate, num_channels);
    c_ctx.feed(data);
    c_ctx.finish();
    let c_raw = c_ctx.get_raw_fingerprint();
    let c_encoded = c_ctx.get_fingerprint();
    let c_hash = c_ctx.get_fingerprint_hash();

    // Rust implementation
    let mut rs_ctx = Fingerprinter::new(algorithm_rs);
    rs_ctx.start(sample_rate as u32, num_channels as u16).unwrap();
    rs_ctx.feed(data).unwrap();
    rs_ctx.finish().unwrap();
    let rs_raw = rs_ctx.fingerprint().to_vec();
    let rs_encoded = rs_ctx.encode();
    let rs_hash = rs_ctx.hash();

    (c_raw, c_encoded, c_hash, rs_raw, rs_encoded, rs_hash)
}

#[test]
fn test_stereo_44100_as_mono() {
    // This matches the C test_api.cpp TestFp test: loads stereo file but declares 1 channel,
    // so the interleaved stereo data is treated as mono samples.
    let data = load_raw_audio("test_stereo_44100.raw");

    let (c_raw, c_encoded, c_hash, rs_raw, rs_encoded, rs_hash) =
        fingerprint_both(&data, 44100, 1, ALGORITHM_TEST2, Algorithm::Test2);

    // Check against known expected values from C test suite
    assert_eq!(
        c_encoded, "AQAAC0kkZUqYREkUnFAXHk8uuMZl6EfO4zu-4ABKFGESWIIMEQE",
        "C encoded fingerprint doesn't match known value"
    );
    assert_eq!(c_hash, 3732003127, "C hash doesn't match known value");

    // Compare C vs Rust
    compare_fingerprints("stereo_44100_as_mono", &c_raw, &rs_raw);
    assert_eq!(
        c_encoded, rs_encoded,
        "Encoded fingerprints differ: C={}, Rust={}",
        c_encoded, rs_encoded
    );
    assert_eq!(
        c_hash, rs_hash,
        "Hashes differ: C={}, Rust={}",
        c_hash, rs_hash
    );
}

#[test]
fn test_silence() {
    // 130 * 1024 zero samples at 44100 Hz mono
    let data = vec![0i16; 130 * 1024];

    let (c_raw, c_encoded, c_hash, rs_raw, rs_encoded, rs_hash) =
        fingerprint_both(&data, 44100, 1, ALGORITHM_TEST2, Algorithm::Test2);

    // Check known values
    assert_eq!(c_encoded, "AQAAA0mUaEkSRZEGAA");
    assert_eq!(c_hash, 627964279);
    assert_eq!(c_raw.len(), 3);
    assert!(c_raw.iter().all(|&v| v == 627964279));

    // Compare
    compare_fingerprints("silence", &c_raw, &rs_raw);
    assert_eq!(c_encoded, rs_encoded, "Encoded fingerprints differ");
    assert_eq!(c_hash, rs_hash, "Hashes differ");
}

#[test]
fn test_mono_44100() {
    let data = load_raw_audio("test_mono_44100.raw");

    let (c_raw, c_encoded, c_hash, rs_raw, rs_encoded, rs_hash) =
        fingerprint_both(&data, 44100, 1, ALGORITHM_TEST2, Algorithm::Test2);

    compare_fingerprints("mono_44100", &c_raw, &rs_raw);
    assert_eq!(c_encoded, rs_encoded, "Encoded fingerprints differ");
    assert_eq!(c_hash, rs_hash, "Hashes differ");
}

#[test]
fn test_mono_11025() {
    // At the internal sample rate — no resampling needed
    let data = load_raw_audio("test_mono_11025.raw");

    let (c_raw, c_encoded, c_hash, rs_raw, rs_encoded, rs_hash) =
        fingerprint_both(&data, 11025, 1, ALGORITHM_TEST2, Algorithm::Test2);

    compare_fingerprints("mono_11025", &c_raw, &rs_raw);
    assert_eq!(c_encoded, rs_encoded, "Encoded fingerprints differ");
    assert_eq!(c_hash, rs_hash, "Hashes differ");
}

#[test]
fn test_stereo_44100() {
    // Stereo with proper 2-channel declaration (will be downmixed to mono)
    let data = load_raw_audio("test_stereo_44100.raw");

    let (c_raw, c_encoded, c_hash, rs_raw, rs_encoded, rs_hash) =
        fingerprint_both(&data, 44100, 2, ALGORITHM_TEST2, Algorithm::Test2);

    compare_fingerprints("stereo_44100", &c_raw, &rs_raw);
    assert_eq!(c_encoded, rs_encoded, "Encoded fingerprints differ");
    assert_eq!(c_hash, rs_hash, "Hashes differ");
}

#[test]
fn test_mono_8000() {
    let data = load_raw_audio("test_mono_8000.raw");

    let (c_raw, c_encoded, c_hash, rs_raw, rs_encoded, rs_hash) =
        fingerprint_both(&data, 8000, 1, ALGORITHM_TEST2, Algorithm::Test2);

    compare_fingerprints("mono_8000", &c_raw, &rs_raw);
    assert_eq!(c_encoded, rs_encoded, "Encoded fingerprints differ");
    assert_eq!(c_hash, rs_hash, "Hashes differ");
}

#[test]
fn test_encode_roundtrip_binary() {
    let fp: Vec<u32> = vec![1, 0];
    let algorithm = 55;

    let c_binary = c_encode_fingerprint(&fp, algorithm, false);
    let expected_binary: Vec<u8> = vec![55, 0, 0, 2, 65, 0];

    assert_eq!(c_binary, expected_binary, "C binary encoding doesn't match expected");

    // Rust encode (only supports base64, so test base64 path)
    let c_base64 = c_encode_fingerprint(&fp, algorithm, true);
    let c_base64_str = String::from_utf8(c_base64).unwrap();
    assert_eq!(c_base64_str, "NwAAAkEA", "C base64 encoding doesn't match");

    // Rust encode uses Algorithm enum which only goes 0-4,
    // so we test with a valid algorithm for the Rust side
    let fp2: Vec<u32> = vec![1, 0];
    let c_b64 = c_encode_fingerprint(&fp2, ALGORITHM_TEST2, true);
    let c_b64_str = String::from_utf8(c_b64).unwrap();
    let rs_b64_str = chromaprint::encode_fingerprint(&fp2, Algorithm::Test2);
    assert_eq!(
        c_b64_str, rs_b64_str,
        "Encode mismatch for [1, 0]: C={}, Rust={}",
        c_b64_str, rs_b64_str
    );
}

#[test]
fn test_decode_known_fingerprint() {
    let encoded = "AQAAEwkjrUmSJQpUHflR9mjSJMdZpcO_Imdw9dCO9Clu4_wQPvhCB01w6xAtXNcAp5RASgDBhDSCGGIAcwA";

    // C decode
    let (c_raw, c_algo) = c_decode_fingerprint(encoded.as_bytes(), true)
        .expect("C decode failed");
    assert_eq!(c_algo, 1);
    assert_eq!(c_raw.len(), 19);

    // Rust decode
    let (rs_raw, rs_algo) = chromaprint::decode_fingerprint(encoded)
        .expect("Rust decode failed");
    assert_eq!(rs_algo, Algorithm::Test2);
    assert_eq!(rs_raw.len(), 19);

    compare_fingerprints("decode_known", &c_raw, &rs_raw);
}

#[test]
fn test_simhash() {
    let fp: Vec<u32> = vec![19681, 22345, 312312, 453425];

    let c_hash = c_hash_fingerprint(&fp);
    let rs_hash = chromaprint::hash_fingerprint(&fp);

    assert_eq!(c_hash, 17249, "C simhash doesn't match expected");
    assert_eq!(
        c_hash, rs_hash,
        "SimHash mismatch: C={}, Rust={}",
        c_hash, rs_hash
    );
}

#[test]
fn test_encode_decode_roundtrip_via_both() {
    // Generate a fingerprint with C, encode with Rust, decode with C, verify it matches
    let data = load_raw_audio("test_stereo_44100.raw");

    let c_ctx = CChromaprint::new(ALGORITHM_TEST2);
    c_ctx.start(44100, 1);
    c_ctx.feed(&data);
    c_ctx.finish();
    let c_raw = c_ctx.get_raw_fingerprint();

    // Encode with Rust
    let rs_encoded = chromaprint::encode_fingerprint(&c_raw, Algorithm::Test2);

    // Decode with C
    let (c_decoded, c_algo) = c_decode_fingerprint(rs_encoded.as_bytes(), true)
        .expect("C failed to decode Rust-encoded fingerprint");
    assert_eq!(c_algo, ALGORITHM_TEST2);
    compare_fingerprints("cross_encode_decode", &c_raw, &c_decoded);

    // Encode with C, decode with Rust
    let c_encoded = c_ctx.get_fingerprint();
    let (rs_decoded, rs_algo) = chromaprint::decode_fingerprint(&c_encoded)
        .expect("Rust failed to decode C-encoded fingerprint");
    assert_eq!(rs_algo, Algorithm::Test2);
    compare_fingerprints("cross_decode_encode", &c_raw, &rs_decoded);
}

#[test]
fn test_resampler_filter_lengths() {
    // Declaring the same samples at other input rates exercises every resampler
    // filter length (and SIMD dot-product tail) that real inputs produce:
    // 16 kHz -> 30 taps, 22.05 kHz -> 40, 32 kHz -> 59, 48 kHz -> 88,
    // 88.2 kHz -> 160, 96 kHz -> 175, 192 kHz -> 349.
    // The 2 s clip is repeated to 10 s at each rate so every rate yields a fingerprint.
    let clip = load_raw_audio("test_mono_44100.raw");

    for sample_rate in [16000, 22050, 32000, 48000, 88200, 96000, 192000] {
        let data: Vec<i16> = clip.iter().copied().cycle().take(sample_rate as usize * 10).collect();
        let (c_raw, c_encoded, _, rs_raw, rs_encoded, _) =
            fingerprint_both(&data, sample_rate, 1, ALGORITHM_TEST2, Algorithm::Test2);
        assert!(!c_raw.is_empty(), "{sample_rate} Hz: empty C fingerprint");
        compare_fingerprints(&format!("mono_as_{sample_rate}"), &c_raw, &rs_raw);
        assert_eq!(c_encoded, rs_encoded, "{sample_rate} Hz: encoded fingerprints differ");
    }
}

/// Feed `data` to both implementations in the given chunk sizes (cycled until
/// the data runs out) and return (C raw, Rust raw).
fn fingerprint_both_chunked(
    data: &[i16],
    sample_rate: i32,
    num_channels: i32,
    chunk_sizes: &[usize],
) -> (Vec<u32>, Vec<u32>) {
    let c_ctx = CChromaprint::new(ALGORITHM_TEST2);
    c_ctx.start(sample_rate, num_channels);
    let mut rs_ctx = Fingerprinter::new(Algorithm::Test2);
    rs_ctx.start(sample_rate as u32, num_channels as u16).unwrap();

    let mut pos = 0;
    for &size in chunk_sizes.iter().cycle() {
        if pos == data.len() {
            break;
        }
        let end = (pos + size).min(data.len());
        c_ctx.feed(&data[pos..end]);
        rs_ctx.feed(&data[pos..end]).unwrap();
        pos = end;
    }
    c_ctx.finish();
    rs_ctx.finish().unwrap();
    (c_ctx.get_raw_fingerprint(), rs_ctx.fingerprint().to_vec())
}

/// 20 s of the stereo test clip, repeated, so every chunking yields a fingerprint.
fn stereo_20s() -> Vec<i16> {
    let clip = load_raw_audio("test_stereo_44100.raw");
    clip.iter().copied().cycle().take(44100 * 2 * 20).collect()
}

#[test]
fn test_chunks_splitting_multichannel_frames() {
    // Since v1.6.2 the C library carries a partial frame over to the next
    // feed call, so a frame split across chunks is still mixed as one frame.
    let data = stereo_20s();
    let chunkings: [&[usize]; 7] = [&[1], &[3], &[333], &[1001], &[4095], &[4096], &[1, 2, 997, 5, 4093, 7]];
    for chunks in chunkings {
        let (c_raw, rs_raw) = fingerprint_both_chunked(&data, 44100, 2, chunks);
        assert!(!c_raw.is_empty());
        compare_fingerprints(&format!("stereo chunks {chunks:?}"), &c_raw, &rs_raw);
    }

    // More than two channels takes a separate mixing path.
    for chunks in [&[1000usize][..], &[1001], &[2, 7, 1999]] {
        let (c_raw, rs_raw) = fingerprint_both_chunked(&data, 44100, 3, chunks);
        assert!(!c_raw.is_empty());
        compare_fingerprints(&format!("3 channels chunks {chunks:?}"), &c_raw, &rs_raw);
    }

    // Mono never splits a frame; it must stay unaffected.
    let (c_raw, rs_raw) = fingerprint_both_chunked(&data, 44100, 1, &[333, 1001]);
    compare_fingerprints("mono chunks [333, 1001]", &c_raw, &rs_raw);
}

#[test]
fn test_trailing_partial_frame() {
    // A stream whose length is not a whole number of frames: both drop the
    // incomplete last frame.
    let mut data = stereo_20s();
    data.push(12345);
    let (c_raw, rs_raw) = fingerprint_both_chunked(&data, 44100, 2, &[1001]);
    compare_fingerprints("trailing partial frame", &c_raw, &rs_raw);
}

#[test]
fn test_restart_discards_partial_frame() {
    // A stream that ends on a partial frame, then a restart: the partial
    // frame must not leak into the next stream.
    let data = stereo_20s();
    let first = &data[..44100 * 2 * 5 + 1];

    let c_ctx = CChromaprint::new(ALGORITHM_TEST2);
    let mut rs_ctx = Fingerprinter::new(Algorithm::Test2);
    for ctx_start in 0..2 {
        c_ctx.start(44100, 2);
        rs_ctx.start(44100, 2).unwrap();
        let stream = if ctx_start == 0 { first } else { &data[..] };
        for part in stream.chunks(1001) {
            c_ctx.feed(part);
            rs_ctx.feed(part).unwrap();
        }
        if ctx_start == 1 {
            c_ctx.finish();
            rs_ctx.finish().unwrap();
        }
    }
    let c_raw = c_ctx.get_raw_fingerprint();
    assert!(!c_raw.is_empty());
    compare_fingerprints("restart after partial frame", &c_raw, rs_ctx.fingerprint());
}

#[test]
fn test_random_chunkings_match_c() {
    // Only the fingerprint has to match C; this sweeps the chunk sizes,
    // channel counts and input rates (integer and fractional resampling
    // ratios, and no resampling) that callers can produce.
    let clip = load_raw_audio("test_stereo_44100.raw");
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for num_channels in [1, 2, 3, 6] {
        for sample_rate in [11025, 22050, 44100, 48000] {
            // 12 s of audio, plus a few samples so the stream ends mid-frame.
            let len = sample_rate as usize * num_channels as usize * 12 + 5;
            let data: Vec<i16> = clip.iter().copied().cycle().take(len).collect();
            let chunks: Vec<usize> = (0..64).map(|_| 1 + (next() % 9000) as usize).collect();
            let (c_raw, rs_raw) = fingerprint_both_chunked(&data, sample_rate, num_channels, &chunks);
            assert!(!c_raw.is_empty(), "{num_channels} ch at {sample_rate} Hz: empty C fingerprint");
            compare_fingerprints(&format!("{num_channels} ch at {sample_rate} Hz, random chunks"), &c_raw, &rs_raw);
        }
    }
}
