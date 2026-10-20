use chromaprint_test_suite::ffi::CChromaprint;
use chromaprint_test_suite::utils::{compare_fingerprints, load_raw_audio};
use chromaprint::{Algorithm, Fingerprinter};

/// Feed `data` in `chunk` sized pieces (the whole buffer when `None`) to C
/// and Rust, and require identical raw, encoded and hashed fingerprints.
fn compare_algorithm(
    c_algo: i32,
    rs_algo: Algorithm,
    name: &str,
    data: &[i16],
    num_channels: u16,
    chunks: Option<&[usize]>,
) {
    let whole = [data.len()];
    let chunks = chunks.unwrap_or(&whole);
    let pieces = || {
        let mut pos = 0;
        chunks.iter().cycle().map_while(move |&size| {
            (pos < data.len()).then(|| {
                let end = (pos + size).min(data.len());
                let piece = &data[pos..end];
                pos = end;
                piece
            })
        })
    };

    // C implementation
    let c_ctx = CChromaprint::new(c_algo);
    c_ctx.start(44100, num_channels as i32);
    for piece in pieces() {
        c_ctx.feed(piece);
    }
    c_ctx.finish();
    let c_raw = c_ctx.get_raw_fingerprint();
    let c_encoded = c_ctx.get_fingerprint();
    let c_hash = c_ctx.get_fingerprint_hash();
    // Two empty fingerprints are equal without anything having been compared.
    assert!(!c_raw.is_empty(), "{name}: input too short, C produced no sub-fingerprints");

    // Rust implementation
    let mut rs_ctx = Fingerprinter::new(rs_algo);
    rs_ctx.start(44100, num_channels).unwrap();
    for piece in pieces() {
        rs_ctx.feed(piece).unwrap();
    }
    rs_ctx.finish().unwrap();
    let rs_raw = rs_ctx.fingerprint().to_vec();
    let rs_encoded = rs_ctx.encode();
    let rs_hash = rs_ctx.hash();

    compare_fingerprints(name, &c_raw, &rs_raw);
    assert_eq!(c_encoded, rs_encoded, "{name}: encoded fingerprints differ");
    assert_eq!(c_hash, rs_hash, "{name}: hashes differ");
}

/// The stereo test clip repeated to `seconds` of 44.1 kHz audio with
/// `num_channels` interleaved channels.
fn repeated_clip(seconds: usize, num_channels: u16) -> Vec<i16> {
    let clip = load_raw_audio("test_stereo_44100.raw");
    clip.iter().copied().cycle().take(44100 * seconds * num_channels as usize).collect()
}

/// A 200 Hz tone fading in from silence over 5 s, then 30 s of the test clip,
/// mono. Test4's silence remover cuts the intro where the tone's level
/// crosses its threshold, so the exact threshold and window decide where its
/// fingerprint starts.
fn quiet_intro() -> Vec<i16> {
    let fade = 44100 * 5;
    let mut data: Vec<i16> = (0..fade)
        .map(|i| {
            let amplitude = 300.0 * i as f64 / fade as f64;
            let phase = 2.0 * std::f64::consts::PI * 200.0 * i as f64 / 44100.0;
            (amplitude * phase.sin()).round() as i16
        })
        .collect();
    data.extend(repeated_clip(30, 1));
    data
}

/// Every algorithm is compared on long enough input to produce a fingerprint
/// (Test3 has no frame overlap and needs the most audio), mono and stereo,
/// whole and in random chunk sizes.
fn test_algorithm(c_algo: i32, rs_algo: Algorithm, name: &str) {
    compare_algorithm(c_algo, rs_algo, &format!("{name} 30 s mono"), &repeated_clip(30, 1), 1, None);
    compare_algorithm(c_algo, rs_algo, &format!("{name} quiet intro"), &quiet_intro(), 1, None);
    let stereo = repeated_clip(120, 2);
    compare_algorithm(c_algo, rs_algo, &format!("{name} 120 s stereo"), &stereo, 2, None);
    let chunks = [1, 4095, 333, 10_007, 2, 65_536, 1001];
    compare_algorithm(c_algo, rs_algo, &format!("{name} 120 s stereo, chunked"), &stereo, 2, Some(&chunks));
}

#[test]
fn test_algorithm_test1() {
    test_algorithm(0, Algorithm::Test1, "TEST1");
}

#[test]
fn test_algorithm_test2() {
    test_algorithm(1, Algorithm::Test2, "TEST2");
}

#[test]
fn test_algorithm_test3() {
    test_algorithm(2, Algorithm::Test3, "TEST3");
}

#[test]
fn test_algorithm_test4() {
    test_algorithm(3, Algorithm::Test4, "TEST4");
}

#[test]
fn test_algorithm_test5() {
    test_algorithm(4, Algorithm::Test5, "TEST5");
}
