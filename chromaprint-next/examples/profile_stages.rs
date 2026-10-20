//! Per-stage time breakdown of the fingerprinting pipeline.
//!
//! ```text
//! cargo run --release -p chromaprint-next --features profiling --example profile_stages -- \
//!     [file.raw] [sample_rate] [channels] [seconds] [runs] [algorithm]
//! ```
//!
//! `file.raw` is 16-bit little-endian interleaved PCM. It is repeated to fill a
//! single stream of `seconds` length, which is fingerprinted `runs` times. The
//! defaults use the C chromaprint stereo test file (44100 Hz, 2 channels),
//! 120 s, 20 runs, algorithm 1 (Test2, the default).

use std::path::PathBuf;
use std::time::Instant;

use chromaprint::profile;
use chromaprint::{Algorithm, Fingerprinter};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).map(String::as_str);

    let path = arg(0).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chromaprint/tests/data/test_stereo_44100.raw")
    });
    let sample_rate: u32 = arg(1).map_or(44100, |s| s.parse().expect("sample_rate"));
    let channels: u16 = arg(2).map_or(2, |s| s.parse().expect("channels"));
    let seconds: f64 = arg(3).map_or(120.0, |s| s.parse().expect("seconds"));
    let runs: u32 = arg(4).map_or(20, |s| s.parse().expect("runs"));
    let algorithm = arg(5)
        .map_or(Some(Algorithm::default()), |s| Algorithm::from_u8(s.parse().expect("algorithm")))
        .expect("algorithm must be 0-4");

    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let clip: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert!(!clip.is_empty(), "{} is empty", path.display());
    let stream_len = (seconds * sample_rate as f64) as usize * channels as usize;
    let stream: Vec<i16> = clip.iter().copied().cycle().take(stream_len).collect();

    // Warm-up run so FFT planning and first-touch page faults are not measured.
    fingerprint(&stream, sample_rate, channels, algorithm);

    profile::reset();
    let start = Instant::now();
    let mut sub_fingerprints = 0;
    for _ in 0..runs {
        sub_fingerprints = fingerprint(&stream, sample_rate, channels, algorithm);
    }
    let wall_ns = start.elapsed().as_nanos() as u64;
    let report = profile::take();

    println!(
        "{}: {seconds} s at {sample_rate} Hz x {channels} ch, {algorithm:?}, {runs} runs, {sub_fingerprints} sub-fingerprints per run",
        path.display()
    );
    println!(
        "wall {:.3} ms per run ({:.0}x realtime)\n",
        wall_ns as f64 / runs as f64 / 1e6,
        seconds * 1e9 * runs as f64 / wall_ns as f64
    );
    println!("{:<16} {:>10} {:>12} {:>12} {:>7}", "stage", "calls/run", "ms/run", "ns/call", "% wall");
    let mut rows: Vec<_> = report.iter().filter(|(_, s)| s.calls > 0).collect();
    rows.sort_by_key(|(_, s)| std::cmp::Reverse(s.nanos));
    for (stage, stats) in rows {
        print_row(stage.name(), stats.calls, stats.nanos, runs, wall_ns);
    }
    let untimed = wall_ns.saturating_sub(report.total_nanos());
    print_row("(untimed glue)", 0, untimed, runs, wall_ns);
}

fn fingerprint(stream: &[i16], sample_rate: u32, channels: u16, algorithm: Algorithm) -> usize {
    let mut fp = Fingerprinter::new(algorithm);
    fp.start(sample_rate, channels).unwrap();
    fp.feed(stream).unwrap();
    fp.finish().unwrap();
    std::hint::black_box(fp.encode());
    fp.fingerprint().len()
}

fn print_row(name: &str, calls: u64, nanos: u64, runs: u32, wall_ns: u64) {
    let per_call = if calls > 0 { format!("{:.0}", nanos as f64 / calls as f64) } else { "-".into() };
    let calls = if calls > 0 { format!("{}", calls / runs as u64) } else { "-".into() };
    println!(
        "{:<16} {:>10} {:>12.3} {:>12} {:>6.1}%",
        name,
        calls,
        nanos as f64 / runs as f64 / 1e6,
        per_call,
        nanos as f64 * 100.0 / wall_ns as f64
    );
}
