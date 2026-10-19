use std::fs;
use std::path::PathBuf;

/// Load a raw 16-bit little-endian PCM audio file from the C test data directory.
pub fn load_raw_audio(filename: &str) -> Vec<i16> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("..");
    path.push("chromaprint");
    path.push("tests");
    path.push("data");
    path.push(filename);

    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    assert!(
        bytes.len() % 2 == 0,
        "Audio file has odd number of bytes: {}",
        bytes.len()
    );

    let (chunks, _) = bytes.as_chunks::<2>();
    chunks.iter().map(|c| i16::from_le_bytes(*c)).collect()
}

/// Compare two raw fingerprint arrays with detailed diff output on failure.
pub fn compare_fingerprints(name: &str, c_raw: &[u32], rs_raw: &[u32]) {
    compare_fingerprints_labeled(name, "C", c_raw, "Rust", rs_raw);
}

/// Compare two raw fingerprint arrays with custom labels and detailed diff output on failure.
pub fn compare_fingerprints_labeled(name: &str, label_a: &str, a: &[u32], label_b: &str, b: &[u32]) {
    if a == b {
        return;
    }

    let mut diffs = Vec::new();
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let a_val = a.get(i);
        let b_val = b.get(i);
        if a_val != b_val {
            diffs.push(format!(
                "  [{}] {}={} {}={}",
                i,
                label_a, a_val.map_or("MISSING".to_string(), |v| v.to_string()),
                label_b, b_val.map_or("MISSING".to_string(), |v| v.to_string()),
            ));
        }
        if diffs.len() >= 20 {
            diffs.push(format!("  ... and more (showing first 20 of {} entries)", max_len));
            break;
        }
    }

    panic!(
        "{}: fingerprints differ!\n  {} length: {}, {} length: {}\n  Differences:\n{}",
        name,
        label_a, a.len(),
        label_b, b.len(),
        diffs.join("\n"),
    );
}
