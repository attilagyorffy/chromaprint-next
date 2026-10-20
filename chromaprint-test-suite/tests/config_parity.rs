//! The algorithm constants must equal the C library's exactly.
//!
//! Fingerprint comparisons only catch a wrong constant when some input lands
//! between the right and the wrong value: a Test1 quantizer threshold moved by
//! 1e-3 still passed every comparison on the test audio. These tests read the
//! C source in the `chromaprint` submodule and compare the tables directly.

use chromaprint::config::{get_config, CHROMA_FILTER_COEFFICIENTS};
use chromaprint::Algorithm;

fn c_source(file: &str) -> String {
    let path = format!("{}/../chromaprint/src/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Every numeric literal in `text`, in order.
fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter_map(|token| token.parse().ok())
        .collect()
}

/// The body of `static const <type> <name>[...] = { ... };`.
fn c_array<'a>(src: &'a str, name: &str) -> &'a str {
    let start = src
        .find(&format!("{name}["))
        .unwrap_or_else(|| panic!("{name} not found in C source"));
    let body = &src[start..];
    let open = body.find('{').unwrap() + 1;
    let close = body.find("};").unwrap();
    &body[open..close]
}

/// Number of the C configuration class whose classifier table `TestN` uses:
/// Test4 and Test5 inherit theirs from Test2.
fn c_table_owner(header: &str, n: usize) -> usize {
    let class = format!("class FingerprinterConfigurationTest{n} : public FingerprinterConfiguration");
    let line = header
        .lines()
        .find(|l| l.starts_with(&class))
        .unwrap_or_else(|| panic!("{class} not found in C header"));
    match line[class.len()..].trim().strip_prefix("Test") {
        Some(base) => base.parse().unwrap(),
        None => n,
    }
}

#[test]
fn classifiers_match_c_source() {
    let src = c_source("fingerprinter_configuration.cpp");
    let header = c_source("fingerprinter_configuration.h");
    let algorithms = [Algorithm::Test1, Algorithm::Test2, Algorithm::Test3, Algorithm::Test4, Algorithm::Test5];
    for (index, algorithm) in algorithms.into_iter().enumerate() {
        let n = index + 1;
        let owner = c_table_owner(&header, n);
        let table = c_array(&src, &format!("kClassifiersTest{owner}"));
        let c_rows: Vec<Vec<f64>> = table
            .lines()
            .filter(|l| l.contains("Classifier("))
            .map(numbers)
            .collect();
        let rust = get_config(algorithm).classifiers;
        assert_eq!(c_rows.len(), rust.len(), "{algorithm:?}: classifier count");
        for (i, (c, r)) in c_rows.iter().zip(rust).enumerate() {
            let f = r.filter;
            let q = r.quantizer;
            let rust_row = [f.kind as f64, f.y as f64, f.height as f64, f.width as f64, q.t0, q.t1, q.t2];
            assert_eq!(c.len(), 7, "{algorithm:?} classifier {i}: unexpected C row");
            for (field, (cv, rv)) in ["kind", "y", "height", "width", "t0", "t1", "t2"]
                .iter()
                .zip(c.iter().zip(rust_row))
            {
                assert_eq!(
                    cv.to_bits(),
                    rv.to_bits(),
                    "{algorithm:?} classifier {i} {field}: C {cv} vs Rust {rv}"
                );
            }
        }
    }
}

#[test]
fn chroma_filter_coefficients_match_c_source() {
    let src = c_source("fingerprinter_configuration.cpp");
    let c = numbers(c_array(&src, "kChromaFilterCoefficients"));
    let rust: Vec<u64> = CHROMA_FILTER_COEFFICIENTS.iter().map(|v| v.to_bits()).collect();
    assert_eq!(c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), rust);
    for algorithm in [Algorithm::Test1, Algorithm::Test2, Algorithm::Test3, Algorithm::Test4, Algorithm::Test5] {
        assert_eq!(
            get_config(algorithm).filter_coefficients,
            CHROMA_FILTER_COEFFICIENTS,
            "{algorithm:?}: filter coefficients"
        );
    }
}
