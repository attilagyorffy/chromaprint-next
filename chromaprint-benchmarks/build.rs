use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    let mut cfg = cmake::Config::new("../chromaprint");
    cfg.define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_TOOLS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        // Never pick up a system FFmpeg: we only link libchromaprint statically
        // and do not emit link directives for libavutil/libswresample.
        .define("AUDIO_PROCESSOR_LIB", "internal");
    if target_os != "macos" {
        // macOS auto-selects vDSP (Accelerate). Elsewhere, pin the vendored
        // KissFFT so the build is hermetic regardless of installed FFmpeg/FFTW.
        cfg.define("FFT_LIB", "kissfft");
    }
    let dst = cfg.build();

    println!("cargo:rerun-if-changed=build.rs");
    // Rebuild the C library when the submodule moves; cargo checks these
    // paths recursively.
    println!("cargo:rerun-if-changed=../chromaprint/src");
    println!("cargo:rerun-if-changed=../chromaprint/CMakeLists.txt");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=chromaprint");

    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
