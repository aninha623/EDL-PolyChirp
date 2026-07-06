fn main() {
    println!("cargo:rerun-if-env-changed=TASK");
    println!("cargo:rerun-if-env-changed=MULT");
    println!("cargo:rerun-if-env-changed=MODEL");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_RUNTIME_IREE").is_some() {
        build_iree_emitc();
    } else if let (Ok(task), Ok(model), Ok(mult)) =
        (std::env::var("TASK"), std::env::var("MODEL"), std::env::var("MULT"))
    {
        println!("cargo:rerun-if-changed=models/{task}/{model}/size_{mult}/model.tflite");
    }

    emit_spectrogram_tables();
}

struct MelCfg {
    n: usize,
    mel_bins: usize,
    min_hz: f64,
    max_hz: f64,
}

const SAMPLE_RATE: f64 = 16_000.0;

fn pick_mel_cfg() -> Option<MelCfg> {
    if std::env::var_os("CARGO_FEATURE_INPUT_MEL_DEFAULT").is_some() {
        Some(MelCfg { n: 1024, mel_bins: 80, min_hz: 80.0, max_hz: 8000.0 })
    } else if std::env::var_os("CARGO_FEATURE_INPUT_MEL_HIGHER").is_some() {
        Some(MelCfg { n: 1024, mel_bins: 40, min_hz: 2500.0, max_hz: 8000.0 })
    } else if std::env::var_os("CARGO_FEATURE_INPUT_MEL_ULTRALIGHT").is_some() {
        Some(MelCfg { n: 512, mel_bins: 40, min_hz: 2500.0, max_hz: 8000.0 })
    } else {
        None
    }
}

fn emit_spectrogram_tables() {
    use std::io::Write;
    let Some(cfg) = pick_mel_cfg() else { return };
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // MEL_BINS is consumed by both the fast path and the runtime
    // tiny_chirp_preprocess path, so emit it unconditionally.
    let mut bins = std::fs::File::create(out_dir.join("mel_bins.rs")).unwrap();
    writeln!(bins, "pub const MEL_BINS: usize = {};", cfg.mel_bins).unwrap();

    if std::env::var_os("CARGO_FEATURE_TINY_CHIRP_PREPROCESS").is_some() {
        return;
    }

    // Hann window: w[i] = 0.5 - 0.5 cos(2*pi*i / (N-1))
    let mut hann = std::fs::File::create(out_dir.join("hann_table.rs")).unwrap();
    writeln!(hann, "pub const HANN: [f32; {}] = [", cfg.n).unwrap();
    for i in 0..cfg.n {
        let phase = 2.0 * std::f64::consts::PI * (i as f64) / (cfg.n - 1) as f64;
        writeln!(hann, "    {:.9e}f32,", (0.5 - 0.5 * phase.cos()) as f32).unwrap();
    }
    writeln!(hann, "];").unwrap();

    // Sparse triangular mel filterbank, same formula as the old runtime
    // mel_bin_edge / mel_to_hz / hz_to_mel, just moved off the hot path.
    let fft_bins = cfg.n / 2;
    let low_mel = 2595.0 * (1.0 + cfg.min_hz / 700.0).log10();
    let high_mel = 2595.0 * (1.0 + cfg.max_hz / 700.0).log10();
    let edge = |i: usize| -> usize {
        let frac = i as f64 / (cfg.mel_bins + 1) as f64;
        let mel = low_mel + frac * (high_mel - low_mel);
        let hz = 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
        let bin = ((cfg.n as f64 + 1.0) * hz / SAMPLE_RATE) as usize;
        bin.min(fft_bins - 1)
    };

    let mut entries: Vec<(u16, u8, f32)> = Vec::new();
    for m in 0..cfg.mel_bins {
        let (left, center, right) = (edge(m), edge(m + 1), edge(m + 2));
        for k in left..center {
            let w = (k - left) as f64 / (center - left).max(1) as f64;
            entries.push((k as u16, m as u8, w as f32));
        }
        for k in center..right {
            let w = (right - k) as f64 / (right - center).max(1) as f64;
            entries.push((k as u16, m as u8, w as f32));
        }
    }

    let mut mel = std::fs::File::create(out_dir.join("mel_table.rs")).unwrap();
    writeln!(mel, "pub const MEL_FILTERS_LEN: usize = {};", entries.len()).unwrap();
    writeln!(mel, "pub const MEL_FILTERS: [(u16, u8, f32); MEL_FILTERS_LEN] = [").unwrap();
    for (k, m, w) in &entries {
        writeln!(mel, "    ({k}, {m}, {:.9e}f32),", w).unwrap();
    }
    writeln!(mel, "];").unwrap();
}

fn build_iree_emitc() {
    let task = std::env::var("TASK").expect("TASK env var (set by laze)");
    let model = std::env::var("MODEL").expect("MODEL env var (set by laze)");
    let mult = std::env::var("MULT").expect("MULT env var (set by laze)");
    let target = std::env::var("TARGET").unwrap();
    let context = std::env::var("CARGO_CFG_CONTEXT").unwrap_or_default();

    let iree_path = std::env::var("IREE_PATH")
        .expect("IREE_PATH must point at the patched IREE checkout (.cargo/config.toml)");
    let clang_config = std::env::var("CLANG_CONFIG_PATH").unwrap_or_default();
    if context.contains("native") {
        println!("cargo:rustc-link-arg=--gcc-toolchain=/usr");
    } else if !clang_config.is_empty() {
        // newlib config (C startup + _sbrk for IREE's malloc), via laze `linker-clang`
        println!("cargo:rustc-link-arg=--config={clang_config}");
        // point clang's -fuse-ld=lld at the lld the rust toolchain ships
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
        let out = std::process::Command::new(&rustc).args(["--print", "sysroot"]).output().unwrap();
        let sysroot = String::from_utf8(out.stdout).unwrap();
        let host = std::env::var("HOST").unwrap();
        println!("cargo:rustc-link-arg=-B{}/lib/rustlib/{}/bin/gcc-ld", sysroot.trim(), host);
    }

    let emitc_c = format!("models/{task}/{model}/size_{mult}/iree_build/iree_module_emitc.c");
    println!("cargo:rerun-if-changed={emitc_c}");
    if !std::path::Path::new(&emitc_c).exists() {
        panic!(
            "missing IREE EmitC artifact at {emitc_c}; run the rust_bench notebook codegen cell first"
        );
    }

    // Compute kernels static lib + its query fn (resolved by the static loader).
    let static_lib = format!("models/{task}/{model}/size_{mult}/iree_build/iree_module.o");
    println!("cargo:rerun-if-changed={static_lib}");
    println!("cargo:rustc-link-arg={static_lib}");
    println!("cargo:rustc-env=IREE_LIB_QUERY_FN_NAME=iree_module_linked_library_query");

    let include_dir = format!("{iree_path}/runtime/src");
    let mut build = cc::Build::new();
    build
        .file(&emitc_c)
        .target(&target)
        .include(&include_dir)
        .define("EMITC_IMPLEMENTATION", None)
        .define("IREE_PLATFORM_GENERIC", "1")
        .warnings(false)
        .flag("-fno-stack-protector");

    for obj in build.compile_intermediates() {
        println!("cargo:rustc-link-arg={}", obj.display());
    }
}
