#![no_main]
#![no_std]

#[cfg(not(feature = "energy_trace"))]
use ariel_os::debug::{ExitCode, exit};
#[cfg(not(feature = "energy_trace"))]
use ariel_os::log::info;
#[cfg(not(feature = "energy_trace"))]
use ariel_os::time::Instant;

#[cfg(any(
    feature = "input_mel_default",
    feature = "input_mel_higher",
    feature = "input_mel_ultralight",
))]
mod spectrogram;
#[cfg(feature = "input_time")]
mod audio_raw;

#[cfg(all(feature = "runtime_iree", not(feature = "bench_spectro_only")))]
mod model_runtime_iree;

// GPIO bracket for PPK2 energy capture; see the energy-trace main below.
#[cfg(feature = "energy_trace")]
mod trace;


// the energy build only reads AUDIO; the log-facing consts go unused there
#[cfg_attr(feature = "energy_trace", allow(dead_code))]
mod audio_sample {
    include!(concat!("audio_samples/", env!("TASK"), "/sample.rs"));
}

#[cfg(not(feature = "bench_spectro_only"))]
// microflow energy builds never read scores, so this const goes unused there
#[cfg_attr(all(feature = "energy_trace", not(feature = "runtime_iree")), allow(dead_code))]
const NUM_CLASSES: usize = audio_sample::NUM_CLASSES;
const N_RUNS: usize = 10;

#[cfg(all(not(feature = "runtime_iree"), not(feature = "kernel_streaming"), not(feature = "bench_spectro_only"), not(feature = "energy_trace")))]
#[microflow::model(
    concat!("models/", env!("TASK"), "/", env!("MODEL"), "/size_", env!("MULT"), "/model.tflite"),
    enable_timing = true,
)]
struct BirdModel;

#[cfg(all(not(feature = "runtime_iree"), feature = "kernel_streaming", not(feature = "bench_spectro_only"), not(feature = "energy_trace")))]
#[microflow::model(
    concat!("models/", env!("TASK"), "/", env!("MODEL"), "/size_", env!("MULT"), "/model.tflite"),
    enable_kernel_streaming = true,
    enable_timing = true,
)]
struct BirdModel;

// energy_trace variants drop `enable_timing`: it puts a per-layer Instant read
// + defmt write *inside* predict_quantized, which would land in the D0 pulse
// (and RTT can block with no probe draining during a capture).
#[cfg(all(not(feature = "runtime_iree"), not(feature = "kernel_streaming"), not(feature = "bench_spectro_only"), feature = "energy_trace"))]
#[microflow::model(
    concat!("models/", env!("TASK"), "/", env!("MODEL"), "/size_", env!("MULT"), "/model.tflite"),
)]
struct BirdModel;

#[cfg(all(not(feature = "runtime_iree"), feature = "kernel_streaming", not(feature = "bench_spectro_only"), feature = "energy_trace"))]
#[microflow::model(
    concat!("models/", env!("TASK"), "/", env!("MODEL"), "/size_", env!("MULT"), "/model.tflite"),
    enable_kernel_streaming = true,
)]
struct BirdModel;

#[cfg(not(feature = "bench_spectro_only"))]
macro_rules! infer {
    ($input:expr) => {{
        #[cfg(not(feature = "runtime_iree"))]
        { BirdModel::predict_quantized($input) }
        #[cfg(feature = "runtime_iree")]
        { model_runtime_iree::predict_quantized::<_, NUM_CLASSES>(&$input) }
    }};
}

#[cfg(all(not(feature = "bench_spectro_only"), not(feature = "energy_trace")))]
macro_rules! class_score {
    ($p:expr, $c:expr) => {{
        #[cfg(not(feature = "runtime_iree"))]
        { $p[(0, $c)] }
        #[cfg(feature = "runtime_iree")]
        { $p[$c] }
    }};
}

#[cfg(all(not(feature = "bench_spectro_only"), not(feature = "energy_trace")))]
#[ariel_os::task(autostart)]
async fn main() {
    info!("tiny-chirp on {}.", ariel_os::buildinfo::BOARD);
    info!("model={} size={}", env!("MODEL"), env!("MULT"));
    info!(
        "audio: sample_rate={} Hz, samples={}",
        audio_sample::SAMPLE_RATE,
        audio_sample::AUDIO.len()
    );

    let audio = audio_sample::AUDIO;

    #[cfg(not(feature = "runtime_iree"))]
    let (q_scale, q_zp) = {
        let (s, z, _) = BirdModel::expose_input();
        (s[0], z[0])
    };
    #[cfg(feature = "runtime_iree")]
    let (q_scale, q_zp) = (model_runtime_iree::INPUT_Q_SCALE, model_runtime_iree::INPUT_Q_ZP);

    // preprocess once before the timed loop so the N_RUNS iterations are inference-only
    let pre0 = Instant::now().as_micros();
    #[cfg(any(feature = "input_mel_default", feature = "input_mel_higher", feature = "input_mel_ultralight"))]
    let input = spectrogram::compute(audio, q_scale, q_zp);
    #[cfg(feature = "input_time")]
    let input = audio_raw::prepare(audio, q_scale, q_zp);
    let pre = Instant::now().as_micros() - pre0;

    let mut prediction = infer!(input);

    let mut total_inf: u64 = 0;
    for i in 0..N_RUNS {
        let t1 = Instant::now().as_micros();
        prediction = infer!(input);
        let t2 = Instant::now().as_micros();
        let inf = t2 - t1;
        total_inf += inf;
        info!("run {} : pre {} us inf {} us", i, pre, inf);
    }
    let avg_pre = pre;
    let avg_inf = total_inf / N_RUNS as u64;

    info!("per-class scores:");
    for c in 0..NUM_CLASSES {
        let s: f32 = class_score!(prediction, c);
        let score_milli = (s * 1000.0 + 0.5) as i32;
        let sign = if score_milli < 0 { "-" } else { "" };
        let abs_v = if score_milli < 0 { -score_milli } else { score_milli };
        info!(
            "  {} : {}{}.{:03}",
            audio_sample::CLASS_NAMES[c],
            sign,
            abs_v / 1000,
            abs_v % 1000
        );
    }
    info!("avg preprocess: {} us ({} runs)", avg_pre, N_RUNS);
    info!("avg inference: {} us ({} runs)", avg_inf, N_RUNS);

    exit(ExitCode::SUCCESS);
}

// Energy-capture variant of the bench above: no timing, no logging — the
// firmware runs silent so nothing but inference sits inside the D0 pulse.
// One PPK2 capture of the D1 window then yields per-inference latency (pulse
// width) and energy (current integral); see src/trace.rs and notebook 08.
#[cfg(all(not(feature = "bench_spectro_only"), feature = "energy_trace"))]
#[ariel_os::task(autostart, peripherals)]
async fn main(p: trace::TracePins) {
    // Busy-wait gaps, not Timer: the sweep must stay await-free so all the heavy
    // work runs in the first poll on the executor stack, like the normal bench —
    // resuming an inference-sized workload after an await hardfaults on this
    // board. The busy gap reads as active current between pulses; energy is
    // integrated inside D0 only, so that's cosmetic.
    use ariel_os::reexports::embassy_time::block_for;
    use ariel_os::time::Duration;

    // Identification banner: compile-time truth about what this binary is,
    // printed once at boot (probe still attached in the session flow, so it
    // lands in probe-rs' output where the grid asserts it against the spec).
    ariel_os::log::info!(
        "energy_trace banner: model={} size={} task={} streaming={} iree={}",
        env!("MODEL"), env!("MULT"), env!("TASK"),
        cfg!(feature = "kernel_streaming"), cfg!(feature = "runtime_iree"),
    );

    let (mut d0, mut d1) = trace::pair(p);
    let audio = audio_sample::AUDIO;

    #[cfg(not(feature = "runtime_iree"))]
    let (q_scale, q_zp) = {
        let (s, z, _) = BirdModel::expose_input();
        (s[0], z[0])
    };
    #[cfg(feature = "runtime_iree")]
    let (q_scale, q_zp) = (model_runtime_iree::INPUT_Q_SCALE, model_runtime_iree::INPUT_Q_ZP);

    #[cfg(any(feature = "input_mel_default", feature = "input_mel_higher", feature = "input_mel_ultralight"))]
    let input = spectrogram::compute(audio, q_scale, q_zp);
    #[cfg(feature = "input_time")]
    let input = audio_raw::prepare(audio, q_scale, q_zp);

    let mut prediction = infer!(input); // warm-up, outside both brackets

    // Sweeps repeat forever: D1 spans each 10-pulse sweep, D0 pulses once per
    // inference (100 ms gap), 1 s idle between sweeps. Looping frees the capture
    // from boot timing — any window ≥ one sweep sees complete D1-gated pulses.
    loop {
        d1.scope(|| {
            for _ in 0..N_RUNS {
                prediction = d0.scope(|| infer!(input));
                core::hint::black_box(&prediction); // outside D0; defeats dead-code elim
                block_for(Duration::from_millis(100));
            }
        });
        block_for(Duration::from_millis(1000));
    }
}

// Energy-capture variant of the spectrogram bench: same D0/D1 bracket and
// looping sweeps as the inference energy main, with preprocess inside D0.
#[cfg(all(feature = "bench_spectro_only", feature = "energy_trace"))]
#[ariel_os::task(autostart, peripherals)]
async fn main(p: trace::TracePins) {
    use ariel_os::reexports::embassy_time::block_for;
    use ariel_os::time::Duration;

    let (mut d0, mut d1) = trace::pair(p);
    ariel_os::log::info!(
        "energy_trace banner: spectro-only task={} frames={} bins={} tiny_chirp={}",
        env!("TASK"),
        spectrogram::FRAMES,
        spectrogram::FFT_BINS,
        cfg!(feature = "tiny_chirp_preprocess"),
    );

    let audio = audio_sample::AUDIO;
    let (q_scale, q_zp) = (1.0f32, 0i8);
    let mut s = spectrogram::compute(audio, q_scale, q_zp); // warm-up, outside brackets

    loop {
        d1.scope(|| {
            for _ in 0..N_RUNS {
                s = d0.scope(|| spectrogram::compute(audio, q_scale, q_zp));
                core::hint::black_box(&s);
                block_for(Duration::from_millis(100));
            }
        });
        block_for(Duration::from_millis(1000));
    }
}

#[cfg(all(feature = "bench_spectro_only", not(feature = "energy_trace")))]
#[ariel_os::task(autostart)]
async fn main() {
    use core::sync::atomic::Ordering;

    info!("tiny-chirp spectrogram bench on {}.", ariel_os::buildinfo::BOARD);
    #[cfg(feature = "tiny_chirp_preprocess")]
    info!("preprocess: tiny_chirp (runtime hann + triangular mel)");
    #[cfg(not(feature = "tiny_chirp_preprocess"))]
    info!("preprocess: fast (precomputed hann + sparse mel)");
    info!(
        "audio: sample_rate={} Hz, samples={}",
        audio_sample::SAMPLE_RATE,
        audio_sample::AUDIO.len()
    );
    info!(
        "mel cfg: N={} H={} FFT_BINS={} FRAMES={}",
        spectrogram::FRAME_LENGTH,
        spectrogram::FRAME_STEP,
        spectrogram::FFT_BINS,
        spectrogram::FRAMES,
    );

    let audio = audio_sample::AUDIO;
    let q_scale = 1.0f32;
    let q_zp = 0i8;

    let _warm = spectrogram::compute(audio, q_scale, q_zp);
    spectrogram::reset_steps();

    let mut total: u64 = 0;
    for i in 0..N_RUNS {
        let t0 = Instant::now().as_micros();
        let s = spectrogram::compute(audio, q_scale, q_zp);
        core::hint::black_box(&s);
        let t1 = Instant::now().as_micros();
        let dt = t1 - t0;
        total += dt;
        info!("run {} : pre {} us", i, dt);
    }
    let n = N_RUNS as u64;
    info!("avg preprocess: {} us ({} runs)", total / n, N_RUNS);
    info!(
        "avg step: framing {} us, fft {} us, mag {} us, mel {} us, logq {} us",
        spectrogram::FRAMING_US.load(Ordering::Relaxed) as u64 / n,
        spectrogram::FFT_US.load(Ordering::Relaxed) as u64 / n,
        spectrogram::MAG_US.load(Ordering::Relaxed) as u64 / n,
        spectrogram::MEL_US.load(Ordering::Relaxed) as u64 / n,
        spectrogram::LOGQ_US.load(Ordering::Relaxed) as u64 / n,
    );
    exit(ExitCode::SUCCESS);
}
