use ariel_os::time::Instant;
use core::sync::atomic::{AtomicU32, Ordering};
use libm::{logf, sqrtf};
#[cfg(feature = "tiny_chirp_preprocess")]
use libm::{cosf, log10f, powf};
use microflow::buffer::Buffer4D;
use microflow::quantize::quantize;
use nalgebra::SMatrix;

#[cfg(not(feature = "input_mel_ultralight"))]
pub const FRAME_LENGTH: usize = 1024;
#[cfg(not(feature = "input_mel_ultralight"))]
pub const FRAME_STEP: usize = 256;
#[cfg(not(feature = "input_mel_ultralight"))]
pub const FFT_BINS: usize = 512;
#[cfg(not(feature = "input_mel_ultralight"))]
pub const FRAMES: usize = 184;

#[cfg(feature = "input_mel_ultralight")]
pub const FRAME_LENGTH: usize = 512;
#[cfg(feature = "input_mel_ultralight")]
pub const FRAME_STEP: usize = 480;
#[cfg(feature = "input_mel_ultralight")]
pub const FFT_BINS: usize = 256;
#[cfg(feature = "input_mel_ultralight")]
pub const FRAMES: usize = 99;

include!(concat!(env!("OUT_DIR"), "/mel_bins.rs"));
#[cfg(not(feature = "tiny_chirp_preprocess"))]
include!(concat!(env!("OUT_DIR"), "/hann_table.rs"));
#[cfg(not(feature = "tiny_chirp_preprocess"))]
include!(concat!(env!("OUT_DIR"), "/mel_table.rs"));

#[cfg(feature = "tiny_chirp_preprocess")]
const SAMPLE_RATE: f32 = 16_000.0;
#[cfg(all(feature = "tiny_chirp_preprocess", feature = "input_mel_default"))]
const MIN_HZ: f32 = 80.0;
#[cfg(all(feature = "tiny_chirp_preprocess", feature = "input_mel_default"))]
const MAX_HZ: f32 = 8000.0;
#[cfg(all(
    feature = "tiny_chirp_preprocess",
    any(feature = "input_mel_higher", feature = "input_mel_ultralight"),
))]
const MIN_HZ: f32 = 2500.0;
#[cfg(all(
    feature = "tiny_chirp_preprocess",
    any(feature = "input_mel_higher", feature = "input_mel_ultralight"),
))]
const MAX_HZ: f32 = 8000.0;

const EPS: f32 = 1e-6;

pub static FRAMING_US: AtomicU32 = AtomicU32::new(0);
pub static FFT_US: AtomicU32 = AtomicU32::new(0);
pub static MAG_US: AtomicU32 = AtomicU32::new(0);
pub static MEL_US: AtomicU32 = AtomicU32::new(0);
pub static LOGQ_US: AtomicU32 = AtomicU32::new(0);

pub fn reset_steps() {
    FRAMING_US.store(0, Ordering::Relaxed);
    FFT_US.store(0, Ordering::Relaxed);
    MAG_US.store(0, Ordering::Relaxed);
    MEL_US.store(0, Ordering::Relaxed);
    LOGQ_US.store(0, Ordering::Relaxed);
}

#[cfg(feature = "tiny_chirp_preprocess")]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * log10f(1.0 + hz / 700.0)
}

#[cfg(feature = "tiny_chirp_preprocess")]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (powf(10.0, mel / 2595.0) - 1.0)
}

#[cfg(feature = "tiny_chirp_preprocess")]
fn hann_window(i: usize) -> f32 {
    let phase = (2.0 * core::f32::consts::PI * i as f32) / (FRAME_LENGTH as f32 - 1.0);
    0.5 - 0.5 * cosf(phase)
}

#[cfg(feature = "tiny_chirp_preprocess")]
fn mel_bin_edge(i: usize, low_mel: f32, high_mel: f32) -> usize {
    let frac = i as f32 / (MEL_BINS + 1) as f32;
    let mel = low_mel + frac * (high_mel - low_mel);
    let hz = mel_to_hz(mel);
    let bin = ((FRAME_LENGTH as f32 + 1.0) * hz / SAMPLE_RATE) as usize;
    if bin >= FFT_BINS { FFT_BINS - 1 } else { bin }
}

pub fn compute(
    audio: &[f32],
    scale: f32,
    zero_point: i8,
) -> Buffer4D<i8, 1, FRAMES, MEL_BINS, 1> {
    let mut out: SMatrix<[i8; 1], FRAMES, MEL_BINS> = SMatrix::from_element([0i8; 1]);

    #[cfg(feature = "tiny_chirp_preprocess")]
    let low_mel = hz_to_mel(MIN_HZ);
    #[cfg(feature = "tiny_chirp_preprocess")]
    let high_mel = hz_to_mel(MAX_HZ);

    let mut framing_us: u32 = 0;
    let mut fft_us: u32 = 0;
    let mut mag_us: u32 = 0;
    let mut mel_us: u32 = 0;
    let mut logq_us: u32 = 0;

    let mut frame_idx = 0usize;
    while frame_idx < FRAMES {
        let mut frame = [0f32; FRAME_LENGTH];
        let base = frame_idx * FRAME_STEP;

        let t0 = Instant::now().as_micros();
        let mut n = 0usize;
        while n < FRAME_LENGTH {
            let sample = if base + n < audio.len() { audio[base + n] } else { 0.0 };
            #[cfg(not(feature = "tiny_chirp_preprocess"))]
            { frame[n] = sample * HANN[n]; }
            #[cfg(feature = "tiny_chirp_preprocess")]
            { frame[n] = sample * hann_window(n); }
            n += 1;
        }
        let t1 = Instant::now().as_micros();

        #[cfg(not(feature = "input_mel_ultralight"))]
        let spectrum = microfft::real::rfft_1024(&mut frame);
        #[cfg(feature = "input_mel_ultralight")]
        let spectrum = microfft::real::rfft_512(&mut frame);
        let t2 = Instant::now().as_micros();

        let mut magnitudes = [0f32; FFT_BINS];
        let mut b = 0usize;
        while b < FFT_BINS {
            let c = spectrum[b];
            magnitudes[b] = sqrtf(c.re * c.re + c.im * c.im);
            b += 1;
        }
        let t3 = Instant::now().as_micros();

        let mut mel_energies = [0f32; MEL_BINS];
        #[cfg(not(feature = "tiny_chirp_preprocess"))]
        {
            let mut e = 0usize;
            while e < MEL_FILTERS_LEN {
                let (fft_bin, mel_bin, w) = MEL_FILTERS[e];
                mel_energies[mel_bin as usize] += magnitudes[fft_bin as usize] * w;
                e += 1;
            }
        }
        #[cfg(feature = "tiny_chirp_preprocess")]
        {
            let mut m = 0usize;
            while m < MEL_BINS {
                let left = mel_bin_edge(m, low_mel, high_mel);
                let center = mel_bin_edge(m + 1, low_mel, high_mel);
                let right = mel_bin_edge(m + 2, low_mel, high_mel);
                let mut energy = 0.0f32;
                let mut k = left;
                while k < center {
                    let w = (k - left) as f32 / (center - left).max(1) as f32;
                    energy += magnitudes[k] * w;
                    k += 1;
                }
                k = center;
                while k < right {
                    let w = (right - k) as f32 / (right - center).max(1) as f32;
                    energy += magnitudes[k] * w;
                    k += 1;
                }
                mel_energies[m] = energy;
                m += 1;
            }
        }
        let t4 = Instant::now().as_micros();

        let mut m = 0usize;
        while m < MEL_BINS {
            out[(frame_idx, m)] = [quantize(logf(mel_energies[m] + EPS), scale, zero_point)];
            m += 1;
        }
        let t5 = Instant::now().as_micros();

        framing_us = framing_us.wrapping_add((t1 - t0) as u32);
        fft_us = fft_us.wrapping_add((t2 - t1) as u32);
        mag_us = mag_us.wrapping_add((t3 - t2) as u32);
        mel_us = mel_us.wrapping_add((t4 - t3) as u32);
        logq_us = logq_us.wrapping_add((t5 - t4) as u32);

        frame_idx += 1;
    }

    FRAMING_US.fetch_add(framing_us, Ordering::Relaxed);
    FFT_US.fetch_add(fft_us, Ordering::Relaxed);
    MAG_US.fetch_add(mag_us, Ordering::Relaxed);
    MEL_US.fetch_add(mel_us, Ordering::Relaxed);
    LOGQ_US.fetch_add(logq_us, Ordering::Relaxed);

    [out]
}
