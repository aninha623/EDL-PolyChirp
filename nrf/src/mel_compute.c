#include <math.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#ifdef MEL_FFT_CPU
#include <arm_math.h>
#endif

#include "sample_input.h"
#include "mel_compute.h"

#define MEL_SAMPLE_RATE 16000.0
#define MEL_MIN_HZ 2500.0
#define MEL_MAX_HZ 8000.0
#define MEL_EPS 1e-6f
#define MEL_PI 3.14159265358979323846f
#define INV_SAMPLE (1.0f / 32768.0f)

_Static_assert((1 << MEL_FFT_LOG2) == MEL_FRAME_LENGTH, "FFT log2 must match frame length");
_Static_assert(MEL_FFT_BINS == MEL_FRAME_LENGTH / 2, "fft bins must be frame length / 2");

static float hann[MEL_FRAME_LENGTH];
static uint16_t mel_fft_bin[2 * MEL_FFT_BINS];
static uint8_t mel_out_bin[2 * MEL_FFT_BINS];
static float mel_weight[2 * MEL_FFT_BINS];
static int mel_filters_len;

static float frame[MEL_FRAME_LENGTH];
static float spec_re[MEL_FRAME_LENGTH];
static float spec_im[MEL_FRAME_LENGTH];
static float mag[MEL_FFT_BINS];
static float mel_energy[MEL_NUM_BINS];

uint64_t mel_framing_cyc, mel_fft_cyc, mel_mag_cyc, mel_mel_cyc, mel_logq_cyc;

#ifdef MEL_FFT_CPU
static arm_rfft_fast_instance_f32 fft_inst;
static float fft_spec[MEL_FRAME_LENGTH];
#else
#define FFT_FULLSCALE (1u << 22)
static const float fft_recover = (float)MEL_FRAME_LENGTH / (float)FFT_FULLSCALE;
static const int32_t int24_cap = (1 << 23) - 1;
static int32_t fft_in[2 * MEL_FRAME_LENGTH];
static int32_t fft_out[2 * MEL_FRAME_LENGTH];
#endif

void mel_reset_steps(void)
{
	mel_framing_cyc = mel_fft_cyc = mel_mag_cyc = mel_mel_cyc = mel_logq_cyc = 0;
}

static int mel_edge(int i, double low_mel, double high_mel)
{
	double frac = (double)i / (double)(MEL_NUM_BINS + 1);
	double mel = low_mel + frac * (high_mel - low_mel);
	double hz = 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
	int bin = (int)((MEL_FRAME_LENGTH + 1.0) * hz / MEL_SAMPLE_RATE);
	return bin >= MEL_FFT_BINS ? MEL_FFT_BINS - 1 : bin;
}

void mel_compute_init(void)
{
#ifdef MEL_FFT_CPU
	arm_rfft_fast_init_f32(&fft_inst, MEL_FRAME_LENGTH);
#endif
	for (int i = 0; i < MEL_FRAME_LENGTH; i++) {
		hann[i] = 0.5f - 0.5f * cosf(2.0f * MEL_PI * i / (MEL_FRAME_LENGTH - 1));
	}

	const double low_mel = 2595.0 * log10(1.0 + MEL_MIN_HZ / 700.0);
	const double high_mel = 2595.0 * log10(1.0 + MEL_MAX_HZ / 700.0);
	int e = 0;
	for (int m = 0; m < MEL_NUM_BINS; m++) {
		int left = mel_edge(m, low_mel, high_mel);
		int center = mel_edge(m + 1, low_mel, high_mel);
		int right = mel_edge(m + 2, low_mel, high_mel);
		for (int k = left; k < center; k++) {
			mel_fft_bin[e] = k;
			mel_out_bin[e] = m;
			mel_weight[e] = (float)(k - left) / (float)(center - left > 0 ? center - left : 1);
			e++;
		}
		for (int k = center; k < right; k++) {
			mel_fft_bin[e] = k;
			mel_out_bin[e] = m;
			mel_weight[e] = (float)(right - k) / (float)(right - center > 0 ? right - center : 1);
			e++;
		}
	}
	mel_filters_len = e;
}

int mel_compute(const int16_t *audio, int8_t *out_int8)
{
	const float in_scale = g_input_quant_scale;
	const int32_t in_zp = g_input_quant_zp;

	for (int f = 0; f < MEL_TARGET_FRAMES; f++) {
		const int base = f * MEL_FRAME_STEP;

		const uint32_t t0 = k_cycle_get_32();
		for (int n = 0; n < MEL_FRAME_LENGTH; n++) {
			int idx = base + n;
			float s = idx < MEL_AUDIO_LEN ? (float)audio[idx] : 0.0f;
			frame[n] = s * hann[n] * INV_SAMPLE;
		}
		const uint32_t t1 = k_cycle_get_32();

#ifdef MEL_FFT_CPU
		arm_rfft_fast_f32(&fft_inst, frame, fft_spec, 0);
		spec_re[0] = fft_spec[0];  // DC; Nyquist (fft_spec[1]) is bin MEL_FFT_BINS, unused
		spec_im[0] = 0.0f;
		for (int k = 1; k < MEL_FFT_BINS; k++) {
			spec_re[k] = fft_spec[2 * k];
			spec_im[k] = fft_spec[2 * k + 1];
		}
#else
		for (int n = 0; n < MEL_FRAME_LENGTH; n++) {
			int32_t v = (int32_t)lroundf(frame[n] * (float)FFT_FULLSCALE);
			if (v > int24_cap) v = int24_cap;
			if (v < -int24_cap) v = -int24_cap;
			fft_in[2 * n] = v;
		}
		int err = mel_fft_axon(fft_in, fft_out);
		if (err) return err;
		for (int k = 0; k < MEL_FFT_BINS; k++) {
			spec_re[k] = (float)fft_out[2 * k] * fft_recover;
			spec_im[k] = (float)fft_out[2 * k + 1] * fft_recover;
		}
#endif
		const uint32_t t2 = k_cycle_get_32();

		for (int k = 0; k < MEL_FFT_BINS; k++) {
			mag[k] = sqrtf(spec_re[k] * spec_re[k] + spec_im[k] * spec_im[k]);
		}
		const uint32_t t3 = k_cycle_get_32();

		for (int m = 0; m < MEL_NUM_BINS; m++) {
			mel_energy[m] = 0.0f;
		}
		for (int i = 0; i < mel_filters_len; i++) {
			mel_energy[mel_out_bin[i]] += mag[mel_fft_bin[i]] * mel_weight[i];
		}
		const uint32_t t4 = k_cycle_get_32();

		int8_t *row = &out_int8[f * MEL_NUM_BINS];
		for (int m = 0; m < MEL_NUM_BINS; m++) {
			int32_t q = (int32_t)lroundf(logf(mel_energy[m] + MEL_EPS) / in_scale) + in_zp;
			if (q < -128) q = -128;
			if (q > 127) q = 127;
			row[m] = (int8_t)q;
		}
		const uint32_t t5 = k_cycle_get_32();

		mel_framing_cyc += (uint32_t)(t1 - t0);
		mel_fft_cyc += (uint32_t)(t2 - t1);
		mel_mag_cyc += (uint32_t)(t3 - t2);
		mel_mel_cyc += (uint32_t)(t4 - t3);
		mel_logq_cyc += (uint32_t)(t5 - t4);
	}
	return 0;
}
