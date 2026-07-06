#include <assert.h>  // static_assert in the axon-generated model header

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <axon/nrf_axon_platform.h>
#ifndef MEL_BENCH_ONLY
#include <drivers/axon/nrf_axon_nn_infer.h>
#include "generated/nrf_axon_model_current_model_.h"
#endif
#include "sample_input.h"
#include "mel_compute.h"

/* ENERGY_TRACE (make trace=1): PPK2 bracket pins from the board overlay —
 * D0 high around each inference, D1 around the sweep. Timing + printk stay
 * as-is (they sit outside D0), so a trace run still prints latencies. */
#ifdef ENERGY_TRACE
#include <zephyr/drivers/gpio.h>

#define TRACE_NODE DT_PATH(zephyr_user)
static const struct gpio_dt_spec trace_d0 = GPIO_DT_SPEC_GET(TRACE_NODE, infer_trace_gpios);
static const struct gpio_dt_spec trace_d1 = GPIO_DT_SPEC_GET(TRACE_NODE, sweep_trace_gpios);

static int trace_init(void)
{
	if (!gpio_is_ready_dt(&trace_d0) || !gpio_is_ready_dt(&trace_d1) ||
	    gpio_pin_configure_dt(&trace_d0, GPIO_OUTPUT_INACTIVE) ||
	    gpio_pin_configure_dt(&trace_d1, GPIO_OUTPUT_INACTIVE)) {
		printk("trace gpio init failed\n");
		return -1;
	}
	return 0;
}

#define TRACE_SET(pin, v) gpio_pin_set_dt(&(pin), (v))
#else
#define TRACE_SET(pin, v)
#endif

#define N_RUNS 10
#define ull unsigned long long

#ifdef MEL_FFT_CPU
#define FFT_BACKEND "cpu"
#else
#define FFT_BACKEND "axon"
#endif

static int8_t mel_buf_int8[MEL_TARGET_FRAMES * MEL_NUM_BINS];

static uint64_t cycles_to_us(uint64_t cycles)
{
	return k_cyc_to_ns_floor64(cycles) / 1000;
}

static void print_avg_steps(void)
{
	printk("avg step: framing %llu us, fft %llu us, mag %llu us, mel %llu us, logq %llu us\n",
	       (ull)(cycles_to_us(mel_framing_cyc) / N_RUNS),
	       (ull)(cycles_to_us(mel_fft_cyc) / N_RUNS),
	       (ull)(cycles_to_us(mel_mag_cyc) / N_RUNS),
	       (ull)(cycles_to_us(mel_mel_cyc) / N_RUNS),
	       (ull)(cycles_to_us(mel_logq_cyc) / N_RUNS));
}

int main(void)
{
	if (nrf_axon_platform_init()) {
		printk("axon init failed\n");
		return -1;
	}

#ifdef ENERGY_TRACE
	if (trace_init()) {
		return -1;
	}
#endif

	const int16_t *audio = g_sample_clip_audio;
	mel_compute_init();

#ifdef MEL_BENCH_ONLY
	printk("frames=%d bins=%d fft=%s (spectro-only)\n",
	       MEL_TARGET_FRAMES, MEL_NUM_BINS, FFT_BACKEND);

	if (mel_compute(audio, mel_buf_int8)) {
		printk("warmup mel_compute failed\n");
		return -1;
	}
	mel_reset_steps();

	uint64_t total_pre = 0;
	TRACE_SET(trace_d1, 1); /* D1 high — whole sweep */
	for (int i = 0; i < N_RUNS; i++) {
		const uint32_t t0 = k_cycle_get_32();
		TRACE_SET(trace_d0, 1); /* D0 high — one preprocess */
		const int mel_err = mel_compute(audio, mel_buf_int8);
		TRACE_SET(trace_d0, 0);
		if (mel_err) {
			printk("run %d: mel_compute failed\n", i);
			return -1;
		}
		const uint64_t pre = cycles_to_us(k_cycle_get_32() - t0);
		total_pre += pre;
		printk("run %d : pre %llu us\n", i, (ull)pre);
#ifdef ENERGY_TRACE
		k_sleep(K_MSEC(100)); /* pulse separation for the PPK2 capture */
#endif
	}
	TRACE_SET(trace_d1, 0);
	printk("avg preprocess: %llu us (%d runs)\n", (ull)(total_pre / N_RUNS), N_RUNS);
	print_avg_steps();
#else
	if (nrf_axon_nn_model_init_vars(&model_current_model)) {
		printk("axon init failed\n");
		return -1;
	}

	const int n_out = model_current_model.output_dimensions.width;
	const float s = (float)model_current_model.output_dequant_mult /
	                (float)(1U << model_current_model.output_dequant_round);
	const int32_t zp = model_current_model.output_dequant_zp;

	printk("model=%s frames=%d bins=%d outputs=%d targets=%d fft=%s\n",
	       model_current_model.model_name, MEL_TARGET_FRAMES, MEL_NUM_BINS,
	       n_out, NUM_TARGET_CLASSES, FFT_BACKEND);

	int32_t out_q[MAX_MODEL_OUTPUTS];
	if (n_out > MAX_MODEL_OUTPUTS) {
		printk("model has %d outputs, only %d allocated\n", n_out, MAX_MODEL_OUTPUTS);
		return -1;
	}

	if (mel_compute(audio, mel_buf_int8)) {
		printk("warmup mel_compute failed\n");
		return -1;
	}
	nrf_axon_nn_model_infer_sync(&model_current_model, mel_buf_int8, (int8_t *)out_q);

	uint64_t total_pre = 0;
	uint64_t total_inf = 0;
	mel_reset_steps();

	TRACE_SET(trace_d1, 1); /* D1 high — whole sweep */
	for (int i = 0; i < N_RUNS; i++) {
		const uint32_t t0 = k_cycle_get_32();
		if (mel_compute(audio, mel_buf_int8)) {
			printk("run %d: mel_compute failed\n", i);
			return -1;
		}
		const uint32_t t1 = k_cycle_get_32();
		TRACE_SET(trace_d0, 1); /* D0 high — inference only */
		nrf_axon_nn_model_infer_sync(&model_current_model, mel_buf_int8, (int8_t *)out_q);
		TRACE_SET(trace_d0, 0);
		const uint32_t t2 = k_cycle_get_32();

		const uint64_t pre = cycles_to_us(t1 - t0);
		const uint64_t inf = cycles_to_us(t2 - t1);
		total_pre += pre;
		total_inf += inf;
		printk("run %d : pre %llu us inf %llu us\n", i, (ull)pre, (ull)inf);
#ifdef ENERGY_TRACE
		k_sleep(K_MSEC(100)); /* idle gap: separates pulses, shows the floor */
#endif
	}
	TRACE_SET(trace_d1, 0);

	printk("per-class scores:\n");
	for (int c = 0; c < NUM_TARGET_CLASSES; c++) {
		const float v = (float)(out_q[c] - zp) * s;
		printk("  %s : %.3f\n", g_class_names[c], (double)v);
	}

	printk("avg preprocess: %llu us (%d runs)\n", (ull)(total_pre / N_RUNS), N_RUNS);
	print_avg_steps();
	printk("avg inference: %llu us (%d runs)\n", (ull)(total_inf / N_RUNS), N_RUNS);
#endif

	printk("DONE\n");
	return 0;
}
