#pragma once

#include <stdint.h>

#include "generated_data/sample_input_meta.h"  // MEL_FFT_LOG2, frame/bin sizes

void mel_compute_init(void);
int mel_compute(const int16_t *audio, int8_t *out_int8);
int mel_fft_axon(const int32_t *complex_in, int32_t *complex_out);

extern uint64_t mel_framing_cyc, mel_fft_cyc, mel_mag_cyc, mel_mel_cyc, mel_logq_cyc;
void mel_reset_steps(void);
