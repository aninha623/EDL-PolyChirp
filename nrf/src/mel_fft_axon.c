#include <stdbool.h>
#include <stdint.h>

#include <drivers/axon/nrf_axon_dsp_intrinsics.h>
#include <drivers/axon/nrf_axon_driver.h>

#include "mel_compute.h"

int mel_fft_axon(const int32_t *complex_in, int32_t *complex_out)
{
	return axon_fft_24(complex_in, complex_out, MEL_FFT_LOG2, true, true,
			   NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
}
