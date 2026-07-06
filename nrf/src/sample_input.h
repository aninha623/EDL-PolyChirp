#pragma once

#include <stdint.h>
#include <stddef.h>

#include "generated_data/sample_input_meta.h"

extern const int16_t g_sample_clip_audio[];
extern const char *const g_class_names[NUM_TARGET_CLASSES];

extern const float g_input_quant_scale;
extern const int32_t g_input_quant_zp;
