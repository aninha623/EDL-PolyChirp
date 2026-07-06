#include "sample_input.h"
#include "generated_data/class_names.h"

#define INCBIN(symbol, file, align) \
	__asm__(".section .rodata." #symbol ",\"a\",%progbits\n" \
		".balign " #align "\n" \
		".global " #symbol "\n" \
		".type " #symbol ",%object\n" \
		#symbol ":\n" \
		".incbin \"" file "\"\n" \
		".size " #symbol ", . - " #symbol "\n" \
		".previous\n")

INCBIN(g_sample_clip_audio, "generated_data/sample_audio.bin", 2);

const float g_input_quant_scale = MEL_INPUT_QUANT_SCALE;
const int32_t g_input_quant_zp = MEL_INPUT_QUANT_ZP;

const char *const g_class_names[NUM_TARGET_CLASSES] = CLASS_NAMES_INIT;
