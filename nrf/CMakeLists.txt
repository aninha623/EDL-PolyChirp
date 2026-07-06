cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(multi_chirp_npu)

# generated/ + generated_data/ sit next to this file; resolve includes + .incbin from here
target_compile_options(app PRIVATE -Wa,-I${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} src)

target_sources(app PRIVATE
    src/main.c
    src/sample_input.c
    src/mel_compute.c
    src/mel_fft_axon.c
)

if(DEFINED ENV{MEL_FFT_CPU})
    target_compile_definitions(app PRIVATE MEL_FFT_CPU)
endif()
if(DEFINED ENV{MEL_BENCH_ONLY})
    target_compile_definitions(app PRIVATE MEL_BENCH_ONLY)
endif()
if(DEFINED ENV{ENERGY_TRACE})
    target_compile_definitions(app PRIVATE ENERGY_TRACE)
endif()
