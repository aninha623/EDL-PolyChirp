# nRF Axon (NPU) firmware driver. `make` for help.
NCS_VERSION ?= v3.3.0
BOARD       ?= nrf54lm20dk/nrf54lm20b/cpuapp
BUILD_DIR   ?= nrf/build_ncs
ZEPHYR_BASE ?= $(HOME)/ncs/zephyr
SERIAL      ?= /dev/ttyACM1
LOG_FILE    ?= logs/listen.log
PYTHON      ?= .venv/bin/python

# run_model knobs
model       ?= mel_ultralight
task        ?= detection_49_2
m           ?= 1
preprocess  ?= npu

ifeq ($(preprocess),cpu)
BUILD_ENV += MEL_FFT_CPU=1
endif
ifeq ($(bench_only),1)
BUILD_ENV += MEL_BENCH_ONLY=1
endif
ifeq ($(trace),1)
BUILD_ENV += ENERGY_TRACE=1
BUILD_ARGS = -- -DEXTRA_CONF_FILE=trace_energy.conf
endif

LAUNCH = nrfutil toolchain-manager launch --ncs-version $(NCS_VERSION) -- bash -c
WEST   = ZEPHYR_BASE=$(ZEPHYR_BASE) west

.PHONY: help build flash listen run_model clean menuconfig

help:
	@echo "make run_model     -- codegen + build + flash one model (run 'make listen' separately)"
	@echo "                      model=$(model) task=$(task) m=$(m) preprocess=$(preprocess) (cpu|npu)"
	@echo "make build         -- build the firmware (preprocess=$(preprocess))"
	@echo "make flash         -- flash the firmware to BOARD=$(BOARD)"
	@echo "make listen        -- capture serial output to $(LOG_FILE)"
	@echo "make clean         -- remove $(BUILD_DIR)"
	@echo "make menuconfig    -- Zephyr menuconfig"

# codegen + build + flash; run `make listen` separately to capture output
run_model:
	$(PYTHON) -m building.benching.npu_inference --task $(task) --model $(model) --mult $(m)
	$(MAKE) build preprocess=$(preprocess)
	$(MAKE) flash

build:
	$(LAUNCH) '$(BUILD_ENV) $(WEST) build -b $(BOARD) -p always -d $(BUILD_DIR) nrf $(BUILD_ARGS)'

flash:
	$(LAUNCH) '$(WEST) flash --skip-rebuild -d $(BUILD_DIR)'

listen:
	@mkdir -p $(dir $(LOG_FILE))
	@: > $(LOG_FILE)
	@echo "Logging $(SERIAL) -> $(LOG_FILE) (Ctrl-A Ctrl-X to exit)"
	picocom -b 115200 --quiet --logfile $(LOG_FILE) $(SERIAL)

menuconfig:
	$(LAUNCH) '$(WEST) build -t menuconfig -d $(BUILD_DIR)'

clean:
	rm -rf $(BUILD_DIR)
