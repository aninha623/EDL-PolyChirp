# PolyChirp

Multi-species bird-call classification for microcontrollers. Extends [TinyChirp](https://github.com/TinyPART/TinyChirp) ([paper](https://arxiv.org/abs/2407.21453)) to more species and a place-based multi-label task, trained in Python (Keras/TF + PTQ) and deployed in Rust via [MicroFlow](https://github.com/matteocarnelos/microflow-rs) ([paper](https://arxiv.org/pdf/2409.19432)) on Ariel OS.

## Layout

- `building/`: Python — datasets, training, post-training quantization, eval
- `src/`, `Cargo.toml`, `build.rs`, `laze-project.yml`: Rust MicroFlow firmware
- `nrf/`, `Makefile`: nRF54 Axon (NPU) firmware (Zephyr / nRF Connect SDK)
- `datasets/<collection>/{training,validation,testing}/<Class>/`
- `models/<collection>/<model>/…` and `results/` are generated locally (git-ignored)

## Setup

- `cd building && uv sync` ([uv](https://docs.astral.sh/uv/))
- `building/.env`:
  - `XC_API_KEY=...` ([xeno-canto](https://xeno-canto.org/account/api))
  - `EBIRD_API_KEY=...` ([ebird](https://ebird.org/api/keygen), required)

## Pipelines

Each pipeline is driven by its own notebooks; instructions live there.

- `building/geographic_task/`: place + target species, multi-label model, scored on long recordings
- `building/geographic_scale/`: scaling sweep rooted in a place

Model architectures live in `building/models/` (see `building.models.EXTENDED_LIST`): `cnn1d`, `transformer`, `mel_cnn`, `mel_cnn_new`, `sincnet` family, `leaf`, `wrennet`, `ds_cnn`, `mobilenetv2`, `mcunet_in0`.

## Deploy

Default backend is `microflow-rs` (compile-time TFLite → Rust). Pick a board, model, task, and size, then build and flash:

```bash
laze build -b {board-id} -s model-{m} -s task-{t} -s size-{s} run
```

Binary size:

```bash
runtime=build/bin/{board-id}/cargo/thumbv8m.main-none-eabihf/release/tiny-chirp-microflow
arm-none-eabi-size "$runtime"
```

### nRF54 Axon NPU

`nrf/` is a separate Zephyr / nRF Connect SDK app that runs mel + inference on the nRF54 Axon NPU; `make` drives it (`make help`). `make build`/`flash`/`listen` operate on checked-in sources; the axon-generated model assets under `nrf/generated/` are produced by the NPU codegen and are not tracked, so bring your own trained model.
