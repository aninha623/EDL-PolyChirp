//! GPIO bracket for external (PPK2) energy + latency measurement.
//!
//! `Trace::scope(|| block)` holds a pin high around `block`, so one D0 pulse
//! per inference carries both its latency (pulse width) and energy (current
//! integral over the pulse). Compiled only with the `energy_trace` feature
//! (laze: `-s energy-trace`); normal builds never see this module.
//!
//! nRF54LM20 DK pins: D0 = P1.12, D1 = P1.13 — free pads on the GPIO_P1
//! header, same wiring as demo-blinky and the nrf/ C firmware.

use ariel_os::gpio::{Level, Output};
use ariel_os::hal::peripherals; // referenced by define_peripherals! below

#[cfg(context = "nrf54lm20dk-app")]
ariel_os::hal::define_peripherals!(TracePins {
    d0: P1_12,
    d1: P1_13,
});

// Pico 2 (and W): D0/D1 = PIN_2/PIN_3 (physical header pins 4/5), same
// wiring as demo-blinky. PPK2 runs in SOURCE mode on this board.
#[cfg(any(context = "rpi-pico2", context = "rpi-pico2-w"))]
ariel_os::hal::define_peripherals!(TracePins {
    d0: PIN_2,
    d1: PIN_3,
});
// Other boards: add a cfg(context = "...") block with two free pins.

/// A GPIO pin held high for the duration of a measured block.
pub struct Trace<'a> {
    pin: Output<'a>,
}

impl<'a> Trace<'a> {
    fn new(pin: Output<'a>) -> Self {
        Self { pin }
    }

    /// Runs `f` with the pin high — raised before, lowered after.
    pub fn scope<R>(&mut self, f: impl FnOnce() -> R) -> R {
        self.pin.set_high();
        let r = f();
        self.pin.set_low();
        r
    }
}

/// The two bracket pins: D0 = per-inference pulse, D1 = sweep marker.
pub fn pair(p: TracePins) -> (Trace<'static>, Trace<'static>) {
    (
        Trace::new(Output::new(p.d0, Level::Low)),
        Trace::new(Output::new(p.d1, Level::Low)),
    )
}
