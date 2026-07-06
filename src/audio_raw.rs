use microflow::buffer::Buffer4D;
use microflow::quantize::quantize;
use nalgebra::SMatrix;

pub const AUDIO_LEN: usize = 48_000;

type AudioImage = SMatrix<[i8; 1], AUDIO_LEN, 1>;

pub fn prepare(audio: &[f32], scale: f32, zero_point: i8) -> Buffer4D<i8, 1, AUDIO_LEN, 1, 1> {
    let mut out: AudioImage = SMatrix::from_element([0i8; 1]);
    let mut i = 0usize;
    while i < AUDIO_LEN {
        let sample = if i < audio.len() { audio[i] } else { 0.0 };
        out[(i, 0)] = [quantize(sample, scale, zero_point)];
        i += 1;
    }
    [out]
}
