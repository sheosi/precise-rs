
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use precise_rs::Precise;

pub fn precise_benchmark(c: &mut Criterion) {
    c.bench_function("creation", |b| b.iter(|| Precise::new("hey_mycroft.tflite").unwrap()));

    let mut reader = hound::WavReader::open("test.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
    let mut precise = Precise::new("hey_mycroft.tflite").unwrap();
    
    c.bench_function("running", |b| b.iter(|| precise.update(&samples).unwrap()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);