use criterion::{criterion_group, criterion_main, Criterion};
use precise_rs::Precise;

pub fn precise_benchmark(c: &mut Criterion) {
    c.bench_function("creation", |b| b.iter(|| Precise::new("test_data/hey_mycroft.tflite").unwrap()))
        .sample_size(20);

    let mut reader = hound::WavReader::open("test_data/test.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
    let mut precise = Precise::new("test_data/hey_mycroft.tflite").unwrap();
    
    c.bench_function("running", |b| b.iter(|| precise.update(&samples).unwrap()))
        .sample_size(20);
}

criterion_group!(benches, precise_benchmark);
criterion_main!(benches);