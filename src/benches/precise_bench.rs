use std::time::Duration;
use criterion::{Bencher, criterion_group, criterion_main, Criterion};
use precise_rs::Precise;

fn running_bench(b: &mut Bencher) {
    let mut reader = hound::WavReader::open("test_data/test.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
    let mut precise = Precise::new("test_data/hey_mycroft.tflite").unwrap();
    b.iter(|| precise.update(&samples).unwrap())
}

pub fn precise_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("Precise");
    g.sample_size(20);
    g.measurement_time(Duration::new(20,0));
    g.warm_up_time(Duration::new(10,0));
    
    g.bench_function("creation", |b| b.iter(|| Precise::new("test_data/hey_mycroft.tflite").unwrap()));
    
    g.bench_function("running", running_bench);
}

criterion_group!(benches, precise_benchmark);
criterion_main!(benches);