use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_example(c: &mut Criterion) {
    c.bench_function("example_bench", |b| b.iter(|| black_box(42)));
}

criterion_group!(benches, bench_example);
criterion_main!(benches);
