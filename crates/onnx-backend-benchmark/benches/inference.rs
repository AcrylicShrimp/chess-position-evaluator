use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main, Throughput};
use onnx_backend_benchmark::{create_session, generate_random_input, run_inference};
use std::path::PathBuf;

/// Get the model path relative to workspace root.
fn get_model_path() -> PathBuf {
    // CARGO_MANIFEST_DIR points to crates/onnx-backend-benchmark
    // Go up two levels to reach workspace root
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .join("../../models/onnx/small-0.6337-model-best.onnx")
}

/// Benchmark latency for different batch sizes.
/// Primary benchmark for game tree search NPS estimation.
fn bench_latency(c: &mut Criterion) {
    let model_path = get_model_path();
    let mut session = create_session(&model_path).expect("Failed to load model");

    let batch_sizes = [1, 2, 4, 8, 16, 32];

    let mut group = c.benchmark_group("latency");

    for &batch_size in &batch_sizes {
        // Pre-generate input to exclude from timing
        let input = generate_random_input(batch_size);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &input,
            |b, input| {
                b.iter(|| {
                    run_inference(&mut session, input).expect("Inference failed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark throughput with larger batches.
/// Secondary benchmark for batch processing scenarios.
fn bench_throughput(c: &mut Criterion) {
    let model_path = get_model_path();
    let mut session = create_session(&model_path).expect("Failed to load model");

    let batch_sizes = [64, 128, 256, 512];

    let mut group = c.benchmark_group("throughput");

    for &batch_size in &batch_sizes {
        let input = generate_random_input(batch_size);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &input,
            |b, input| {
                b.iter(|| {
                    run_inference(&mut session, input).expect("Inference failed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_latency, bench_throughput);
criterion_main!(benches);
