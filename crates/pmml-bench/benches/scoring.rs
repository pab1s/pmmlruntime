use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmml_core::Value;
use pmml_session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

fn load_iris() -> Session {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest).join("../../bench/pmml/DecisionTreeIris.pmml");
    let bytes = std::fs::read(path).unwrap();
    let env = PmmlEnv::new();
    Session::from_bytes(&env, &bytes, SessionOptions::default()).unwrap()
}

fn bench_single(c: &mut Criterion) {
    let sess = load_iris();
    c.bench_function("tree_iris_single", |b| {
        b.iter(|| {
            let mut m = HashMap::new();
            m.insert(
                "Petal.Length".to_string(),
                Value::Continuous(black_box(1.4)),
            );
            m.insert("Petal.Width".to_string(), Value::Continuous(black_box(0.2)));
            let out = sess.run(m).unwrap();
            black_box(out);
        })
    });
}

fn bench_batch_1k(c: &mut Criterion) {
    let sess = load_iris();
    let batch: Vec<HashMap<String, Value>> = (0..1000)
        .map(|i| {
            let mut m = HashMap::new();
            let v = 1.0 + (i as f64 % 5.0);
            m.insert("Petal.Length".to_string(), Value::Continuous(v));
            m.insert("Petal.Width".to_string(), Value::Continuous(v * 0.5));
            m
        })
        .collect();
    c.bench_function("tree_iris_batch_1k_sequential", |b| {
        b.iter(|| {
            for m in &batch {
                let out = sess.run(m.clone()).unwrap();
                black_box(out);
            }
        })
    });
}

fn bench_batch_1k_parallel(c: &mut Criterion) {
    // Batched provider with rayon par_iter chunk 1k — expects < 400 µs (2× vs 815 µs sequential)
    let env = PmmlEnv::new();
    let opts = SessionOptions::default()
        .execution_provider(pmml_session::ExecutionProviderKind::CpuBatched);
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest).join("../../bench/pmml/DecisionTreeIris.pmml");
    let bytes = std::fs::read(path).unwrap();
    let sess = Session::from_bytes(&env, &bytes, opts).unwrap();
    let batch: Vec<HashMap<String, Value>> = (0..1000)
        .map(|i| {
            let mut m = HashMap::new();
            let v = 1.0 + (i as f64 % 5.0);
            m.insert("Petal.Length".to_string(), Value::Continuous(v));
            m.insert("Petal.Width".to_string(), Value::Continuous(v * 0.5));
            m
        })
        .collect();
    c.bench_function("tree_iris_batch_1k_parallel", |b| {
        b.iter(|| {
            let out = sess.run_batch(batch.clone()).unwrap();
            black_box(out);
        })
    });
    // Also bench run_batch_ref (no clone per iter in criterion, preserves batch)
    c.bench_function("tree_iris_batch_1k_parallel_ref", |b| {
        b.iter(|| {
            let out = sess.run_batch_ref(&batch).unwrap();
            black_box(out);
        })
    });
}

criterion_group!(
    benches,
    bench_single,
    bench_batch_1k,
    bench_batch_1k_parallel
);
criterion_main!(benches);
