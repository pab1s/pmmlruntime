//! Criterion benches for PMML scoring — `tree_iris_single` / `tree_iris_batch_1k_*`.
//!
//!mirrors `BENCHMARK.md` §3 (tiny-batch fallback) and §5 (Arrow wins at 100k). `criterion` `black_box`
//! prevents elision; batch `1k` compares sequential `run` loop vs `run_batch` (`CpuBatched` `par_chunks(256)`).
//! `load_iris()` reads `bench/pmml/DecisionTreeIris.pmml` relative to `CARGO_MANIFEST_DIR`; it panics if missing
//! (bench harness only, not library).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

/// Load Iris `TreeModel` session for benches (cold path, unwraps for bench harness).
///
/// Reads `CARGO_MANIFEST_DIR/../../bench/pmml/DecisionTreeIris.pmml` and `Session::from_bytes` with `default()` (`CpuSerial`).
/// Panics if file missing (so bench setup fails fast rather than silent skip).
fn load_iris() -> Session {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest).join("../../bench/pmml/DecisionTreeIris.pmml");
    let bytes = std::fs::read(path).unwrap();
    let env = PmmlEnv::new();
    Session::from_bytes(&env, &bytes, SessionOptions::default()).unwrap()
}

/// Bench single-row `run` (`Petal.Length=1.4`, `Petal.Width=0.2` → `setosa`).
///
/// Measures hot-path `with_value_buffer` stack `64` + `eval_tree` (~402 ns). `black_box` on inputs/outputs
/// prevents optimizer from eliding `run`.
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

/// Bench `1k` rows sequential via `for m in &batch { run(m.clone()) }`.
///
/// Baseline without `rayon`; `batch` is `Vec<HashMap>` sized `1000` with synthetic `1.0 + (i%5)`.
/// Used to show `815 µs` sequential vs ` <400 µs` batched in `BENCHMARK.md`.
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

/// Bench `1k` rows parallel via `Session::run_batch` / `run_batch_ref` (`CpuBatched`).
///
/// Creates a `CpuBatched` `Session` (same Iris PMML, `ExecutionProviderKind::CpuBatched`) and a `1k` `Vec<HashMap>` batch.
/// Benches `run_batch(batch.clone())` (clone per iter) and `run_batch_ref(&batch)` (no clone, preserves bench batch).
/// Expect `<400 µs` (~2× vs sequential) because `n=1000` shards into `par_chunks(256)`.
///
/// # Panics
///
/// Panics if Iris PMML missing (bench setup).
fn bench_batch_1k_parallel(c: &mut Criterion) {
    // Batched provider with rayon par_iter chunk 1k — expects < 400 µs (2× vs 815 µs sequential)
    let env = PmmlEnv::new();
    let opts = SessionOptions::default()
        .execution_provider(pmmlruntime::session::ExecutionProviderKind::CpuBatched);
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
