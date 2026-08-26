//! `pmml-bench` — criterion benches and large-batch trials.
//!
//! This crate holds `benches/scoring.rs` (`criterion` `bench_single`, `bench_batch_1k`, `bench_batch_1k_parallel`)
//! and `src/bin/large_trial.rs` (10k / 100k / 1M / 10M throughput trial with `make_hash_batch` / `make_arrow_batch`).
//! It is not a library for consumers; `placeholder()` exists so `cargo doc` links to this crate.
//!
//! # What belongs here
//!
//! - `benches/scoring.rs` — `criterion_group!` benches for `tree_iris_single`, `tree_iris_batch_1k_sequential`, `tree_iris_batch_1k_parallel`, `tree_iris_batch_1k_parallel_ref`.
//! - `src/bin/large_trial.rs` — `run_trial(size)` for 10k … 10M, with `time()` / `fmt_thr()` helpers and SIMD regression check.
//! - `tests/tree_parity.rs` — parity vs PMML (not in `src`).
//!
//! # Performance targets
//!
//! Single `run` ~402 ns, `run_batch_arrow` ~61 ns/row at 100k (Arrow `CpuBatched` `par_chunks(256)`).
//! `large_trial` prints `ms total | rows/sec | ns/row` for each provider (`CpuSerial` vs `CpuBatched`) and path (`HashMap` vs `RecordBatch`).

/// Placeholder to keep `pmml_bench` crate non-empty for `cargo doc`.
///
/// Not used; benches are in `benches/` and binary is `src/bin/large_trial.rs`.
pub fn placeholder() {}
