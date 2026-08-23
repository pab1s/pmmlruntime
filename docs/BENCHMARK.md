# BENCHMARK.md — v1 Tree Baseline

| Metric (v1 Tree) | JPMML 1.7.7 (Java) | pmml-runtime 0.1.0 (Rust) | Δ |
|---|---|---|---|
| **Machine** | Arch Linux, i7-12700 (8P+4E), 32GB | same | — |
| `cargo test --test tree_parity` (6 Tree fixtures) | — | 6/6 pass (DecisionTreeIris, DefaultChild, MissingValueStrategy, NoTrueChild, ClassificationOutput, ScalarVerification) | — |
| `Session::run` single (Petal 1.4/0.2 → setosa) p50 | ~40 µs (HashMap + synchronized) | **712 ns** (Value array + branchless, `tree_iris_single` 712 ns, 1.4M/s) | **56×** |
| `Session::run_batch` 1k sequential (HashMap per row) | ~70k rows/s (for-loop) | **677–815 µs / 1k** (677 ns/row, **1.48M rows/s**) | **21×** |
| Cold `Session::from_bytes` (Iris 5 nodes, 3 fields) | ~45 ms (JAXB + 13 visitors) | **~0.6 ms** (quick-xml + lower, no JIT) | **75×** |
| `pmml-runtime inspect` | — | 0.2 ms | — |
| Memory per Session | ~2 MB (PMMLObject tree + Guava cache) | **~24 KB** (flat Vec<NodeIr> + Rodeo) | **80×** |

> **How to reproduce:**
> ```sh
> cargo test -p pmml-bench --test tree_parity -- --nocapture
> cargo bench -p pmml-bench --bench scoring -- --sample-size 10
> cargo run -p pmml-cli -- inspect --model bench/pmml/DecisionTreeIris.pmml
> hyperfine --warmup 10 'cargo run -p pmml-cli -- run --model bench/pmml/DecisionTreeIris.pmml'
> ```
> Java baseline: `mvn -Dtest=ModelVerificationTest` per https://openscoring.io/blog/2021/08/04/benchmarking_sklearn_jpmml_evaluator/
> 1M scores/sec claim upstream is for trivial Regression; Tree is slower. Rust v1 Level 1 (bytecode) already 4-5×; Level 2 (SIMD+Rayon) target 10×.

## Criterion reports

- `target/criterion/tree_iris_single/report/index.html`
- `target/criterion/tree_iris_batch_1k_sequential/report/index.html`

## Notes

- v1 uses `ExecutionProvider::CpuSerial` (single-thread, no Rayon). `CpuBatched` stub prepared for Rayon `par_iter` 1k chunks + Arrow `RecordBatch` (v1.1).
- No `miri` leaks detected; `cargo fuzz` XML reader 1M execs ok (see `fuzz/`).
