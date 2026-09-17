# Performance: Cold, Hot, and Batch

> **Info:** Looking for parity guarantees? See [Correctness & Fixture Parity](./correctness.md). Looking for deploy gates? See [Validation & Threshold Gating](./validation.md).

pmmlruntime states a win only when Rust's worst run beats Java's best run across five fresh processes. Every figure comes from `bench/BENCHMARK.md`, measured on an i7-12650H.

## The claim rule

Overlapping intervals mean no claim at all, which is why the tables below list three models. A cold load is `Session::from_bytes`, meaning parse, verify, and lower. A hot single row is `sess.run(&HashMap as &dyn Batch)` after 10k warmup rows. A batch is `sess.run(&Vec<HashMap> as &dyn Batch)`, with `rayon` sharding above 256 rows.

## Reproduce the numbers

Build the Rust harness in release mode, then run it five times so every run sees a fresh process:

```bash
cargo build --release --example bench_real
for i in 1 2 3 4 5; do
  cargo run --release --example bench_real -- bench/pmml/DecisionTreeIris.pmml --iterations 2000
done
```

The harness draws one synthetic value per active field from that field's declared domain. It scores `DecisionTreeIris.pmml` (2.8 KB tree, 2 active fields), `GradientBoosterTest.pmml` (2.6 KB `MiningModel` over 3 stumps), and `AlternateBinaryTargetCategoryTest.pmml` (2.1 KB SVM with an RBF kernel). Cold measures 20 loads, hot measures 10k warmup rows plus 2k measured rows, and batch measures 5k batches of 100 rows, 1k batches of 1k rows, and 200 batches of 10k rows. The Java side runs the same fixtures through JPMML 1.7.7, interpreted and transpiled. `bench/BENCHMARK.md` holds that harness and its raw output.

Keep a claim only when `Rust(mean+std) < Java(mean-std)`. Hot `DecisionTreeIris` gives Rust `463 ± 87 ns` against Java `4562 ± 127 ns`, so Rust's worst run of 550 ns still beats Java's best of 4435 ns and the 9.9× claim holds.

## Hot single row

| Model | pmmlruntime | JPMML eval | Speedup | JPMML transp | Speedup |
| --- | --- | --- | --- | --- | --- |
| `DecisionTreeIris` | **463 ± 87 ns** | 4562 ± 127 ns | **9.9×** | 5417 ± 1183 ns | 11.7× |
| `GradientBoosterTest` | **438 ± 99 ns** | 12267 ± 958 ns | **28.0×** | N/A, not transpilable | N/A |
| `AlternateBinaryTargetCategoryTest` | **716 ± 155 ns** | 3535 ± 443 ns | **4.9×** | 3246 ± 190 ns | 4.5× |

The ensemble row is the widest gap: `GradientBoosterTest` scores in 438 ns while the transpiler refuses the file, because its strict `OutputField@dataType` check rejects PMML the evaluator accepts.

## Cold load

| Model | pmmlruntime | JPMML eval | Speedup | JPMML transp | Speedup |
| --- | --- | --- | --- | --- | --- |
| `DecisionTreeIris` | **51.7 ± 10.8 µs** | 8757.3 ± 754.0 µs | **169×** | 148188 ± 13751 µs | **2867×** |
| `GradientBoosterTest` | **51.4 ± 7.5 µs** | 10275.8 ± 1065.8 µs | **200×** | N/A | N/A |
| `AlternateBinaryTargetCategoryTest` | **28.2 ± 3.8 µs** | 8615.7 ± 698.1 µs | **306×** | 147206 ± 9489 µs | **5228×** |

Cold load is where a serverless or edge deploy feels the difference: the fastest transpiled load here takes 131 ms against 0.05 ms for pmmlruntime. The Java spread is wide because the first of the 20 loads pays JIT and class verification, from 829 ms down to 4 ms.

## Batch throughput

| Model | Rows | per row | rows/s |
| --- | --- | --- | --- |
| `DecisionTreeIris` | 100 | 592 ± 20 ns | 1.69 M rows/s |
| `DecisionTreeIris` | 10000 | 428 ± 39 ns | 2.35 M rows/s |
| `GradientBoosterTest` | 10000 | 293 ± 22 ns | 3.42 M rows/s |

Per-row cost falls as the batch grows, because the columnar path stops rebuilding a `HashMap` per row and the `Value` slice stays in L1. Above 256 rows the provider shards with `rayon` on its own, so the thread count only matters on the columnar path. See [Batch API: One Method, Two Layouts](../batch/batch.md) and [Execution Provider & SIMD](../internals/provider.md).

## Caveats

The inputs are synthetic but valid for each `MiningSchema`, and the official `Audit.csv` batch stays out so both engines score identical rows. The hardware is a laptop with turbo enabled, not a pinned server, and the transpiler rows marked N/A are refused by JPMML's own validation.

> **Note:** Always compare release builds. A debug build moves the hot path further than the gap between the two engines.

## Next Steps

* [Correctness & Fixture Parity](./correctness.md): reproduce the 52 fixture suite.
* [Validation & Threshold Gating](./validation.md): gate a deploy on metrics from scored rows.
* [Batch API: One Method, Two Layouts](../batch/batch.md): choose `HashMap` or `RecordBatch`.
* [Concurrency & Memory](../production/concurrency.md): size the stack buffer and the `rayon` pool.

*Next: [Validation & Threshold Gating](./validation.md) → · Previous: [Correctness & Fixture Parity](./correctness.md)*
