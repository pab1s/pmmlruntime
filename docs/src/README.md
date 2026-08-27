<p align="center">
  <img src="https://raw.githubusercontent.com/pab1s/pmmlruntime/main/docs/images/pmmlruntime.png" alt="pmmlruntime — A fast, modern PMML inference runtime" width="75%">
</p>

<p align="center">
  <a href="https://crates.io/crates/pmmlruntime"><img alt="crates.io" src="https://img.shields.io/crates/v/pmmlruntime?style=flat-square&color=brightgreen"></a>
  <a href="https://docs.rs/pmmlruntime"><img alt="docs.rs" src="https://img.shields.io/docsrs/pmmlruntime?style=flat-square&label=docs.rs"></a>
  <a href="https://github.com/pab1s/pmmlruntime/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square"></a>
  <img alt="rustc" src="https://img.shields.io/badge/rustc-1.78%2B-lightgrey?style=flat-square&logo=rust">
  <img alt="pmml" src="https://img.shields.io/badge/PMML-4.4-4B8BBE?style=flat-square">
</p>

<p align="center">
  <b>Fast, safe, zero-JVM runtime for PMML 4.4 — classical ML in pure Rust.</b><br>
  Score XGBoost, LightGBM, sklearn, SparkML and R models from a single PMML file. No JVM. No Python. Just <code>Session::run</code>.
</p>

---

PMML is still the only vendor-neutral format that survives a round-trip between data science (sklearn, XGBoost, LightGBM, Spark, R → PMML via standard converters) and production. `pmmlruntime` is a from-scratch Rust inference engine for that format — **session-based** like ONNX Runtime, **tiny** like tract, **hardened** like a browser XML parser.

---

## Highlights

- **Complete PMML 4.4** — 19 model types, full `DataDictionary`/`MiningSchema`/`Targets`/`Output` semantics (see [Coverage](#coverage)).
- **Session API** — `PmmlEnv` + `Session` + `Batch`. Immutable `Arc<Ir>`, `Send + Sync`, sharded by a `Cpu` `ExecutionProvider`.
- **Two layouts, one method** — `HashMap<String, Value>` for single rows, `RecordBatch` (Arrow) for columnar batches. Same `sess.run(&batch as &dyn Batch)`.
- **Zero-copy, zero-alloc hot path** — `FieldId`/`SymbolId` interned once, `&[Value]` materialized per row in a stack buffer (`≤64` fields) or a `thread_local!` bump buffer; no `HashMap` per row for Arrow.
- **Parallel by default** — `rayon` auto-shard on the `Cpu` provider. `SIMD` (`wide` `f64x4`) for regression batches when `features = ["simd"]`.
- **Hardened XML** — `quick-xml 0.37`, `MAX_DEPTH 512`, 100 MB cap, DTD/XXE blocked, fuzzed (`cargo fuzz`), `miri` clean, `proptest` generators.
- **Embeds everywhere** — pure Rust, optional `python` (`pyo3 0.22`) and `C` ABI (opaque `PmmlEnv`/`PmmlSession` handles).
- **Apache-2.0** — commercial-friendly. MSRV 1.78, stable toolchain, `no_std`-ready core.

## Installation

**Rust**

```toml
[dependencies]
pmmlruntime = "0.1"
# optional SIMD for f64 batch
# pmmlruntime = { version = "0.1", features = ["simd"] }
```

```sh
cargo add pmmlruntime
# with SIMD
cargo add pmmlruntime --features simd
```

**Python** (extension-module, `libpython` only with `python` feature)

```toml
pmmlruntime = { version = "0.1", features = ["python"] }
```

```sh
pip install pmmlruntime  # when published — today: maturin develop --features python
python -c "import pmml_runtime; print(pmml_runtime.hello())"
```

Prerequisites: Rust 1.78+ (`rustup update`), no JDK, no Python at runtime.

## Quickstart

### Rust — single row

```rust
use std::collections::HashMap;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use pmmlruntime::session::batch::Batch;
use pmmlruntime::base::Value;

fn main() -> anyhow::Result<()> {
    let env = PmmlEnv::new();
    let xml = std::fs::read("model.pmml")?;
    let sess = Session::from_bytes(&env, &xml, SessionOptions::default())?;

    let mut input = HashMap::new();
    input.insert("Petal.Length".to_string(), Value::Continuous(1.4));
    input.insert("Petal.Width".to_string(), Value::Continuous(0.2));
    // categorical: sess.symbol_id("setosa").map(Value::Discrete)
    // or: sess.string_to_value("Species", "setosa")

    let out = sess.run(&input as &dyn Batch)?.into_single().unwrap();
    println!("{:?}", out.get("predictedValue")); // Discrete(SymbolId) or Continuous(f64)
    Ok(())
}
```

### Rust — Arrow batch (zero-copy)

```rust
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use pmmlruntime::session::batch::Batch;
use std::sync::Arc;
use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

let env = PmmlEnv::new();
let sess = Session::from_file(&env, "model.pmml", SessionOptions::default())?;

let schema = Arc::new(Schema::new(vec![
    Field::new("Petal.Length", DataType::Float64, true),
    Field::new("Petal.Width",  DataType::Float64, true),
]));
let batch = RecordBatch::try_new(schema, vec![
    Arc::new(Float64Array::from(vec![1.4, 6.0])) as _,
    Arc::new(Float64Array::from(vec![0.2, 2.5])) as _,
])?;

// same `run` — provider detects Columnar, uses col_map + rayon + optional SIMD
let results = sess.run(&batch as &dyn Batch)?.into_rows();
for row in &results {
    println!("{}", row.get("predictedValue").unwrap());
}
```

### Unified PMML — LightGBM / XGBoost / sklearn, same code

After conversion there is no "LightGBM PMML" or "XGBoost PMML". Converters for LightGBM, XGBoost, sklearn and R all emit one PMML 4.4 `MiningModel` (usually `Segmentation` with `multipleModelMethod="sum"` or `"modelChain"` over `TreeModel`/`RegressionModel` stumps). The scorer never knows the origin.

```sh
# Train anywhere, score the same way
cargo run -p pmmlruntime --example score_file -- bench/pmml/GradientBoosterTest.pmml
cargo run -p pmmlruntime --example score_file -- lightgbm.pmml input.csv --output out.csv
cargo run -p pmmlruntime --example score_file -- xgboost.pmml  input.csv --output out.csv
```

See [`crates/pmmlruntime/examples/score_file.rs`](crates/pmmlruntime/examples/score_file.rs) — it prints the model kind (`TreeModel` vs `MiningModel` with N segments), handles CSV → `RecordBatch` via `csv_str_to_record_batch`, and writes `predictedValue` + `Output` fields.

## Coverage

### Models — 19/19

| Model | PMML element | Status |
|---|---|---|
| Tree | `TreeModel` | ✓ |
| Regression | `RegressionModel` | ✓ |
| Mining (ensemble / chain) | `MiningModel` + `Segmentation` | ✓ (`majorityVote`, `weightedAverage`, `modelChain`, `selectFirst`, `selectAll`, …) |
| Scorecard | `Scorecard` | ✓ (reason codes, `pointsAbove`/`pointsBelow`) |
| Clustering | `ClusteringModel` | ✓ (`euclidean`, `squaredEuclidean`) |
| Naive Bayes | `NaiveBayesModel` | ✓ (discrete `PairCounts` + continuous `GaussianDistribution`) |
| k-NN | `NearestNeighborModel` | ✓ (`InlineTable` instances) |
| SVM | `SupportVectorMachineModel` | ✓ (RBF `exp(-γ‖x-sv‖²)`) |
| Neural Network | `NeuralNetwork` | ✓ (layers → neurons, `logistic`/`tanh`/`identity`) |
| General Regression | `GeneralRegressionModel` | ✓ (`PPMatrix`/`ParamMatrix`, factors/covariates) |
| Association Rules | `AssociationModel` | ✓ |
| Rule Set | `RuleSetModel` | ✓ (ordered `SimpleRule`, `defaultScore`) |
| Text | `TextModel` | ✓ |
| Time Series | `TimeSeriesModel` | ✓ (`ARIMA`, `ExponentialSmoothing`, `GARCH`, `StateSpace`, `Spectral`) |
| Anomaly Detection | `AnomalyDetectionModel` | ✓ (`iforest`, `clusterMeanDist`) |
| Bayesian Network | `BayesianNetworkModel` | ✓ |
| Gaussian Process | `GaussianProcessModel` | ✓ |
| Sequence | `SequenceModel` | ✓ |
| Baseline (change detection) | `BaselineModel` | ✓ (`zValue`, `CUSUM`, `chiSquare`, …) |

Every model honors `MiningSchema` (active/predicted/supplementary), `Targets` (rescale/cast/default), and `Output` (20+ `ResultFeature`s: `predictedValue`, `probability`, `confidence`, `entityId`, `reasonCode`, `decisionPath`, …).

### Transforms & expressions

Pooled, topologically sorted `DerivedField` + `TransformationDictionary`, compiled to a `Vec<Op>` bytecode VM:

`Apply` (80+ builtins via `BuiltinId`: arithmetic, trig, `min`/`max`/`median`, `modulo`/`hypot`, strings `uppercase`/`lowercase`/`substring`/`matches`/`replace`, dates via `chrono`, distributions `normalCDF`/`erf` via `statrs`/`libm`, …) · `MapValues` (single + multi-input hash lookup) · `Discretize` · `NormContinuous`/`NormDiscrete` · `Lag` (with `Avg`/`Min`/`Max`/`Sum` window) · `TextIndex` · aggregates (`count`/`sum`/`avg`/`min`/`max`) · `DefineFunction` stubs.

Spec references are `pmml.xsd` line-accurate (e.g., `AnomalyDetectionModel` 1718–1737, `BaselineModel` 3659–3815).

## Architecture

```
bytes ──▶ RawPmml ──▶ Ir ──▶ Session::run(&dyn Batch)
          (quick-xml,        (lower + verify,    (Value[FieldId] + Cpu provider)
           hardened)          Arc, topologically
                              sorted, Vec<Op>)
```

- **`base`** — zero-cost `Value` (`Continuous(f64)` / `Discrete(SymbolId)` / `Missing`), `FieldId(u32)` / `SymbolId(u32)`, `DataType`/`OpType`, `BumpArena`, `PmmlError`.
- **`xml`** — cold path only: `quick-xml 0.37` → `RawPmml`. `MAX_XML_SIZE 100 MB`, `MAX_DEPTH 512`, DTD/XXE rejected.
- **`ir`** — optimized `Ir`: `Arc` immutable, `Vec<NodeIr>` flat (root 0) for trees, `DerivedFieldIr` DAG with `Vec<Op>` bytecode, `Rodeo` interning cold. `lower(RawPmml) → Ir`, `verify_raw`/`verify_ir`.
- **`engine`** — pure evaluation on `&[Value]`: 19 model evaluators + `transform::vm` + optional `simd` (`wide` `f64x4`).
- **`session`** — ergonomic runtime: `PmmlEnv` (cheap `Arc` clone), `Session::from_bytes`/`from_file`, `Session::run(&dyn Batch) → BatchResult`, `Batch`/`BatchCtx`/`ExecutionProvider::Cpu` (rayon auto serial vs parallel), `arrow` helpers (`csv_str_to_record_batch`, `value_maps_to_record_batch`).
- **`ffi` / `python`** — `C` ABI (opaque `PmmlEnv`/`PmmlSession`, `PmmlStatusCode`) and `pyo3 0.22` placeholder (`features = ["python"]`).

Concurrency: `Session` is `Send + Sync`. Scoring is `&self` and uses a `≤64`-field stack buffer (L1-hot, 1 KB) or a `thread_local!` heap buffer — no per-row allocation, safe to share one `Session` across threads.

## Performance

| Path | Input | Notes |
|---|---|---|
| Cold `from_bytes` | Iris 2.9 KB → `Ir` → `Session` | ~68 µs (XML 0.37 + verify + lower) |
| Hot `run` single row | `&HashMap<String, Value>` | ~402 ns (stack `Value[64]`, no alloc) |
| Batch 100 k rows | `RecordBatch` | Arrow `col_map` + `rayon` sharding; `simd` regression `f64x4` when `features = ["simd"]` |

Stack threshold `64` covers ~90% of fixtures (Iris 3, Diabetes 8, Shopping 22). Larger models spill to a `thread_local Vec<Value>` that only grows. Numbers are single-threaded on a desktop x86_64; treat them as order-of-magnitude, not guarantees. Run `cargo test -- --nocapture` for the fixture bench harness.

## Security & Correctness

- **XML hardening** — depth, size, and entity caps enforced in `xml::reader`; `cargo test --test hardening` and `cargo fuzz` cover `fuzz_unmarshal` (60 s, 1 M execs, `rss_limit_mb=2048`).
- **Fuzz + miri + proptest** — `fuzz/fuzz_targets/fuzz_unmarshal.rs`, `cargo +nightly miri test -p pmmlruntime`, `proptest` for round-trip invariants (CI runs all three).
- **52 PMML fixtures** in [`bench/pmml/`](bench/pmml/) (decision trees, gradient boosters, mining chains, anomaly/baseline, text/sequence/time-series, …) plus `all_fixtures` parity tests (unsupported markup → `SKIP`, not `FAIL`).
- **Pedantic clippy** — `-W clippy::pedantic -D warnings` on workspace + all targets, `rustfmt` checked.

## API Overview

```rust
// Construct once (cold), share everywhere
let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, bytes, SessionOptions::default())?;
let sess = Session::from_file(&env, "model.pmml", SessionOptions::default())?;

// Introspect
sess.num_active_fields() // MiningSchema active count
sess.field_id("age")     // Option<FieldId> — for Value[FieldId] fast path
sess.symbol_id("sales")  // Option<SymbolId>
sess.string_to_value("dept", "sales") // DataType/OpType-aware Value
&sess.ir.field_names     // FieldId → name
&sess.ir.symbol_names    // SymbolId → string
&sess.ir.model           // ModelIr::Tree | Mining | ...

// Score — one method, any layout
let out: HashMap<String, Value> = sess.run(&single_map as &dyn Batch)?.into_single().unwrap();
let rows: Vec<HashMap<String, Value>> = sess.run(&vec_of_maps as &dyn Batch)?.into_rows();
let rows = sess.run(&record_batch as &dyn Batch)?.into_rows();
let batch: RecordBatch = sess.run(&batch as &dyn Batch)?.into_record_batch(schema, None)?;
```

- `Batch` is object-safe `Send + Sync`: impls for `HashMap<String, Value>`, `Vec<HashMap<_, _>>`, `[HashMap<_, _>]`, `RecordBatch`.
- `BatchResult::Rows(Vec<HashMap<String, Value>>)` today; `.into_record_batch(schema)` converts when you need Arrow output.
- `SessionOptions { graph_optimization_level, intra_op_num_threads }` — forwarded to the `Cpu` provider.

**C ABI**

```c
PmmlEnv *env = NULL;
PmmlCreateEnv(&env);
PmmlSession *sess = NULL;
PmmlCreateSession(env, "model.pmml", &sess);
// … scoring via PmmlRun* (planned)
PmmlReleaseSession(sess);
PmmlReleaseEnv(env);
```

**Python**

```python
import pmml_runtime
pmml_runtime.hello()  # "pmml-runtime" — InferenceSession planned
```

## Examples

```sh
# Inspect any PMML (LightGBM / XGBoost / sklearn — same binary)
cargo run -p pmmlruntime --example score_file -- bench/pmml/DecisionTreeIris.pmml
cargo run -p pmmlruntime --example score_file -- bench/pmml/GradientBoosterTest.pmml

# Batch CSV → scored CSV
cargo run -p pmmlruntime --example score_file -- model.pmml input.csv --output out.csv
# input.csv: header row matching MiningSchema active fields, e.g. Petal.Length,Petal.Width
```

## Developing

```sh
cargo fmt --check
cargo clippy --workspace -- -W clippy::pedantic -D warnings
cargo check --workspace
cargo test --workspace -- --nocapture
cargo test -p pmmlruntime --test all_fixtures -- --nocapture
cargo test -p pmmlruntime --test hardening -- --nocapture

# miri (nightly)
cargo +nightly miri test -p pmmlruntime --lib -- --nocapture

# fuzz (nightly, 60 s)
cargo +nightly fuzz run fuzz_unmarshal -- -max_total_time=60 -rss_limit_mb=2048 -print_final_stats=1
```

Project layout:

```
crates/pmmlruntime/src/{base,xml,ir,engine,session,ffi,python}
bench/pmml/*.pmml          # 52 fixtures
crates/pmmlruntime/examples/score_file.rs
crates/pmmlruntime/tests/  # all_fixtures, hardening, knn/nn/regression/svm/…
fuzz/fuzz_targets/fuzz_unmarshal.rs
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Acknowledgments

PMML spec by [DMG](https://dmg.org/pmml/v4-4-1/Index.html). Design inspiration from [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) and [ONNX Runtime](https://github.com/microsoft/onnxruntime) — the former for PMML evaluation semantics and completeness, the latter for the session/env/provider API (`PmmlEnv`/`Session`/`Batch`/`ExecutionProvider`). XML via [`quick-xml`](https://github.com/tafia/quick-xml), Arrow via [`arrow-rs`](https://github.com/apache/arrow-rs), numerics via [`statrs`](https://github.com/statrs-dev/statrs)/[`libm`](https://github.com/rust-lang/libm), interning via [`lasso`](https://github.com/dyn-tracing/lasso).
