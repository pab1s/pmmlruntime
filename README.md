# pmmlruntime

pmmlruntime is a PMML 4.4 scoring engine in Rust. It loads JPMML-compatible PMML, validates it, and scores it without a JVM.

[![CI](https://github.com/pab1s/pmmlruntime/actions/workflows/ci/badge.svg)](https://github.com/pab1s/pmmlruntime/actions)
[![crates.io](https://img.shields.io/crates/v/pmmlruntime.svg)](https://crates.io/crates/pmmlruntime)
[![docs.rs](https://img.shields.io/docsrs/pmmlruntime)](https://docs.rs/pmmlruntime)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](./LICENSE)

Dual-licensed under MIT or Apache-2.0. PMML spec by [DMG](https://dmg.org/pmml/v4-4/GeneralStructure.html).

### Documentation quick links

* [API docs (docs.rs)](https://docs.rs/pmmlruntime)
* [Architecture](./docs/ARCHITECTURE.md)
* [Benchmarks vs JPMML](./docs/BENCHMARK.md)
* [Porting map (Java → Rust)](./docs/PORTING.md)

### Benchmarks vs JPMML

Same `bench/pmml` fixtures (45 files from `jpmml-evaluator-testing`), same machine (i7-12700, `cargo bench --bench scoring`, release). JPMML `1.6` on OpenJDK 17, pmmlruntime `0.1.0`.

| Path | pmmlruntime | JPMML | Notes |
|---|---|---|---|
| Cold `Session::from_bytes` (DecisionTreeIris, 5 nodes) | **68 µs** | 553 ms | `quick-xml 0.37` + `lower` + `verify`; JPMML includes JAXB + visitors |
| Single `run` (Petal.Length=1.4/Petal.Width=0.2 → setosa) | **402 ns** | 1.22 µs | `HashMap<String,Value>` → `AHashMap` + stack `64` + flat `Vec<NodeIr>` |
| Batch `run_batch` 1k (sequential `CpuSerial`) | **336 µs** (2.97M rows/s) | — | `with_value_buffer` reuse, no rayon |
| Batch `run_batch_arrow` 1k (Arrow `Float64Array`) | **249 µs** (4.0M rows/s) | — | `RecordBatch` zero-copy, no per-row `HashMap` |
| Batch `run_batch_arrow` 100k (`CpuBatched`, `rayon` `par_chunks(256)`) | **61 ns/row** (16.5M rows/s) | 696 ns/row | `11×` on same 100k Iris; `Arrow` + `thread_local Vec<Value>` |

More tables, methodology and `criterion` plots in [`docs/BENCHMARK.md`](./docs/BENCHMARK.md). A single benchmark is never enough — run `cargo bench -p pmml-bench` on your hardware.

### Installation

The library is one crate:

```sh
cargo add pmmlruntime
```

Or add to `Cargo.toml`:

```toml
[dependencies]
pmmlruntime = "0.1.0"
```

To try the CLI / benches (same repo, separate members):

```sh
cargo run -p pmml-cli -- --help
cargo bench -p pmml-bench --bench scoring
```

Requires Rust `1.78+`, `edition=2021`.

### Example

```rust
use std::collections::HashMap;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use pmmlruntime::base::Value;

let env = PmmlEnv::new();
let bytes = std::fs::read("bench/pmml/DecisionTreeIris.pmml")?;
let sess = Session::from_bytes(&env, &bytes, SessionOptions::default())?;

let mut input = HashMap::new();
input.insert("Petal.Length".to_string(), Value::Continuous(1.4));
input.insert("Petal.Width".to_string(), Value::Continuous(0.2));

let out = sess.run(input)?;
assert_eq!(out.get("predictedValue"), Some(&Value::Discrete(pmmlruntime::base::SymbolId(0))));
// out = {"predictedValue": Discrete(setosa), "probability_setosa": 1.0, ...}
# Ok::<(), pmmlruntime::PmmlError>(())
```

Batch (1k rows, same Iris):

```rust
let batch: Vec<HashMap<String, Value>> = (0..1000).map(|i| {
    let mut m = HashMap::new();
    m.insert("Petal.Length".to_string(), Value::Continuous(1.0 + (i % 5) as f64));
    m
}).collect();
let outs = sess.run_batch(batch)?; // CpuBatched shards with rayon
```

Arrow zero-copy (100k):

```rust
let csv = std::fs::read_to_string("input.csv")?;
let batch = pmmlruntime::session::arrow::csv_str_to_record_batch(&csv, None, true)?;
let outs = sess.run_batch_arrow(&batch)?;
```

CLI:

```sh
cargo run -p pmml-cli -- inspect --model bench/pmml/DecisionTreeIris.pmml
cargo run -p pmml-cli -- run --model model.pmml --batch input.csv --output out.csv
cargo run -p pmml-cli -- verify --model model.pmml
```

### What it scores

PMML 4.4, verified on `bench/pmml` 45/45:

- **Models (12):** `TreeModel`, `RegressionModel`, `MiningModel`/`ModelChain`, `Scorecard`, `ClusteringModel`, `GeneralRegressionModel`, `NaiveBayesModel`, `NearestNeighborModel`, `NeuralNetwork`, `SupportVectorMachineModel`, `AssociationModel`, `RuleSetModel`
- **Output/Targets:** 26 `ResultFeature` (4 unsupported → `Missing`: `confidenceIntervalLower/Upper`, `standardError/Deviation`)
- **Transformations:** `TransformationDictionary`/`LocalTransformations` → `DerivedField` DAG (`Vec<Op>`), 100+ `BuiltinId` (`statrs`/`libm`/`chrono`), `Lag` (cap 128)
- **Batch:** `Vec<HashMap>` (JPMML-compat) or `RecordBatch` (`arrow 53` `Float64Array`/`StringArray`), `CpuSerial` vs `CpuBatched` (fallback `<256` rows serial)

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the full spine.

### Why use pmmlruntime

- You already score PMML with JPMML and want to drop the JVM (one static binary, WASM-ready, no `LoadingCache`/`BiMap` port).
- You need `402ns` single-row or `16.5M rows/s` batched on real fixtures, not microbenchmarks.
- You need hardening for untrusted XML — `MAX_DEPTH 512`, `100 MB` cap, DTD/XXE disabled in one place (`src/xml/reader.rs`).
- You want `cargo doc -p pmmlruntime --open` — one crate, `base/xml/ir/engine/session/ffi/python` modules, not 9 crates.

### Why not use pmmlruntime

- You need models not yet implemented: `AnomalyDetectionModel`, `BayesianNetworkModel`, `TimeSeriesModel`, `GaussianProcessModel`, `TextModel`, `BaselineModel` (all return `PmmlError::UnsupportedMarkup`).
- You need `TableLocator` with external `InlineTable` files > `100 MB` or custom `Visitor` mutation — not yet supported.
- You need a portable, ubiquitous tool on every platform today — JPMML on the JVM is still the reference.

### Is it really faster than JPMML?

Yes on the fixtures above, because:

- `quick-xml` pull parser replaces JAXB + 13 visitors; no XML object graph allocation on the hot path.
- `Ir` is `Arc` immutable, flat `Vec<NodeIr>` (branchless `match`), `AHashMap<String,FieldId>` (`Borrow<str>` zero-alloc, 3× SipHash) and stack `64` `Value` buffer (`with_value_buffer`); JPMML boxes per `MiningField`.
- `rayon` `par_chunks(256)` + `thread_local Vec<Value>` for `CpuBatched`; JPMML `LoadingCache` + `BiMap` add contention.

Numbers are on Iris (5 nodes). Wider trees/regressions show similar ratios; see `docs/BENCHMARK.md` for the full `criterion` `measurement-time 2` runs. If you find a fixture where JPMML wins, please file an issue with the PMML.

### Hardening

PMML is untrusted XML. Only `crates/pmmlruntime/src/xml/reader.rs` enforces the boundary. `PmmlReader::from_bytes`/`new_reader` reject `>100 MB` before allocating a parser; `read_event` increments `depth` on `Start` and errors `>512`. `quick-xml 0.37` does not expand entities — `<!ENTITY xxe SYSTEM "file:///etc/passwd">` stays literal.

### Repository layout

```text
crates/pmmlruntime/   lib: base (Value/FieldId/PmmlError) | xml (unmarshal) | ir (Ir/lower/verify) | engine (12 models, vm, simd) | session (Session/Batch/Provider) | ffi | python
crates/pmml-cli/      bin pmml-runtime: inspect/run/verify
crates/pmml-bench/    benches/scoring.rs + src/bin/large_trial.rs + tests/tree_parity.rs
bench/pmml/           45 PMML fixtures
docs/                 ARCHITECTURE.md, BENCHMARK.md, PORTING.md, OWNERSHIP.tsv
```

### Building and testing

```sh
git clone https://github.com/pab1s/pmmlruntime.git
cd pmmlruntime
cargo fmt --check
cargo clippy --workspace -- -W clippy::pedantic -D warnings
cargo check --workspace
cargo test --workspace
cargo test -p pmmlruntime --test all_fixtures -- --nocapture  # 45/45
cargo bench -p pmml-bench --bench scoring -- --sample-size 30
```

MSRV `1.78`, `resolver=2`. `cargo test --workspace` runs 70 unit + 9 integration + 122 doctests.

### License

MIT or Apache-2.0. See [`LICENSE`](./LICENSE). Upstream JPMML is AGPL-3.0 / BSD — this is a green-field port, not a relicense.
