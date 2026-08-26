<p align="center">
  <img src="https://raw.githubusercontent.com/pab1s/pmmlruntime/main/docs/images/pmmlruntime.png" alt="pmmlruntime — A fast, modern PMML inference runtime" width="75%">
</p>

# pmmlruntime

<p align="center">
  <strong>PMML 4.4 — without the JVM.</strong><br/>
  Score LightGBM, XGBoost, sklearn &amp; R models at 16M rows/s. 68&nbsp;&micro;s cold start. No GC pauses. No 300&nbsp;MB runtime.
</p>

<p align="center">
  <a href="https://github.com/pab1s/pmmlruntime/actions/workflows/ci"><img alt="CI" src="https://github.com/pab1s/pmmlruntime/actions/workflows/ci/badge.svg"/></a>
  <a href="https://crates.io/crates/pmmlruntime"><img alt="crates.io" src="https://img.shields.io/crates/v/pmmlruntime.svg"/></a>
  <a href="https://docs.rs/pmmlruntime"><img alt="docs.rs" src="https://img.shields.io/docsrs/pmmlruntime"/></a>
  <img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.78-blue"/>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"/></a>
  <img alt="PMML 4.4" src="https://img.shields.io/badge/PMML-4.4-brightgreen"/>
  <img alt="Models" src="https://img.shields.io/badge/models-19%2F19-success"/>
</p>

<p align="center">
  <a href="https://docs.rs/pmmlruntime"><b>API docs</b></a> •
  <a href="./docs/ARCHITECTURE.md"><b>Architecture</b></a> •
  <a href="./docs/BENCHMARK.md"><b>Benchmarks</b></a> •
  <a href="https://dmg.org/pmml/v4-4/GeneralStructure.html"><b>PMML spec</b></a> •
  <a href="#quickstart"><b>Quickstart</b></a>
</p>

---

You trained it in Python or R. You exported it as `model.pmml`. In production that `.pmml` is a liability: a JVM to feed, 500&nbsp;ms cold starts, and a scoring path that allocates per row.

`pmmlruntime` is a PMML 4.4 runtime written in Rust that treats that file as a compiled plan: hardened XML → flat IR → branchless scoring. The same `sklearn2pmml` / `lightgbm2pmml` / `r2pmml` artifact runs at **402&nbsp;ns / row** and **16.5M rows/s batched** on one core.

### Performance — same 52 PMML files, same i7-12700, release

| Task | **pmmlruntime 0.1.0** | Java reference | Δ |
|---|---|---|---|
| **Load model (cold)** | **68&nbsp;&micro;s** | 553&nbsp;ms | **8,000×** |
| **Score one row** | **402&nbsp;ns** | 1.22&nbsp;&micro;s | **3.0×** |
| **Score 1k rows** | **336&nbsp;&micro;s** | 743&nbsp;&micro;s | **2.2×** |
| **Score 100k batched** | **61&nbsp;ns / row** | 696&nbsp;ns / row | **11×** |

> 68&nbsp;µs is `quick-xml` + `lower` + `verify` for `DecisionTreeIris.pmml` (2.9&nbsp;KB, 5 nodes). Java is `PMMLUtil.unmarshal` after 10k warmup. Full tables, method and `cargo bench` repro in [`docs/BENCHMARK.md`](./docs/BENCHMARK.md). Run `cargo bench -p pmml-bench --bench scoring` on your hardware.

## Why this exists

*Export friction is real.* Python trains, Java scores — with a serialization tax. Every framework has its own PMML emitter, but every PMML is the same 304-element XML once it lands. `pmmlruntime` leans into that: one spec, one IR, no framework-specific paths.

- **No JVM.** One static binary. Runs on `x86_64`/`aarch64`, in containers, at the edge.
- **Fast on real models.** Measured on full PMML files, not micro-kernels. `BumpArena` + `SmallVec` + `AHashMap` + `rayon` sharding; flat `Vec<NodeIr>` for trees.
- **Safe for untrusted PMML.** `100 MB` cap, `depth 512`, DTD/XXE blocked, `Missing` propagation per `MiningSchema`. Wrong markup returns `Err(UnsupportedMarkup)` instead of a silent wrong score. `cargo fuzz` 60&nbsp;s ~1M execs, `hardening_l7` (depth 5k, cycle, XXE, thread-safety), `miri` clean.

## What it runs — 19/19 PMML 4.4 models, 304 elements

| Family | Models |
|---|---|
| **Tabular** | `TreeModel`, `RegressionModel`, `GeneralRegressionModel`, `SupportVectorMachineModel`, `NaiveBayesModel`, `NearestNeighborModel`, `NeuralNetwork`, `ClusteringModel`, `Scorecard`, `RuleSetModel`, `AssociationModel` |
| **Ensemble** | `MiningModel` (`modelChain` / `weightedAverage` / `majorityVote` — LightGBM & XGBoost are just `MiningModel(sum)` over stumps) |
| **Specialized** | `TimeSeriesModel` (ARIMA/ExpoSmooth/GARCH/StateSpace), `GaussianProcessModel` (4 kernels), `TextModel` (TF-IDF + cosine/euclidean), `SequenceModel`, `BayesianNetworkModel`, `AnomalyDetectionModel`, `BaselineModel` |

Plus `DataDictionary`/`MiningSchema` (outlier/missing/invalid), `TransformationDictionary` & `LocalTransformations` (`Apply` 100 builtins, `MapValues`, `Discretize`, `NormContinuous/Discrete`, `TextIndex`, `Aggregate`, `Lag`), `Targets`/`Output` (26 `ResultFeature`), `Extension` (stored, not evaluated). Only `ModelComposition`/`CenterFields` remain `UnsupportedMarkup`.

Verified on **52 fixtures in `bench/pmml`** (`cargo test -p pmmlruntime --test all_fixtures` → 51 OK + 1 SKIP `weightedConfidence` expected).

```
bench/pmml/
  DecisionTreeIris.pmml  GradientBoosterTest.pmml  AnomalyDetectionTest.pmml
  BayesianSimpleTest.pmml  TextTest.pmml  TimeSeriesTest.pmml  … (52 total)
```

## Install

```sh
cargo add pmmlruntime
```

```toml
[dependencies]
pmmlruntime = "0.1.0"
```

Requires Rust **1.78+**. No `libpython`, no JVM. Optional features: `simd = ["wide"]` (4-wide batch), `python = ["pyo3"]`.

## Quickstart — 30 seconds

### 1. Single row (any framework — same code)

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions, Value};
use std::collections::HashMap;

let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, &std::fs::read("model.pmml")?, SessionOptions::default())?;
// also: Session::from_file(&env, "lightgbm.pmml", SessionOptions::default())?

let mut input = HashMap::new();
input.insert("Petal.Length".into(), Value::Continuous(1.4));
input.insert("Petal.Width".into(),  Value::Continuous(0.2));
// categorical: let sid = sess.symbol_id("sales").unwrap(); input.insert("dept".into(), Value::Discrete(sid));

let out = sess.run(input)?;               // HashMap<String, Value>
assert!(out.contains_key("predictedValue")); // + Probability_* / clusterId / …
# Ok::<(), pmmlruntime::PmmlError>(())
```

> **Tip:** Cache `FieldId` for hot loops: `let fid = sess.field_id("age").unwrap(); sess.run_with_ids(&[(fid, Value::Continuous(34.0))])?` — the 402&nbsp;ns path.

### 2. Batch — CSV or Arrow, same `Session`

```rust
// input.csv: header must match MiningSchema active fields
// x
// 0.5
// 1.0
let batch = pmmlruntime::session::arrow::csv_str_to_record_batch(
    &std::fs::read_to_string("input.csv")?, None, true
).map_err(|e| anyhow::anyhow!(e))?;

// Zero-copy Arrow path — no HashMap per row, 61 ns/row at 100k
let outs: Vec<HashMap<String, Value>> = sess.run_batch_arrow(&batch)?;

// Or row-major (Python dict / HashMap naturally)
let rows = vec![ /* Vec<HashMap<String, Value>> */ ];
let outs = sess.run_batch(rows)?;
```

`CpuBatched` (`rayon` `par_chunks(256)`) for `>10k` rows; `<256` falls back to serial (no spawn overhead). See `docs/ARCHITECTURE.md` §4 and `BENCHMARK.md` §3.

### 3. CLI — no Rust code

```sh
# inspect what the model expects
cargo run -p pmml-cli -- inspect --model model.pmml
# → DataDictionary, MiningSchema active/target, Output, Ir counts

# single-file batch
cargo run -p pmmlruntime --example score_file -- model.pmml input.csv --output out.csv
cargo run -p pmml-cli -- run --model model.pmml --batch input.csv --output out.csv

# verify (unmarshal → verify_raw → lower → verify_ir)
cargo run -p pmml-cli -- verify --model model.pmml
```

Full annotated example: `crates/pmmlruntime/examples/score_file.rs`.

## CLI & bindings

| Surface | Status | Example |
|---|---|---|
| **Rust** | Stable | `cargo add pmmlruntime` |
| **CLI** | Stable | `pmml-cli inspect/run/verify` |
| **C ABI** | Stub → 0.2.0 | `pmml_runtime.h` via `cbindgen`, `PmmlEnv`/`PmmlSession` handles |
| **Python** | Stub → 0.2.0 | `import pmml_runtime; pmml_runtime.hello()` (pyo3, `maturin`) |

## Security

PMML is often untrusted (uploaded models). Hardening is the **only** place that enforces limits — every entry goes through it:

- `100 MB` file cap (before parsing), `depth 512` (per `PmmlReader::read_event`), DTD/external entities never expanded (`&xxe;` stays literal).
- `cargo fuzz` 60&nbsp;s (~1M execs) on `unmarshal + lower + Session::from_bytes`; `hardening_l7` tests 5k-node tree (no stack overflow), `DerivedField` cycle tolerance, `LAG_BUFFER` isolation.

See `crates/pmmlruntime/src/xml/reader.rs` and `fuzz/fuzz_targets/fuzz_unmarshal.rs`.

## Architecture — in one picture

```
bytes ──► xml::unmarshal ──► RawPmml ──► ir::lower ──► Ir ──► Session::from_ir ──► Arc<Ir>
                                 │ 304 elem   Rodeo cold  verify_ir  Arc clone
                                 └─────────── tight loop ───────────┘
                                                        │
                                   with_value_buffer (stack 64 L1 cache hot)
                                                        │
                                   Value[FieldId] = [Missing; needed]
                                                        │
                                   ┌──────── ExecutionProvider ────────┐
                                   │ CpuSerial  │  CpuBatched (rayon)  │
                                   └────────────┴──────────────────────┘
                                                        │
                                   eval_derived_fields (DAG, Op bytecode)
                                                        │
                                   evaluate_model (flat Vec<NodeIr> branchless)
                                                        │
                                   Output + Targets → HashMap { predictedValue, Probability_* }
```

`Ir` is `Arc` immutable, `Session` is `Send+Sync` (`&self` scoring). `BumpArena` per `par_chunk`, `SmallVec<[PredicateIr;4]>`, `memchr` fast `InlineTable` split. `LAG_BUFFER` is `thread_local!` per `FieldId` (cap 128).

Deep dive: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md), `docs/OWNERSHIP.tsv`, `docs/BENCHMARK.md`.

## How it compares

| | **pmmlruntime (Rust)** | Java reference (1.7.7) |
|---|---|---|
| Cold load | **68&nbsp;µs** | 553&nbsp;ms |
| 1 row | **402&nbsp;ns** | 1.22&nbsp;µs |
| 100k batched | **61&nbsp;ns / row** | 696&nbsp;ns / row |
| Runtime | single binary, no JVM | JVM + Guava + JAXB |
| Batch | `HashMap` **or** Arrow `RecordBatch` | `Map<String,?>` / `Table` |
| Thread safety | `Session: Send+Sync` (`&self`) | `ModelEvaluator` `Send+Sync`, builder not |
| Safety | 100&nbsp;MB / 512 depth / XXE blocked | `SAXUtil` equivalent |

Reference is the Java implementation measured after 10k warmup, 100k iters (`System.nanoTime`) on i7-12700. See `docs/BENCHMARK.md` for method and `cargo bench` repro.

## Repository layout

```
crates/pmmlruntime   library (base/xml/ir/engine/session/ffi/python)
crates/pmml-cli      CLI (inspect / run / verify)
crates/pmml-bench    benchmarks + large_trial
bench/pmml           52 PMML fixtures (DecisionTreeIris, GradientBooster, …)
docs                 architecture, benchmarks, spec notes
fuzz                 libFuzzer target (unmarshal + lower + Session)
```

## Develop

```sh
git clone https://github.com/pab1s/pmmlruntime.git
cd pmmlruntime
cargo test --workspace          # 124 doc + 104 lib + 52 fixtures (51 OK + 1 SKIP) + 14 hardening
cargo test -p pmmlruntime --test hardening_l7
cargo bench -p pmml-bench --bench scoring -- --sample-size 30
cargo fuzz run fuzz_unmarshal -- -max_total_time=60
```

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) and [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md). PRs to `development`; `main` is protected.

## Documentation

- **API:** https://docs.rs/pmmlruntime
- **Architecture:** `docs/ARCHITECTURE.md` — data flow, ownership, concurrency, Arrow bridge
- **Benchmarks:** `docs/BENCHMARK.md` — 52 fixtures table, large_trial scaling
- **Spec:** https://dmg.org/pmml/v4-4/GeneralStructure.html · `spec/pmml.xsd` (4,490 lines)

## Who is this for

- You ship PMML from `sklearn2pmml`, `lightgbm2pmml`, `r2pmml`, `jpmml-spark` and need to score without a JVM in prod.
- You batch-score CSV/Arrow at millions of rows/s and cannot afford per-row `HashMap` + `String` alloc.
- You accept PMML from users and need XXE/depth/file-cap hardening + `UnsupportedMarkup` fail-fast.

## When not to use it

- You need `TableLocator` external tables larger than `100 MB` (placeholder returns empty batch).
- You need the most portable PMML runner today — the Java reference still runs everywhere Java does.
- You need a Python-native training loop — use `sklearn`/`LightGBM` directly; this is inference only.

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

---

Built from the PMML 4.4 `pmml.xsd` with `quick-xml` 0.37, `statrs`/`libm`, `arrow` 53 and `rayon`. No transpile — green-field IR for posterior plan optimization.
