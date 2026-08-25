<h1 align="center">pmmlruntime</h1>

<p align="center"><sub>A modern, fast inference runtime for PMML — written in Rust</sub></p>

<p align="center">
  <a href="https://github.com/pab1s/pmmlruntime/stargazers"><img src="https://img.shields.io/github/stars/pab1s/pmmlruntime?style=flat&label=%E2%98%85&color=4C8DFF" alt="GitHub stars" /></a>
  <a href="https://crates.io/crates/pmmlruntime"><img src="https://img.shields.io/crates/v/pmmlruntime?style=flat&label=crates.io&color=4C8DFF" alt="crates.io" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-4C8DFF?style=flat" alt="License: MIT OR Apache-2.0" /></a>
  <img src="https://img.shields.io/badge/PMML-4.4-4C8DFF?style=flat" alt="PMML 4.4" />
  <img src="https://img.shields.io/badge/Rust-1.78%2B-4C8DFF?style=flat&logo=rust&logoColor=white" alt="Rust 1.78+" />
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-4C8DFF?style=flat" alt="Platform" />
</p>

<p align="center">
  <strong>Run PMML where you run Rust.</strong><br/>
  pmmlruntime loads PMML 4.4, hardens the XML, lowers it to a flat <code>Ir</code>, and scores it — <br/>
  single-row in nanoseconds, batched in millions of rows per second — with no JVM.
</p>

```
 bytes: &[u8] ──► xml::unmarshal ──► RawPmml ──► ir::lower ──► Ir ──► Session ──► HashMap<String, Value>
                      │                  │              │           │
                      │  68µs vs 553ms   │  402ns vs    │  61ns/row  │  45/45 fixtures
                      │  JPMML cold      │  1.22µs JVM  │  16.5M/s   │  cargo test
```

> [!NOTE]
> pmmlruntime is a green-field Rust port of [jpmml/jpmml-evaluator](https://github.com/jpmml/jpmml-evaluator) (37,925 LOC) + [jpmml/jpmml-model](https://github.com/jpmml/jpmml-model) (22,405 LOC), not a mechanical transpilation. `Ir` is `Arc` immutable, `Session` is `Send+Sync` — the borrow checker replaces the visitor.

> [!IMPORTANT]
> Early release `0.1.0` on `main`. The Rust API (`Session`, `Value`, `PmmlError`) and file formats are stable; `ffi`/`python` surface is still evolving.

## Why pmmlruntime

- **No JVM, same PMML.** Score the `bench/pmml` fixtures JPMML scores — 45/45 `cargo test --test all_fixtures` on `DecisionTreeIris`, `Regression`, `Scorecard`, `Clustering`, `MiningModel` chains and more — without shipping a runtime.
- **Nanoseconds where it counts.** `68µs` cold `Session::from_bytes` (JPMML `553ms`), `402ns` single `run` (JPMML `1.22µs`), `61ns/row` batched Arrow `100k` (`16.5M rows/s`, `11×` JPMML `696ns`). See `docs/BENCHMARK.md`.
- **Hardened by default.** `quick-xml 0.37` pull parser, `MAX_DEPTH 512`, `100 MB` cap, DTD/XXE disabled, `trim_text` + `expand_empty_elements` — `SAXUtil`-equivalent in one place (`crates/pmmlruntime/src/xml/reader.rs`).
- **One crate, every surface.** `cargo add pmmlruntime` for Rust, `pmml-runtime inspect/run/verify` for the terminal, `pmml_runtime` for Python, `PmmlEnv`/`PmmlSession` for C — same `Ir`, same `Session`.

Read [Architecture](./docs/ARCHITECTURE.md) for the full spine, ownership, and concurrency model.

## Surfaces

| Entry point | Best for | Current capability |
|---|---|---|
| **Library** (`pmmlruntime`) | Embedding in Rust services, WASM, batch jobs | `PmmlEnv::new()` + `Session::from_bytes`/`from_file`, `run`/`run_batch`/`run_batch_arrow`, `Value`/`FieldId`/`SymbolId` |
| **CLI** (`pmml-cli`) | Inspecting models, running CSVs, CI verification | `pmml-runtime inspect --model model.pmml`, `run --model --batch input.csv --output out.csv`, `verify --model` |
| **C ABI** (`ffi`) | Calling from C/C++ and other runtimes | `PmmlCreateEnv`, `PmmlCreateSession`, `PmmlRelease*` (`#[repr(C)]` opaque, `Safety` contracts) |
| **Python** (`python`, feature `python`) | `import pmml_runtime` in notebooks | `pyo3 0.22` `extension-module` stub `hello()` today, `InferenceSession` in `0.2` |

## Current capabilities

### Inference Runtime

- 12 models: `Tree`, `Regression` (with `NormalizationMethod`), `Mining`/`ModelChain`, `Scorecard`, `Clustering`, `NaiveBayes`, `NearestNeighbor`, `NeuralNetwork`, `SupportVectorMachine`, `Association`, `RuleSet`, `GeneralRegression` (+ `RegressionTable` SIMD `wide f64x4` via `simd` feature);
- `MiningSchema` + `Output` + `Targets` (26 `ResultFeature`, 4 unsupported → `Missing`), `TransformationDictionary`/`LocalTransformations` → `DerivedFieldIr` DAG `Vec<Op>` + `vm` (`SmallVec` predicates, `EvalContext`);
- 100+ `BuiltinId` (`statrs`/`libm`/`chrono`), `Lag` per-thread `128`-cap, `Dewey`-style `DerivedField` topo sort;
- `ExecutionProvider`: `CpuSerial` (sequential `with_value_buffer` reuse) and `CpuBatched` (`rayon` `par_chunks(256)`, `<256` fallback serial).

### CLI workspace

- `inspect` prints `DataDictionary`/`MiningSchema`/`Output`/`Ir` counts;
- `run` does single `Petal.Length=1.4/Petal.Width=0.2` or batched CSV via `arrow::csv::Reader` → `RecordBatch` → `record_batch_to_value_maps` → `run_batch` → `arrow::csv::Writer`;
- `verify` runs `unmarshal` → `verify_raw` → `lower` → `verify_ir`.

### Bindings

- `ffi` is `Send+Sync`, reference-counted `PmmlEnv` (`Arc`), `Send+Sync` `PmmlSession` (`Box<Session>`); `Safety` docs per `extern "C"`.
- `python` is `extension-module`, `maturin`-ready; `py.allow_threads(|| sess.run(...))` planned to not block the GIL.

## Quick start

### Releases and downloads

No `crates.io` release yet — `0.1.0` lives on `main`/`development`. Until a tagged source release exists, this README recommends building from source as below. `LICENSE` is `MIT OR Apache-2.0`; upstream JPMML remains `AGPL-3.0`/BSD — this port is green-field, not a relicense.

Signed source releases and verification steps will be documented in `docs/RELEASING.md` when `0.1.0` is tagged.

### Requirements

- Rust `1.78` or newer (CI uses `1.78`, `edition=2021`, `resolver=2`);
- `cargo`, `git`;
- For benches: `cargo bench` with `criterion 0.5`.

### Start

```sh
git clone https://github.com/pab1s/pmmlruntime.git
cd pmmlruntime
cargo check --workspace
cargo test --workspace
cargo doc --workspace --open
```

`cargo doc --workspace --open` opens a single page for `pmmlruntime::{base,xml,ir,engine,session,ffi,python}`. To build only the library docs:

```sh
cargo doc -p pmmlruntime --open
```

### First run

pmmlruntime does not bundle a JVM. On first use, point it at PMML:

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
assert_eq!(out.get("predictedValue").unwrap().as_f64(), None); // Discrete -> SymbolId
# Ok::<(), pmmlruntime::PmmlError>(())
```

Also via crate-root re-exports:

```rust
use pmmlruntime::{PmmlEnv, Session, Value};
```

The session distinguishes `Value::Continuous(f64)`, `Value::Discrete(SymbolId)`, and `Value::Missing` (explicit, not `Option<Value>`). `SymbolId → String` via `sess.ir.symbol_names`.

## Library and terminal entry points

For the library API, see `cargo doc -p pmmlruntime --open` and `crates/pmmlruntime/src/session/session.rs`.

Build the workspace first:

```sh
cargo build --workspace
```

Then run the CLI or score from Rust:

```sh
cargo run -p pmml-cli -- inspect --model bench/pmml/DecisionTreeIris.pmml
cargo run -p pmml-cli -- run --model bench/pmml/DecisionTreeIris.pmml --batch bench/csv/iris.csv --output /tmp/out.csv
cargo run -p pmml-cli -- verify --model bench/pmml/DecisionTreeIris.pmml
```

Scoring from Rust — single row vs batched:

```rust
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, &std::fs::read("model.pmml")?, SessionOptions::default())?;
// single
let out = sess.run(std::collections::HashMap::new())?;
// batched — RowMajor (JPMML-compat) or Columnar Arrow
let batch: Vec<_> = vec![std::collections::HashMap::new(); 1000];
let outs = sess.run_batch(batch)?;
let arrow_batch: arrow::record_batch::RecordBatch = /* arrow::csv::Reader -> RecordBatch */;
let outs2 = sess.run_batch_arrow(&arrow_batch)?;
# Ok::<(), pmmlruntime::PmmlError>(())
```

Workspace CLI uses the same `PmmlEnv`/`Session` as the library; `ffi` and `python` share the same `Ir`. Batch picks `CpuBatched` (`rayon`) automatically for `run_batch` and falls back to `CpuSerial` for `<256` rows.

## Architecture

The spine is:

```text
bytes: &[u8] ──► xml::unmarshal ──► RawPmml ──► ir::lower ──► Ir ──► Session::from_ir ──► Arc<Ir>
                    │                  │              │           │               │
                    │  quick-xml 0.37  │  304 elem   │ Rodeo cold│ verify_ir     │ Arc clone
                    │  MAX_DEPTH 512   │  12 models  │ FieldId   │               │ AHashMap hot
                    │  100 MB cap      │  DTD blocked│ SymbolId  │               │ symbol_names_vec
                    └──────────────────┘             └───────────┘               └───────────────┘
                                                            Session::run(HashMap<String,Value>)
                                                                       │
                                                              with_value_buffer (stack 64)
                                                                       │
                                                              Value[FieldId] = [Missing; …]
                                                                       │
                                                ┌──────────────────────┴──────────────────────┐
                                                │        ExecutionProvider                  │
                                                │  CpuSerial (loop)  │  CpuBatched (rayon)  │
                                                └──────────────────────┬──────────────────────┘
                                                                       │
                                                              eval_derived_fields (DAG, Op vm)
                                                                       │
                                                              evaluate_model (Tree flat Vec<NodeIr>, Regression …)
                                                                       │
                                                              Output + Targets (26 ResultFeature)
                                                                       │
                                                              HashMap<String, Value>
```

Start with [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md). It provides the module map, flow, ownership, concurrency, storage, perf targets, invariants, and six `docs/` deep dives.

## Repository layout

```text
crates/pmmlruntime/   single lib: base (Value/FieldId/PmmlError/arena) | xml (unmarshal) | ir (Ir/lower/verify) | engine (12 models, vm, simd) | session (Session/Batch/Provider) | ffi | python
crates/pmml-cli/      pmml-runtime bin: inspect/run/verify (clap 4.5, arrow csv)
crates/pmml-bench/    benches/scoring.rs (criterion) + src/bin/large_trial.rs (10k/100k/1M/10M) + tests/tree_parity.rs

bench/pmml/           45 PMML fixtures (pmml-evaluator-testing subset)
bench/csv/            CSV fixtures for batch
docs/                 ARCHITECTURE.md, BENCHMARK.md, PORTING.md, OWNERSHIP.tsv, PLAN.md
```

## Hardening and correctness

PMML XML is untrusted input — the only hardening boundary is `crates/pmmlruntime/src/xml/reader.rs`:

```text
crates/pmmlruntime/src/xml/reader.rs  MAX_DEPTH 512, MAX_FILE_BYTES 100 MB, DTD/XXE disabled, trim_text + expand_empty_elements
```

- `pmml-evaluator` Java uses `org.jpmml.model.SAXUtil` + `jakarta.xml.bind`; pmmlruntime uses one auditable place that wraps `quick-xml` 0.37.
- `PmmlReader::from_bytes` and `new_reader` both enforce the caps; `read_event` tracks depth per `Start`/`End` (`Empty` does not increment).
- `xml` returns owned `RawPmml` (`String`/`Vec`, cold); `ir::lower` validates `DataType`/`OpType`/`MiningFunction`/`ResultFeature` case-sensitively per `pmml.xsd:4,490`; `verify_raw`/`verify_ir` reject `UnsupportedMarkup` (`AnomalyDetectionModel`, `TimeSeriesModel`, etc.) — parity gated on `bench/pmml` `45/45`.

Details: [SECURITY.md](./SECURITY.md) (when added), `crates/pmmlruntime/src/xml/reader.rs` module docs, `docs/BENCHMARK.md` Java vs Rust tables.

## Development and verification

Before sending a change, read [CONTRIBUTING.md](./CONTRIBUTING.md).

Common workspace commands:

```sh
cargo fmt --check
cargo clippy --workspace -- -W clippy::pedantic -D warnings
cargo check --workspace
cargo test --workspace
cargo test -p pmmlruntime --test all_fixtures -- --nocapture # 45/45
cargo test -p pmmlruntime --doc
cargo doc --workspace --no-deps
cargo bench -p pmml-bench --bench scoring -- --sample-size 30
```

Run one crate in isolation:

```sh
cargo test -p pmmlruntime --lib
cargo test -p pmml-bench --test tree_parity
cargo run -p pmml-cli -- --help
```

Before submitting code, run `fmt`, `clippy`, and focused tests proportionate to the change, followed by `git diff --check`.

## Documentation

- [Documentation index and architecture map](./docs/ARCHITECTURE.md)
- [Benchmarks — JPMML vs Rust](./docs/BENCHMARK.md)
- [Porting map — Java → Rust per-field](./docs/PORTING.md) + [OWNERSHIP.tsv](./docs/OWNERSHIP.tsv)
- [Contributing guide](./CONTRIBUTING.md)
- [PMML 4.4 spec](https://dmg.org/pmml/v4-4/GeneralStructure.html) (`pmml.xsd:4,490` lines)
- Upstream: [jpmml-evaluator](https://github.com/jpmml/jpmml-evaluator) (`37,925` LOC) + [jpmml-model](https://github.com/jpmml/jpmml-model) (`22,405` LOC)

## License

pmmlruntime is licensed under [MIT OR Apache-2.0](./LICENSE). See [NOTICE](./NOTICE) for attribution. Third-party components remain subject to their respective licenses.

pmmlruntime, JPMML, and PMML are not affiliated with the Apache Software Foundation.

## About

pmmlruntime is a modern, fast inference runtime for PMML 4.4 — written in Rust, hardened for untrusted XML, and built for real batch throughput. No JVM.

### Topics

`pmml` `jpmml` `pmml-evaluator` `inference` `runtime` `rust` `arrow` `scoring` `machine-learning` `batch` `hardening` `quick-xml`

### Resources

[Readme](#readme-ov-file)
[Apache-2.0 OR MIT](#Apache-2.0-OR-MIT-1-ov-file)

