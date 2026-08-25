# pmmlruntime

Fast PMML 4.4 scoring in Rust — no JVM.

[![CI](https://github.com/pab1s/pmmlruntime/actions/workflows/ci/badge.svg)](https://github.com/pab1s/pmmlruntime/actions)
[![crates.io](https://img.shields.io/crates/v/pmmlruntime.svg)](https://crates.io/crates/pmmlruntime)
[![docs.rs](https://img.shields.io/docsrs/pmmlruntime)](https://docs.rs/pmmlruntime)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](./LICENSE)

Load a PMML file and score it. Single rows in nanoseconds, batches in millions of rows per second.

### Quick links

* [API docs](https://docs.rs/pmmlruntime) · [Architecture](./docs/ARCHITECTURE.md) · [Benchmarks](./docs/BENCHMARK.md) · [PMML 4.4 spec](https://dmg.org/pmml/v4-4/GeneralStructure.html)

### Benchmarks

Same 45 PMML files, same machine (i7-12700, release). Reference is the Java implementation.

| Task | pmmlruntime 0.1.0 | Reference |  |
|---|---|---|---|
| Load model (cold) | **68 µs** | 553 ms |  |
| Score one row | **402 ns** | 1.22 µs |  |
| Score 1k rows | **336 µs** | — |  |
| Score 100k rows (batched) | **61 ns / row** | 696 ns / row | 11× |

More tables and method in [`docs/BENCHMARK.md`](./docs/BENCHMARK.md). Run `cargo bench -p pmml-bench` on your hardware.

### Install

```sh
cargo add pmmlruntime
```

```toml
[dependencies]
pmmlruntime = "0.1.0"
```

Requires Rust 1.78+.

### Use it

Single row:

```rust
use std::collections::HashMap;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};

let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, &std::fs::read("model.pmml")?, SessionOptions::default())?;

let mut input = HashMap::new();
input.insert("Petal.Length".to_string(), pmmlruntime::Value::Continuous(1.4));
input.insert("Petal.Width".to_string(), pmmlruntime::Value::Continuous(0.2));

let out = sess.run(input)?;
assert!(out.contains_key("predictedValue"));
# Ok::<(), pmmlruntime::PmmlError>(())
```

Batch from CSV:

```rust
let batch = pmmlruntime::session::arrow::csv_str_to_record_batch(&std::fs::read_to_string("input.csv")?, None, true)?;
let outs = sess.run_batch_arrow(&batch)?;
```

CLI:

```sh
cargo run -p pmml-cli -- inspect --model model.pmml
cargo run -p pmml-cli -- run --model model.pmml --batch input.csv --output out.csv
cargo run -p pmml-cli -- verify --model model.pmml
```

Python and C are available via `ffi` (C ABI) and `python` (pyo3) features.

### What it runs

Verified on the 45 PMML fixtures in `bench/pmml`:

- **Models:** tree, regression, mining/model chains, scorecard, clustering, general regression, naive Bayes, nearest neighbor, neural network, support vector machine, association, rule set
- **Features:** missing values, outliers, mining schema, outputs and targets, transformation dictionaries and local transformations
- **Batch:** single rows or CSV/Arrow batches; automatically uses parallel execution for large batches

See `bench/pmml` for the fixtures. Unsupported PMML returns a clear error instead of a wrong score.

### Why pmmlruntime

- **No JVM.** One static binary, works where Rust works.
- **Fast on real models.** Measured on full PMML files, not synthetic microbenchmarks.
- **Safe for untrusted PMML.** Rejects oversized and overly deep documents, does not expand external entities.
- **One crate.** `cargo add pmmlruntime` and `cargo doc --open` — library, CLI and bench in the same workspace.

### When not to use it

- You need `AnomalyDetection`, `BayesianNetwork`, `TimeSeries`, `GaussianProcess`, `Text` or `Baseline` models — not implemented yet.
- You need external `TableLocator` files larger than 100 MB.
- You need the most widely deployed PMML runner today — the Java reference is still the most portable.

### Repository layout

```text
crates/pmmlruntime   library
crates/pmml-cli      CLI (inspect / run / verify)
crates/pmml-bench    benchmarks and large-batch trials
bench/pmml           45 PMML fixtures
docs                 architecture, benchmarks, and spec notes
```

### Develop

```sh
git clone https://github.com/pab1s/pmmlruntime.git
cd pmmlruntime
cargo test --workspace          # 45/45 fixtures + 122 doctests
cargo bench -p pmml-bench --bench scoring
```

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) and [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

### License

MIT or Apache-2.0. See [`LICENSE`](./LICENSE).
