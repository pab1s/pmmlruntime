<p align="center">
  <img src="https://raw.githubusercontent.com/pab1s/pmmlruntime/main/docs/images/pmmlruntime.png" alt="pmmlruntime" width="65%">
</p>

<p align="center">
  <a href="https://crates.io/crates/pmmlruntime"><img alt="crates.io" src="https://img.shields.io/crates/v/pmmlruntime?style=flat-square&color=brightgreen"></a>
  <a href="https://docs.rs/pmmlruntime"><img alt="docs.rs" src="https://img.shields.io/docsrs/pmmlruntime?style=flat-square&label=docs.rs"></a>
  <a href="https://github.com/pab1s/pmmlruntime/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square"></a>
  <img alt="rustc" src="https://img.shields.io/badge/rustc-1.78%2B-lightgrey?style=flat-square&logo=rust">
  <img alt="pmml" src="https://img.shields.io/badge/PMML-4.4-4B8BBE?style=flat-square">
</p>

<p align="center">
  <b>A fast, modern PMML inference runtime.</b><br>
  Zero-JVM, pure Rust — score sklearn, XGBoost, LightGBM, SparkML and R models from one PMML file.
</p>

---

`sklearn2pmml`, `jpmml-sparkml`, `jpmml-xgboost`, `jpmml-lightgbm` and `r2pmml` all emit the same PMML 4.4 `MiningModel`. `pmmlruntime` is a from-scratch engine for that file — session-based like ONNX Runtime, tiny like tract, hardened like a browser parser. No JVM. No Python at runtime.

```toml
[dependencies]
pmmlruntime = "0.1"
```

```sh
cargo add pmmlruntime
```

> Requires Rust 1.78+. No JDK.

## Quick start

```rust
use std::collections::HashMap;
use pmmlruntime::{PmmlEnv, Session, SessionOptions, Value};
use pmmlruntime::session::batch::Batch;

fn main() -> anyhow::Result<()> {
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &std::fs::read("model.pmml")?, SessionOptions::default())?;

    let mut input = HashMap::new();
    input.insert("Petal.Length".into(), Value::Continuous(1.4));
    input.insert("Petal.Width".into(), Value::Continuous(0.2));

    let out = sess.run(&input as &dyn Batch)?.into_single().unwrap();
    println!("{:?}", out.get("predictedValue"));
    Ok(())
}
```

Load once (`from_bytes` / `from_file`), then `run` — one row or 100k rows through the same call. Sessions are `Send + Sync`; share one across threads.

## Why pmmlruntime?

|  | JPMML-Evaluator | **pmmlruntime** |
|---|---|---|
| **Runtime** | JVM (Java 11+) | Pure Rust |
| **Deploy** | JAR + classpath | Single binary or library |
| **Cold start** | JVM warmup | Microseconds for small models |
| **Batch** | Row-major only | Row-major and columnar (Arrow), sharded across cores |
| **Hardening** | JAXB | Hardened XML, fuzzed and sanitized |
| **License** | AGPL-3.0 (commercial on request) | Apache-2.0 |

For teams that train in Python/R/Spark and ship in Rust or lean containers.

## What it runs

All 19 PMML 4.4 model types — tree, regression, ensembles and chains, scorecard, clustering, Naive Bayes, k-NN, SVM, neural network, and more — with full preprocessing, postprocessing and output handling. Transforms are compiled and evaluated automatically. See [`bench/pmml/`](bench/pmml/) for 52 fixtures.

## One file, any framework

Every converter emits the same PMML. Score them the same way:

```sh
cargo run -p pmmlruntime --example score_file -- bench/pmml/DecisionTreeIris.pmml
cargo run -p pmmlruntime --example score_file -- bench/pmml/GradientBoosterTest.pmml

# CSV in → CSV out (header must match the model’s active fields)
cargo run -p pmmlruntime --example score_file -- model.pmml input.csv --output out.csv
```

Batch scoring uses the same `run` call; pass a `RecordBatch` for columnar speed. See [`score_file.rs`](crates/pmmlruntime/examples/score_file.rs).

## Performance

Order-of-magnitude on desktop x86_64, single-threaded (repro with `cargo test`):

| Path | Observed |
|---|---|
| Load Iris (2.9 KB) | ~68 µs |
| Score one row | ~400 ns |
| Score 100 k rows (columnar) | Sharded across cores |

## Architecture

```
PMML bytes → session → run
```

One immutable session per model, one method to score. Full internals: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [docs.rs](https://docs.rs/pmmlruntime).

## Security & correctness

Hardened XML, fuzzed and checked under `miri`, with fixtures and property tests in CI.

## Documentation

- API reference: [docs.rs/pmmlruntime](https://docs.rs/pmmlruntime)
- Internals: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Example: [`crates/pmmlruntime/examples/score_file.rs`](crates/pmmlruntime/examples/score_file.rs)

## Acknowledgments

PMML spec by [DMG](https://dmg.org/pmml/v4-4-1/Index.html). Evaluation semantics from [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator); API inspiration from [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [tract](https://github.com/sonos/tract).

## Cite

If you use `pmmlruntime` in academic work, please cite as:

```bibtex
@software{pmmlruntime,
  author  = {Olivares, Pablo},
  title   = {pmmlruntime: A fast, modern PMML inference runtime},
  year    = {2026},
  url     = {https://github.com/pab1s/pmmlruntime},
  version = {0.1.0},
  license = {Apache-2.0}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
