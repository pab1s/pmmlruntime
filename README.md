# pmmlruntime

> **Rust port of [jpmml/jpmml-evaluator](https://github.com/jpmml/jpmml-evaluator) — PMML 4.4 reference evaluator**

Green-field spec → Rust with ONNX Runtime design (not mechanical transpilation):
* `Ir` is `Arc` immutable, `Session` is `Send+Sync` — `Drop` not GC, borrow-checker not style guide
* `68µs` cold `Session::from_bytes` + `402 ns` single `run` + `61 ns/row` Arrow 100k batched — `BENCHMARK.md` tables for 45 fixtures
* `0` tests skipped — parity gated on `pmml-evaluator-testing` fixtures `45/45` `cargo test -p pmml-session --test all_fixtures`

**Status:** `0.1.0` on `development` — 9 crates, `13,642` LOC Rust, `cargo doc --open` ready. See `docs/ARCHITECTURE.md`.

**Install**

```toml
[dependencies]
pmml-session = { version = "0.1.0" }
pmml-core    = { version = "0.1.0" }  # Value, FieldId, DataType
```

```sh
cargo add pmml-session pmml-core
cargo add --dev pmml-bench   # criterion benches
```

**Smallest useful example**

```rust
use std::collections::HashMap;
use pmml_core::Value;
use pmml_session::{PmmlEnv, Session, SessionOptions};

let env = PmmlEnv::new();
// Minimal PMML bytes — TreeModel classification (Iris-like). For a real file use Session::from_file.
let xml = br#"
<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary>
<TreeModel functionName="classification">
  <MiningSchema><MiningField name="x"/></MiningSchema>
  <Node score="a"><True/></Node>
</TreeModel></PMML>"#;
let sess = Session::from_bytes(&env, xml, SessionOptions::default())?;
let mut input = HashMap::new();
input.insert("x".to_string(), Value::Continuous(1.4));
let out = sess.run(input)?;
assert_eq!(out.get("predictedValue"), Some(&Value::Discrete(pmml_core::SymbolId(0))));
# Ok::<(), pmml_core::PmmlError>(())
```

CLI:

```sh
cargo run -p pmml-cli -- inspect --model bench/pmml/DecisionTreeIris.pmml
cargo run -p pmml-cli -- run --model bench/pmml/DecisionTreeIris.pmml --input input.csv --output output.csv
```

**Links**

- API docs: `cargo doc --workspace --open` (or `docs.rs` once published) — crate docs `pmml-core`, `pmml-xml`, `pmml-ir`, `pmml-evaluator`, `pmml-session`
- Architecture: `docs/ARCHITECTURE.md` (crate topology, flow, ownership, concurrency)
- Benchmarks: `docs/BENCHMARK.md` (45 fixtures, Java `553 ms` vs Rust `68µs` cold, `1.22µs` vs `402 ns` single, `16.5M rows/s` Arrow 100k batched)
- Porting map: `docs/PORTING.md` + `docs/OWNERSHIP.tsv` (Java → Rust per-field)
- Upstream Java: https://github.com/jpmml/jpmml-evaluator (`pmml-evaluator:37,925 LOC`, `jpmml-model:22,405`)
- Upstream Model: https://github.com/jpmml/jpmml-model
- PMML 4.4 spec: https://dmg.org/pmml/v4-4/GeneralStructure.html (`pmml.xsd:4,490 lines`)
- Original plan: `docs/PLAN.md`

## Repo layout — gitflow

- `main` — releasable (protected)
- `development` — integration (protected)
- `chore/*` `feat/*` `fix/*` — work branches, PR → `development` → `main` (see `CONTRIBUTING.md`)

## Development

```sh
cargo fmt --check
cargo clippy --workspace -- -W clippy::pedantic -D warnings
cargo check --workspace
cargo test --workspace
cargo test -p pmml-session --test all_fixtures -- --nocapture # 45/45
cargo test --doc --workspace
cargo doc --workspace --no-deps
cargo bench -p pmml-bench --bench scoring -- --sample-size 30
```

MSRV `1.78`, `edition=2021`, `license=MIT OR Apache-2.0`.
