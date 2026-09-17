# Observability & Troubleshooting

> **Info:** This page covers failures that surface while scoring PMML in production. For thread safety and buffers, see [Concurrency & Memory](./concurrency.md). For the parser limits, see [Security & Hardening](./security.md).

Failures fall into two groups: a `PmmlError` variant raised on the cold path, and a `Value::Missing` that flows through the model on the hot path. The error table below names the first group, and the FAQ covers the second.

## Concepts

| Concept | Description |
| --- | --- |
| **PmmlError** | Unified error enum. The hot path never panics; the cold path validates XML, IR, and types. |
| **Value::Missing** | Explicit missing value carried by `JumpIfMissing`, never `Option<Value>`. |
| **Batch threshold 256** | Serial below 256 rows, `rayon par_chunks(256)` above. |
| **Active fields** | `MiningSchema.active_fields` sets what a CSV or `RecordBatch` header must contain. |
| **PmmlReader guard** | `MAX_DEPTH 512` and `MAX_FILE_BYTES 100 MB` run before `RawPmml` is allocated. |

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions, Value};
use pmmlruntime::session::batch::Batch;
use std::collections::HashMap;
let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, &std::fs::read("bench/pmml/DecisionTreeIris.pmml")?, SessionOptions::default())?;
let out = sess.run(&HashMap::<String, Value>::new() as &dyn Batch)?; // every field missing
# Ok::<(), Box<dyn std::error::Error>>(())
```

An empty input map still scores. The output follows `missingValueReplacement` instead of raising an error.

## Error variants

| Variant | Raised when | What to do |
| --- | --- | --- |
| **UnsupportedMarkup** | `ModelComposition`, `CenterFields`, or an unknown `*Model` tag; also 4 `ResultFeature` values without strict mode | Read the `verify_raw` message |
| **ParseError { context, message }** | Malformed XML or a `pmml.xsd` violation | Fix the export, then check `context` |
| **ValidationError** | Depth above 512, file above 100 MB, empty `MiningSchema` | Reduce nesting or split the file |
| **MissingField** | A `MiningField` names an absent `DataField` | Add the `DataField` to `DataDictionary` |
| **InvalidValue** | A value cannot coerce to the target `DataType` or `OpType` | Use `string_to_value` |
| **TypeError** | `Discrete` where `Continuous` is expected | Check the predicate typing |
| **Io** | `from_file` cannot read the path | Check the path and its permissions |
| **ArithmeticOverflow** | `checked_add` or `checked_mul` overflowed | Handle the strict mode path |
| **Other** | `anyhow` wrapper for Arrow and IO interop | Inspect the `source` chain |

> **Note:** Unknown `HashMap` keys are ignored, so an extra CSV column never breaks scoring.

## Diagnosing a failure

Work through the row that matches the symptom, then apply the smallest fix and record what you changed.

| Symptom | Check first | Try | Record |
| --- | --- | --- | --- |
| `ValidationError` depth | Nesting passes `MAX_DEPTH 512` on `Event::Start` | Split the PMML, or raise the cap after review | A new case in `hardening.rs` |
| `ValidationError` size | File above 100 MB before the parser exists | Compress the PMML, or stream it into `from_bytes` | The `MAX_FILE_BYTES` docs |
| `UnsupportedMarkup` | `ModelComposition`, `CenterFields`, or an unknown `*Model` tag | Export without that element | The `verify_raw` docs |
| `MissingField` | A `MiningField` names an absent `DataField` | Add the missing `DataField` | A `DataDictionary` entry |
| `InvalidValue` | A string cannot coerce to its `DataType` | Call `sess.string_to_value`, or fix the CSV | The `MiningSchema` |
| Wrong `predictedValue` | The label column was fed in as an input | Keep the label out of the map and compare after `build_output` | An `active_fields` check |
| Probabilities do not sum to 1 | The scored node carries no `ScoreDistribution`, or the `Output` is missing | Re-export with `OutputField feature="probability"` and leaf distributions | A probability check in CI |
| Batch slow at 100 rows | The row count stayed under 256, so it ran serial | Switch to `RecordBatch` and a 100k batch for 61 ns per row | `batch.len()` in the log |

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions, Value};
use pmmlruntime::session::batch::Batch;
use std::collections::HashMap;
let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, &std::fs::read("bench/pmml/DecisionTreeIris.pmml")?, SessionOptions::default())?;
let mut row = HashMap::new();
row.insert("Petal.Length".into(), Value::Missing); // an empty cell becomes Missing
let out = sess.run(&row as &dyn Batch)?.into_single().unwrap();
println!("Missing -> {:?}", out.get("predictedValue")); // follows the default branch
# Ok::<(), Box<dyn std::error::Error>>(())
```

That run produced a prediction without raising. Log `batch.len()` and `sess.num_active_fields()` on entry, because most reported problems turn out to be schema mismatches rather than engine errors.

```bash
cargo test --test hardening -- --nocapture
cargo test --test all_fixtures -- --nocapture
cargo fuzz run unmarshal -- -max_total_time=60
```

Those three commands cover depth, size, XXE, and `Send + Sync`.

> **Tip:** Validate an untrusted upload in a short-lived task, then hand the verified `Session` to hot scoring. You pay the XML cost once.

## FAQ

### Can I share a Session across threads?

Yes. `Session` is `Send + Sync` through `Arc<Ir>`. Call `run(&self)` concurrently, and `with_value_buffer` gives each thread a private slice.

```rust
use std::sync::Arc;
let sess = Arc::new(sess);
let h: Vec<_> = (0..4).map(|_| { let s = sess.clone(); std::thread::spawn(move || s.run as usize) }).collect();
```

You shared the session through `Arc` with no `&mut`.

### Can I pass unknown fields and keep scoring?

Yes. Unknown keys stay `Missing` in `materialize_row`, so vendor `Extension` columns are safe.

> **Warning:** A missing required field still runs the `MiningSchema` outlier and missing treatments before `eval_derived_fields`.

### How does a missing value become Value::Missing?

An empty string, the literal `"Missing"`, a null Arrow cell, and an absent key all become `Value::Missing` through `JumpIfMissing`.

```rust
assert_eq!(sess.string_to_value("Petal.Length", ""), Value::Missing);
```

### Can I score ModelComposition or CenterFields?

No. `verify_raw` returns `UnsupportedMarkup` for those two, for `BaselineRegressionModel`, and for any `*Model` tag the unmarshaller does not recognize. The other 19 model elements verify and score.

### Which CSV header do I need?

The header must list the `active_fields`. Unknown columns are ignored, and a missing active column becomes `Missing`.

### How do I pass a category?

Call `sess.symbol_id("setosa")` or `sess.string_to_value("Species", "setosa")`. A `RecordBatch` interns through `symbol_str_to_id`.

```rust
if let Some(sid) = sess.symbol_id("setosa") { let v = Value::Discrete(sid); }
```

The lookup allocates nothing.

### Can I pick serial or sharded per batch?

Not directly. `CpuProvider` runs serial below 256 rows, or below `threads*4`, and shards anything larger with `par_chunks(256)`. Pass a `HashMap` for one row, which costs 402 ns, and a `RecordBatch` for 100k, which costs 61 ns per row.

### How do I check that probabilities sum to 1?

Read the `Output` `probability(...)` values per class, sum them over the `BatchResult`, and assert the total is within 0.01 of 1.0. This holds when the scored node declares `ScoreDistribution`. The Iris fixture has none on its leaves, so its probabilities are all 0 and the check would fail there; run it on a model whose leaves carry distributions. Drift points at `build_output` or at the `ScoreDistribution` values written by `lower`.

```rust
use pmmlruntime::Value;
use std::collections::HashMap;
fn check_prob_sum(outs: &[HashMap<String, Value>]) -> Result<(), String> {
    for row in outs {
        let sum: f64 = ["setosa","versicolor","virginica"].iter().map(|cls| match row.get(&format!("probability({cls})")) { Some(Value::Continuous(c)) => *c, _ => 0.0 }).sum();
        if (sum - 1.0).abs() > 0.01 { return Err(format!("sum {sum:.3} != 1.0")); }
    }
    Ok(())
}
```

See [Validation & ModelVerification](../evaluation/validation.md) for the full gate.

### How do I find which guard rejected a file?

Match the `PmmlError` message. `ValidationError` with `"depth"` means `MAX_DEPTH 512` was exceeded, and with `"size"` it means `MAX_FILE_BYTES 100 MB` was exceeded. An `"unsupported markup"` prefix points at `verify_raw`.

```rust
match Session::from_bytes(&env, &bytes, SessionOptions::default()) {
    Ok(sess) => println!("ok: {} fields", sess.num_active_fields()),
    Err(e) if e.to_string().contains("depth") => eprintln!("depth guard: {e}"),
    Err(e) if e.to_string().contains("unsupported") => eprintln!("verify_raw: {e}"),
    Err(e) => eprintln!("other: {e}"),
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The message names the boundary, so you do not have to read the crate to find it. See [Security & Hardening](./security.md).

### Why did a categorical prediction change after lowering?

Compare `predictedValue` by decoding it with `sess.ir.symbol_names` rather than reading the raw `SymbolId`. A `SymbolId` is dense per `Ir` and stable after the `lower` topological sort, so a change means interning moved. Run `cargo test --test all_fixtures` to catch it.

```rust
let pred = out.get("predictedValue").and_then(|v| match v {
    Value::Discrete(sid) => sess.ir.symbol_names.get(sid).map(|s| s.as_str()),
    _ => None,
}).unwrap_or("");
assert_eq!(pred, "setosa");
```

Decoding a `Discrete` value costs no allocation.

> **Tip:** Log `batch.len()` and `sess.num_active_fields()` on entry. Most issues are schema mismatches, not engine errors.

## Next Steps

- [Security & Hardening](./security.md): enforce depth 512 and 100 MB before you diagnose anything else.
- [Concurrency & Memory](./concurrency.md): inspect the stack and heap paths, and `rayon` sharding.
- [Values, Fields & Types](../concepts/values.md): work with dense `FieldId` and `SymbolId` slices.
- [MiningSchema, DataDictionary & Output](../concepts/schema.md): fix missing-value and outlier treatments.

*Next: [Migrating from JPMML](./migration.md) → · Previous: [Concurrency & Memory](./concurrency.md)*
