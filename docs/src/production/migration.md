# Migrating from JPMML

> **Info:** This page covers a move from JPMML-Evaluator to pmmlruntime for services that need to drop the JVM. For the parser limits, see [Security & Hardening](./security.md). For lifecycle details, see [Session, Env & Lifecycle](../concepts/session.md).

The PMML file does not change when you migrate. You map one JPMML class to one Rust type and keep the scoring semantics.

## Why migrate

JPMML-Evaluator is the reference evaluator. It loads models with JAXB, caches them in a Guava `LoadingCache`, and ships under `AGPL-3.0`. Its cold path pays for JVM class loading, which measured 8757 µs on the reference machine. pmmlruntime parses the same `pmml.xsd` vocabulary with `quick-xml 0.37` in 51.7 µs, scores one row from a stack buffer in 463 ns, and ships under `Apache-2.0`.

A migration buys you a zero-JVM binary, a deterministic hot path, columnar batches, and the hardened parser. You keep the same 52 fixtures at parity.

## What changes, measured

Four signals decide whether a swap is worth it. Each one is quoted only when the two 1-σ intervals do not overlap.

| Signal | JPMML | pmmlruntime | Gap | Where it matters |
| --- | --- | --- | --- | --- |
| **Cold start** | 8757 ±754 µs | 51.7 ±10.8 µs | **169×** | Serverless scale-out, where `from_bytes` runs per instance |
| **Hot single row** | 4562 ±127 ns | 463 ±87 ns | **9.9×** | Request path with a p99 under 1 µs |
| **Ensemble row** | 12267 ±958 ns | 438 ±99 ns | **28×** | `MiningModel`, where the JPMML transpiler does not apply |
| **Batch, 10k rows** | ~4.5 µs per row | 428 ns per row | **10.5×** | ETL jobs, where `RecordBatch` beats a per-row loop |

## Concepts

| Concept | Description |
| --- | --- |
| **PmmlEnv** | Replaces the JVM bootstrap as an `Arc` handle per process. |
| **Session** | Replaces `Evaluator`: immutable `Arc<Ir>` with a `Send + Sync` `run(&self)`. |
| **Value** | Replaces `FieldValue`: a `Copy` enum of `Continuous`, `Discrete`, or `Missing`. |
| **Batch** | Replaces the per-row `evaluate` loop with one `run(&dyn Batch)` for rows and columns. |
| **Ir** | Replaces the Guava `LoadingCache`: `Arc<Ir>` plus a `BumpArena` per chunk. |

## Mapping JPMML to pmmlruntime

| JPMML (Java) | pmmlruntime (Rust) | Notes |
| --- | --- | --- |
| `LoadingModelEvaluatorBuilder.load(file).build()` | `Session::from_bytes(&env, &bytes, opts)` | `quick-xml`, `verify_raw`, `lower`, `verify_ir` |
| `InputField.prepare(String)` | `Value::Continuous`, `Discrete(SymbolId)`, or `sess.string_to_value` | `Missing` is explicit, never null |
| `Evaluator.evaluate(Map)` | `sess.run(&hashmap as &dyn Batch)` | `into_single()` for one row |
| `List<Map>` loop | `sess.run(&vec as &dyn Batch)` or `&RecordBatch` | Serial below 256 rows, else `rayon par_chunks(256)` |
| `Guava LoadingCache` | `Arc<Ir>` plus `AHashMap` | No invalidation, and `BumpArena` resets per chunk |
| `JAXB` with XJC | `quick-xml 0.37` pull parser | 68 µs against roughly 8 ms, and XXE safe |
| `AGPL-3.0` | `Apache-2.0` | `resolver=2`, `edition=2021` |
| `InMemoryTranspiler` | No transpiler, `engine::mining` sums segments | `GradientBoosterTest.pmml` moves from N/A to pass |

The result is one binary in place of a JVM.

## Migration steps

### Step 1: Load the model

JPMML:

```java
LoadingModelEvaluatorBuilder b = new LoadingModelEvaluatorBuilder();
b.load(new File("DecisionTreeIris.pmml"));
Evaluator evaluator = b.build(); evaluator.verify();
```

pmmlruntime:

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions};
let env = PmmlEnv::new();
let bytes = std::fs::read("bench/pmml/DecisionTreeIris.pmml")?;
let sess = Session::from_bytes(&env, &bytes, SessionOptions::default())?;
```

`Session::from_bytes` replaces the builder. See [Session, Env & Lifecycle](../concepts/session.md).

### Step 2: Prepare inputs

JPMML:

```java
Map<String, FieldValue> args = new HashMap<>();
for (InputField f : evaluator.getInputFields()) {
    FieldValue v = f.prepare("1.4"); // String to FieldValue
    args.put(f.getName(), v);
}
```

pmmlruntime:

```rust
use std::collections::HashMap;
use pmmlruntime::Value;
let mut input = HashMap::new();
input.insert("Petal.Length".into(), Value::Continuous(1.4));
let sid = sess.symbol_id("setosa").unwrap();
input.insert("Species".into(), Value::Discrete(sid));
let v = sess.string_to_value("Petal.Length", "1.4");
```

A field with no value becomes `Value::Missing` rather than `null`.

### Step 3: Score a row

JPMML:

```java
Map<String, ?> result = evaluator.evaluate(args);
Object pred = result.get(evaluator.getTargetField().getName());
```

pmmlruntime:

```rust
use pmmlruntime::session::batch::Batch;
let out = sess.run(&input as &dyn Batch)?.into_single().unwrap();
let pred = out.get("predictedValue").unwrap();
```

One method covers single rows, batches, and Arrow input.

### Step 4: Batch

JPMML evaluates each row in a loop. pmmlruntime shards the whole batch once:

```rust
let batch = vec![input.clone(); 1000];
let rows = sess.run(&batch as &dyn Batch)?.into_rows(); // serial below 256 rows, else rayon
let rows2 = sess.run(&record_batch as &dyn Batch)?.into_rows(); // 61 ns per row at 100k
```

The per-row garbage collection cost is gone, because `Value` is a `Copy` type of 16 bytes.

### Step 5: Share the session

`Arc<Ir>` plus a `BumpArena` reset per chunk replaces the Guava `LoadingCache`. Share one `Arc<Session>`:

```rust
use std::sync::Arc;
let sess = Arc::new(sess);
let handles: Vec<_> = (0..8).map(|_| {
    let s = sess.clone();
    std::thread::spawn(move || {
        use pmmlruntime::session::batch::Batch;
        use std::collections::HashMap;
        use pmmlruntime::Value;
        let mut row = HashMap::new();
        row.insert("Petal.Length".into(), Value::Continuous(1.4));
        s.run(&row as &dyn Batch).unwrap()
    })
}).collect();
for h in handles { h.join().unwrap(); }
```

Eight threads shared the session without `&mut`, and none of them reloaded the model.

The numbers below come from five runs on an i7-12700, recorded in `bench/BENCHMARK.md`.

| Path | JPMML | pmmlruntime | Speedup |
| --- | --- | --- | --- |
| Cold load, Iris | 8757 µs ±754 | 51.7 µs ±10.8 | 169× |
| DecisionTree, single row | 4562 ns ±127 | 463 ns ±87 | 9.9× |
| GradientBooster, single row | 12267 ns ±958 | 438 ns ±99 | 28× |

> **Warning:** The JPMML transpiler fails on `GradientBoosterTest.pmml` because of strict `OutputField@dataType` validation. pmmlruntime scores that file.

> **Tip:** Reproduce these numbers with the 52 fixtures in `bench/pmml/` and `cargo test`.

## Beyond one-to-one

### One env, many models

Reuse one `PmmlEnv` for many `Session` values when you run an A/B canary or isolate tenants.

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions};
let env = PmmlEnv::with_name("prod-v2");
let iris = Session::from_bytes(&env, &std::fs::read("bench/pmml/DecisionTreeIris.pmml")?, SessionOptions::default())?;
let boosted = Session::from_bytes(&env, &std::fs::read("bench/pmml/GradientBoosterTest.pmml")?, SessionOptions::default())?;
let env2 = env.clone(); // atomic increment
# Ok::<(), Box<dyn std::error::Error>>(())
```

| Pattern | JPMML | pmmlruntime | When to use it |
| --- | --- | --- | --- |
| **A/B models** | Two `Evaluator` values with `LoadingCache` | One `PmmlEnv` plus two `Arc<Session>` | Canary on `GradientBooster` |
| **Multi-tenant** | One `Evaluator` per tenant | `PmmlEnv::with_name("tenant-x")` | Isolate `FieldId` per tenant |
| **Session clone** | `LoadingCache.get` | `Arc<Session>::clone()` | 20 ns atomic increment |

### Batch strategies

| Strategy | JPMML | pmmlruntime | When to use it |
| --- | --- | --- | --- |
| **Sequential** | `for (Map m : list) evaluator.evaluate(m)` | `for row in &vec { sess.run(&row) }` | Fewer than 100 rows |
| **Multithreading** | `ExecutorService` | `Arc<Session>` plus `thread::spawn` | A pool already exists |
| **Columnar** | Not available | `sess.run(&RecordBatch)` with `par_chunks(256)` | More than 1k rows, at 61 ns per row |

See [Concurrency & Memory](./concurrency.md).

### Output fields and thresholds

`OutputField` survives the move. Read `out.get("probability(setosa)")` after `build_output`, then gate on a threshold such as `accuracy >= 0.95`, as shown in [Validation & ModelVerification](../evaluation/validation.md).

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions, Value};
use pmmlruntime::session::batch::Batch;
use std::collections::HashMap;
let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, &std::fs::read("bench/pmml/DecisionTreeIris.pmml")?, SessionOptions::default())?;
let mut row = HashMap::new();
row.insert("Petal.Length".into(), Value::Continuous(1.4));
let out = sess.run(&row as &dyn Batch)?.into_single().unwrap();
let prob = match out.get("probability(setosa)") { Some(Value::Continuous(c)) => *c, _ => 0.0 };
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Migration failures

| Symptom | Check first | Try | Record |
| --- | --- | --- | --- |
| `ValidationError` depth | Nesting passes `MAX_DEPTH 512` | Split the PMML, or raise the cap | A case in `hardening.rs` |
| `UnsupportedMarkup` | `ModelComposition`, `CenterFields`, or an unknown `*Model` tag | Export without that element | The `verify_raw` docs |
| `predictedValue` drift | `FieldId` or `ScoreDistribution` drift | Run `cargo test --test all_fixtures` | The `lower` changes |
| Batch slow at 100 rows | The batch stayed under 256 rows and ran serial | Move to `RecordBatch` for 100k rows | `batch.len()` in the log |

```bash
cargo test --test all_fixtures -- --nocapture
cargo test --test hardening -- --nocapture
```

## FAQ

### Do I need to re-export the PMML?

No. `DataDictionary`, `MiningSchema`, and `Output` all survive the move. Only the runtime changes, from `LoadingModelEvaluatorBuilder` to `Session::from_bytes`. Confirm it with `diff expected_jpmml.csv out_rust.csv`.

### Can I run both runtimes during a canary?

Yes. Keep JPMML behind JNI or in a sidecar, and run pmmlruntime through `Arc<Session>` in the same group. Compare `predictedValue` by decoding `symbol_names`, then gate on `accuracy >= 0.95`.

### What replaces the Guava LoadingCache?

`Arc<Ir>` plus a `BumpArena` reset per chunk. Nothing invalidates: a `Session` is immutable after `lower`, and the arena keeps its capacity instead of collecting.

### Can both runtimes read the same CSV?

Yes. The header must match `active_fields`. Unknown columns are ignored, and a missing active column becomes `Value::Missing` through `JumpIfMissing`.

### How does a missing value differ from null?

`Missing` is an explicit variant. An empty string, the literal `"Missing"`, and a null Arrow cell all become `Missing` and then flow through `missingValueReplacement`.

### How do I reproduce the 169× cold and 28× hot numbers?

Run `cargo build --release --example bench_real` and repeat the measurement five times, as shown in [Performance](../evaluation/performance.md). Quote the result only when the Rust mean plus standard deviation stays under the Java mean minus standard deviation.

> **Note:** `RawPmml` covers 304 elements that map to `pmml.xsd` 1:1, and it is dropped after `lower`. Only `Arc<Ir>` survives.

## Next Steps

- [Security & Hardening](./security.md): harden the parser with the 100 MB cap, depth 512, and blocked XXE.
- [Concurrency & Memory](./concurrency.md): share `Session` through `Arc` and `rayon`.
- [Quickstart: Score Iris in 5 Minutes](../getting-started/quickstart.md): score the same fixtures end to end.
- [Architecture Overview](../internals/architecture.md): trace `base → xml → ir → engine → session`.

*Next: [Architecture Overview](../internals/architecture.md) → · Previous: [Observability & Troubleshooting](./troubleshooting.md)*
