# Architecture — pmmlruntime

> `0.1.0` · single crate `pmmlruntime` · `13,642` LOC Rust · `pmml.xsd:4,490` · `BENCHMARK.md` tables for 45 fixtures

This document is the contributor-facing internals. For API contracts see `cargo doc --open`; for porting map see `docs/PORTING.md` + `docs/OWNERSHIP.tsv`.

## 1. Crate topology — single crate with modules (was 9 crates, now merged ONNX Runtime-style)

Previously a 9-crate workspace (`pmml-core`, `pmml-xml`, `pmml-ir`, `pmml-evaluator`, `pmml-session`, …).
Now a **single crate** `pmmlruntime` with modules — one `cargo add pmmlruntime` and one `cargo doc -p pmmlruntime` page:

```
pmmlruntime/
├─ base        # zero-cost types, arena, errors. No XML, no IR. Hot path foundation (was pmml-core).
├─ xml         # Hardened quick-xml 0.37 → RawPmml (5758 LOC, 1:1 with pmml.xsd). Cold only (was pmml-xml).
├─ ir          # Lower RawPmml → Ir (optimized). Interner (Rodeo cold) + verify (was pmml-ir).
├─ engine      # Pure evaluation on &[Value]: mining_schema, 12 models, predicate, output, targets, transform/vm, simd (was pmml-evaluator).
├─ session     # ONNX-style Session API: PmmlEnv + Session + Batch + ExecutionProvider. Primary user API (was pmml-session).
├─ ffi         # C ABI (onnxruntime_c_api.h parity): PmmlEnv/Session opaque, PmmlCreate/Release (was pmml-ffi).
├─ python      # pyo3 0.22 extension-module (future PySession). Stub now (was pmml-python).
├─ cli         # clap CLI: pmml-runtime inspect/run/verify (was pmml-cli, now binary in workspace).
└─ bench       # criterion + large_trial (10k/100k/1M/10M Arrow scaling) (was pmml-bench).
```

```
use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
// re-exports also at crate root: use pmmlruntime::{Value, Session, PmmlEnv};
```

Workspace `Cargo.toml` `resolver=2`, `edition=2021`, `rust-version=1.78`, `license=MIT OR Apache-2.0`, `members = ["crates/pmmlruntime"]`.

## 2. Data & control flow

```
                         cold                                          hot
  bytes: &[u8] ──► xml::unmarshal ──► RawPmml ──► ir::lower ──► Ir ──► Session::from_ir ──► Arc<Ir>
      │                  │                    │              │             │            │
      │  quick-xml 0.37  │  DTD/XXE blocked  │ 304 elem    │ Rodeo cold  │ verify_ir  │ Arc clone not deep copy
      │  MAX_DEPTH 512   │  100 MB cap       │ 12 models   │ FieldId u32 │            │ AHashMap<String,FieldId> hot
      │  trim_text       └──────────────────┘              │ SymbolId u32│            │ symbol_names_vec dense
      └────────────────────────────────────────────────────┘             └────────────┘
                                                        Session::run(HashMap<String,Value>)
                                                                   │
                                                          with_value_buffer (stack 64 L1-hot)
                                                                   │
                                                          Value[FieldId] = [Missing; needed]
                                                                   │
                                         ┌─────────────────────────┴─────────────────────────┐
                                         │              ExecutionProvider                     │
                                         │  CpuSerial (sequential loop)  │  CpuBatched (rayon par_chunks(256)) │
                                         │  with_value_buffer reuse      │  <256 fallback serial               │
                                         └───────────────────────────────┴────────────────────────────────────┘
                                                                   │
                                                          eval_derived_fields(&[DerivedFieldIr], &mut [Value])
                                                          │  DAG topo, Op bytecode, SmallVec predicates
                                                          │
                                                          evaluate_model(&IrModel, &[Value]) -> Value
                                                          │  Tree flat Vec<NodeIr> branchless, Regression intercept+coeff*powi → normalization, etc.
                                                          │
                                                          Output + Targets
                                                          │  build_output(26 ResultFeature, 4 unsupported → Missing)
                                                          │
                                                          HashMap<String,Value> { predictedValue, target_name, Probability_* }

Batch path (ONNX style):
  Vec<HashMap<String,Value>> or RecordBatch ──► Batch trait ──► BatchCtx { name_to_id, col_map, output_fields } ──► provider.eval_batch ──► BatchResult
     RowMajor (JPMML compat, 402ns single)         │  object-safe Send+Sync      no per-row alloc         rayon shard          Rows or Columnar
     Columnar Arrow (61 ns/row 100k)               │  materialize_row → values[FieldId] = Value
```

## 3. Ownership & lifetimes — per-field via OWNERSHIP.tsv

| Component | Owned by | Shared? | Notes |
|---|---|---|---|
| `Ir` | `Arc<Ir>` in `Session` | `Send+Sync` clone is `Arc::clone` (not deep copy) | `field_names: HashMap<FieldId,String>`, `symbol_names: HashMap<SymbolId,String>`, `symbol_names_vec: Vec<String>` dense, `field_metas: Vec<FieldMeta>` |
| `RawPmml` | `lower(RawPmml) -> Ir` consumes | — | Only during cold lower, then dropped |
| `Interner::rodeo` | `Interner` (cold) | — | `!Sync` effectively single-threaded per `lower`; not in `Session` hot path (dead `rodeo,spur_to_id` removed, kept only in `Interner`) |
| `Session::name_to_id` | `Session` (hot) | `AHashMap<String,FieldId>` per `Session` (ahash, `Borrow<str>` zero-alloc) | Also `name_to_id_std: HashMap` for `GeneralRegression`/`Mining` API compat |
| `Session::ir` | `Arc<Ir>` | `Send+Sync` | `Session` is `Send+Sync` but `run(&self)` uses `thread_local! THREAD_VALUES` + stack `64` via `with_value_buffer` |
| `BatchCtx` | per-batch stack | — | `{name_to_id, symbol_str_to_id, Ir refs, col_map: Vec<(FieldId,usize)>}` no per-row alloc |
| `BumpArena` | per-`par_iter` chunk owned | `Send` (owns `Bump`), not `Sync` | `THREAD_ARENA` is `thread_local!` for serial |
| `THREAD_VALUES` | `thread_local! RefCell<Vec<Value>>` | — | Serial `run` overflow >64, reused |
| `Value: Missing` | `Value` variant | — | Not `Option<Value>` double wrap |

See `docs/OWNERSHIP.tsv` for per-field `struct field type java_owner rust_owner notes`.

## 4. Concurrency model

- `Session` is `Send+Sync` (`Arc<Ir>` + `SessionOptions` + `Box<dyn ExecutionProvider>` + `AHashMap` which is `Send+Sync`).
- `Session::run` is `&self` and uses `with_value_buffer` which is `thread_local!` + stack, so concurrent `run` on same `Session` from multiple threads is safe (no `&mut`).
- `ExecutionProvider::eval_batch` is trait object `Send+Sync` + `Sync` for `rayon::par_iter`. `CpuSerial` loops sequentially; `CpuBatched` shards via `par_chunks(256)` (with `with_min_len 256`-like logic). Threshold `<256` (`batch.len() < 256 || threads*4`) falls back to serial to avoid spawn cost (~100µs > 400ns work per row). See `BENCHMARK.md §3`.
- `rayon` pool is per-`PmmlEnv` future (currently global pool).
- `BumpArena` is `Send` so it can be moved into rayon threads; never shared `&self` across threads without `&mut`. Verified `unsafe impl Send`.
- `LAG_BUFFER` in `engine::transform::vm` is `thread_local!` `RefCell<HashMap<FieldId, VecDeque<Value>>>` cap `128`, so `Lag` builtin is per-thread, not cross-batch shared (sequence test uses single thread).

## 5. Storage & serialization boundaries

- **XML**: `quick-xml 0.37` pull `Reader` `trim_text(true)` + `expand_empty_elements = true`, DTD not expanded (XXE blocked), depth `512`, file `100 MB`. Replaces JPMML `jakarta.xml.bind` `XmlTransient`/`XmlJavaTypeAdapter` 267 hits. `Raw*` structs are owned `String`/`Vec` not zero-copy (cold).
- **IR**: `Ir` is `Arc` immutable, flat `Vec<NodeIr>` (branchless, `match` table jump), `DerivedFieldIr` `Vec<Op>` bytecode owned by `Ir`, `SmallVec<[PredicateIr;4]>` for node predicates.
- **Arrow**: `arrow 53` `RecordBatch` zero-copy `Float64Array`/`StringArray`. `HashMap` for 1-row compat, `RecordBatch` for `>10k` (16.5M rows/s). `TableLocator` placeholder returns empty batch with schema (graceful). CSV `arrow::csv::Reader` → `RecordBatch` → `run_batch` → `arrow::csv::Writer`.
- **SerDe**: `serde` for `Raw*` (future), `serde_json/yaml` for CLI, `rmp-serde` for Kryo parity (not yet, thin).
- **Python**: `pyo3 0.22` `extension-module`, `maturin` wheel, `PySession` will wrap `Box<Session>` + `PyDict` ↔ `Value`.

## 6. Performance-sensitive paths — targets vs measured

| Path | Target | Measured (release, Iris 5 nodes, i7-12700 1.78) | Technique |
|---|---|---|---|
| Cold `Session::from_bytes` | `68µs` | `68.8µs` `bench_all` | `quick-xml` 0.37 + `lower` + `verify_ir`, no JAXB Visitors |
| Single `Session::run` | `≤800 ns` | `402 ns` criterion `30 samples` / `393 ns` bench_all 10k | `HashMap<String,Value>` → `with_value_buffer` stack 64 + `AHashMap::get(&str)` ahash `3×` + branchless flat `Vec<NodeIr>` |
| `run_batch` 1k sequential | `≤350µs` | `336µs` `1k` `2.97M rows/s` | `CpuSerial` loop `with_value_buffer` reuse, no rayon |
| `run_batch_arrow` 1k | `≤250µs` | `249µs` `4.0M rows/s` | `RecordBatch` `Float64Array` col_map + `with_value_buffer`, no per-row HashMap |
| `run_batch_arrow` 100k batched | `11× Java 696 ns` | `61 ns/row` `16.5M rows/s` `CpuBatched` | `rayon` `par_chunks(256)` over `par_chunks`, `thread_local Vec<Value>` |
| `cargo bench --bench scoring` | — | `criterion` html `target/criterion` | `measurement-time 2 --warm-up-time 1` |

Gate `cargo bench -p pmml-bench -- --sample-size 30` must be `≤800 ns` single, `≤500µs` batched (now passes via Arrow).

## 7. Important invariants — contributor must preserve

- `Ir.field_names` contains every `FieldId` in `MiningSchemaIr.active_fields` + `target_field` + `DerivedFieldIr`. `lower` asserts via `get_or_intern_field` single source of truth.
- `Ir.symbol_names` + `symbol_names_vec` (dense `Vec<String>` len `max_symbol_id+1`) must agree; `Session` builds both from `Ir`.
- `DerivedFieldIr` DAG is topologically sorted in `lower`; `eval_derived_fields` assumes sorted order (no cycle check hot).
- `Missing` is value not absence — `Value::Missing` after `MiningSchema` `invalid/outlier/missing` handling, `Op::JumpIfMissing` branches.
- `PmmlError::UnsupportedMarkup` for `AnomalyDetectionModel`/`BaselineModel`/`BayesianNetwork`/`GaussianProcess`/`Sequence`/`Text`/`TimeSeries`/`ModelComposition`/`CenterFields` — must not change to `InvalidValue`.
- `Session::max_field_id` is `max(values field_id) +1` `max 16`, `with_value_buffer` needed = `max_field_id.max(ir.num_fields()+4)`. Out-of-bounds `FieldId` is ignored not panic (unknown field → skip).
- `LAG_BUFFER` cap `128` per `FieldId`, `depth 512` for XML, `file 100 MB` — hardening invariants, `miri` + `cargo fuzz` must hold.

## 8. Extension points

- **New model**: add `ModelIr::New(NewIr)` + `RawNewModel` in `xml` `unmarshal`, `lower` arm, `engine/models/new.rs` `evaluate_new`, match in `Session::from_ir` target_name + output_fields, provider `eval_row` dispatch, `verify_raw` not `UnsupportedMarkup`.
- **New `BuiltinId`**: add variant to `ir::ir::BuiltinId`, `engine::transform::builtin::builtin_by_name` alias, `eval_builtin` arm (`statrs`/`libm`/`chrono`), `vm::eval` `CallBuiltin` arity.
- **New `ResultFeature`**: add to `base::field::ResultFeature` `FromStr`/`is_unsupported`, `engine::output::build_output_with_context` match, `Session::run` output mapping for `Scorecard` probabilities.
- **New `ExecutionProvider`**: implement `trait ExecutionProvider { eval_row, eval_batch, preferred_format }`, register in `Session::from_ir` `match options.execution_provider`, feature-flag `crate::batch::Batch` sharding.

## 9. Trade-offs & rejected alternatives

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| XML | `quick-xml 0.37` pull `Reader` manual hardened | `serde` `quick-xml serialize` + `XJC` generated | PMML XSD 304 elements, mixed Attribute/Element, `Extension` vendor payloads; `serde` can't express `pmml.xsd:4,490` ordering, `quick-xml` gives DTD/XXE control, 68µs cold vs JAXB `553 ms` |
| `LoadingCache` | `bumpalo::Bump` arena `thread_local!` + `Arc<Ir>` cache | `moka` Guava `LoadingCache` port | IR is immutable `Arc`, no cache invalidation; arena reset per `run` keeps capacity, `miri` clean, no `Drop` leak like `SSL_SESSION` 6.5KB/call in Bun |
| `BiMap` | `AHashMap<String,FieldId>` hot + `HashMap` std API | `bimap` crate | Hot is `u32` index `values[field_id]` not `BiMap`; `lasso::Rodeo` only `lower()` cold, `AHashMap::get(&str)` zero-alloc via `Borrow<str>` `3×` vs SipHash |
| `RangeMap` | `RangeMap` only where `continuousDomain` exists (rare) | Generic `rangemap` for all `MiningField` | Most `MiningField` no domain; no generalize |
| `Visitor` 13 batteries | `enum ModelIr { Tree(TreeIr) }` + `match` + `lower` passes explicit | JPMML `Visitor` mutation of `PMMLObject` tree | Purity (no muta tree), `cargo check` crate-by-crate, swarm per-file without `stash` |
| Batch | `Batch` trait `RowMajor Vec<HashMap>` + `Columnar RecordBatch` (provider picks) | Only Arrow | Single row `HashMap` 402ns < Arrow >1µs + schema agreement; `Collection`/`List` (Association) and Python `dict` map naturally to `HashMap` |
| Model strategy | Option A port `pmml-model` to Rust (this repo) | Option B JNI bridge `jni` crate | Removes JVM forever, single binary, WASM-ready, MIT/Apache-2.0 not AGPL; JNI keeps XML correctness for free but needs JVM at runtime |
| License | `MIT OR Apache-2.0` (workspace) | `TBD` (README old) + upstream `AGPL-3.0` dual BSD | Transpilation ≠ relicense; green-field port can be MIT/Apache-2.0 before first code commit (now decided) |
| Crate layout | single `pmmlruntime` with `base/xml/ir/engine/session` modules | 9-crate workspace `pmml-core/xml/ir/evaluator/session/...` | ONNX Runtime inspiration: one `cargo add pmmlruntime`, one `cargo doc` page, <20k LOC; easier for users, workspace+facade is also valid but `pmml-*` heritage is `jpmml-evaluator` clone; `publish=false` heritage is redundant |

Ver `OWNERSHIP.tsv` for per-field ownership; `BENCHMARK.md` for Java vs Rust tables; `PLAN.md` for Bun anchor `535k Zig` → `50k hand + 20k generated` mechanical.
