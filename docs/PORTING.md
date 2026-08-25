# PORTING.md — PMML 4.4 → Rust (green-field, ONNX-by-design) v2 Batch

> Basado en `spec/pmml.xsd:4490` + `GeneralStructure.html:932` + `jpmml-evaluator-features.md:96`.
> No es transpilación mecánica JPMML -> Rust; es spec -> Rust con ideas ONNX Runtime.
> Cada regla requiere 2 revisiones adversariales.

| Java / JPMML pattern | Rust pattern (v2 `feat/session-cleanup` final) | Notas / pitfalls |
|---|---|---|
| `class PMML extends PMMLObject` + `JAXB` | `RawPMML` (pmml-xml) + `Ir` (pmml-ir) `Arc<Ir>` | Separar parseo frío (XML) de IR optimizado (FieldId, bytecode). No usar `quick-xml serde`. |
| `null` / `MissingValue` | `Value::Missing` enum, no `Option<Value>` en hot path (Value incluye Missing) | `Missing` es valor, no ausencia. Evita `Option<Value>` doble wrap. |
| `FieldName String` HashMap lookup por fila | `FieldId(u32)` array `values[field_id as usize]` + `AHashMap<String,FieldId>` hot, `Rodeo` solo `lower()` frío | `AHashMap::get(&str)` es zero-alloc via `Borrow<str>` — `Rodeo`/`Spur` eliminado de `Session` (era dead `rodeo,spur_to_id`). Hot es `u32` index. `Discrete string -> SymbolId(u32)`. |
| `instanceof` / `Visitor` 13 baterías | `enum ModelIr { Tree(TreeIr) }` + `match` + `lower` passes explícitos | Un pass por batería JPMML pero puro (no muta tree JPMML, crea `Ir`). |
| `Guava ImmutableBiMap` | `AHashMap<String,FieldId>` (hot) + `HashMap` (std, para `Mining`/`GeneralRegression` API) | `lasso::Rodeo` solo `lower()` — no `Session`. `bimap` no necesario. |
| `Guava RangeSet/RangeMap` | `RangeMap` solo donde `continuousDomain` exista (raro) | La mayoría `MiningField` no usa domain; no generalizar. |
| `Guava LoadingCache` | `bumpalo::Bump` arena `thread_local!` + `Arc<Ir>` cache | No cache Guava; IR es inmutable `Arc`. Arena reset por `run()`. Hot usa `with_value_buffer` stack 64 + `THREAD_VALUES`. |
| `Guava Interner` | `AHashMap` (hot) — `lasso` eliminado de `Session` | 5 hits JPMML -> `Rodeo` solo `lower()`. `Session` usa `ahash` 3× vs SipHash. |
| `Commons Math NormalDistribution/Erf/Mean` | `statrs 0.17`, `libm 0.2` | Gate `VerificationUtil` fixtures eps 1e-9. |
| `JAXB @XmlTransient/@XmlAdapter` | `quick-xml 0.37` pull `Reader` manual + hardened | Billion laughs, DTD disabled, depth 128, `memchr` fast. |
| `synchronized` / `Concurrent` | `Arc<Ir>` `Send+Sync`, `PmmlEnv` con `rayon` pool, `ExecutionProvider` sharding | `Session` es `Send+Sync` pero `run(&self)` usa `thread_local!` + `with_value_buffer` stack. Provider decide serial vs batched. |
| `Factory` / `Builder` | `SessionOptionsBuilder` + `Session::new(env, bytes, opts)` copy-on-build | `Ir` clon `Arc::clone` no deep copy. `GraphOptimizationLevel` para `EnableBasic` vs `EnableExtended`. |
| `defer` / `try-finally` | `Drop` + `Arena::reset` | Evitar leak `Evaluator.verify()` warm-up. |
| `int overflow checkedAdd` | `checked_add` + `PmmlError::ArithmeticOverflow` | Preservar `ArithmeticException` no wrap. |
| `assert(side_effect)` | `debug_assert!` ban si tiene side effect | JPMML `NodeResolver` tenía assert con `insert` -> ban clippy. |
| `Apply function switch` | `enum Op { CallBuiltin(BuiltinId) }` bytecode + `vm::eval` | No `match function_name: String` por fila. Tabla `BuiltinId -> fn`. |
| `Tree Node Box<dyn Predicate>` | `Vec<NodeIr>` flat contiguo + `PredicateIr` enum | Branchless, prefetch, `match` table jump. |
| `HasXxx<E>` traits 30 files | `HasXxx` traits minimal solo donde spec lo requiere | No port 1:1 de `org.dmg.pmml Has*`; colapsar. |
| `Evaluator.evaluate(Map<String,?>)` per row | `Session::run(HashMap<String,Value>)` single + `Batch` trait (`Vec<HashMap>` RowMajor vs `RecordBatch` Columnar) + `ExecutionProvider::eval_batch` | ONNX `OrtValue` + `IoBinding` style. `Session` materializa `Value[FieldId]`, provider hace sharding. `run_batch` → `BatchCtx` + `provider.eval_batch`. Threshold `<256` serial. |
| `Table` / `InlineTable` per row | `arrow::RecordBatch` columnar + `Batch` `Columnar` | `arrow 53` + `RecordBatch` zero-copy. `HashMap` para 1 fila/compat, `RecordBatch` para >10k (16.5M rows/s). No solo Arrow — single row `HashMap` 402ns < Arrow >1µs. |
| `ExecutionProvider` (new) | `trait ExecutionProvider { eval_row, eval_batch, preferred_format }` | `CpuSerial` sequential, `CpuBatched` rayon `par_chunks(256)` con fallback `<256`. `BatchCtx` lleva `output_fields` + `symbol_names_vec`. |

Ver `OWNERSHIP.tsv` para per-field ownership (`Arena` vs `Arc` vs `Box`).
