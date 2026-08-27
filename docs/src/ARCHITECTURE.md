# Architecture — pmmlruntime `0.1.0`

> Single crate. `base -> xml -> ir -> engine -> session -> ffi/python`. Cold path builds `Ir`; hot path scores `Value` slices.

This doc is the contributor-facing internals. API contracts are in `cargo doc --open -p pmmlruntime`.

## 1. Crate topology

```mermaid
graph TD
    base["base<br>Value / FieldId / SymbolId<br>DataType / OpType<br>BumpArena / PmmlError"]
    xml["xml<br>quick-xml 0.37 to RawPmml<br>MAX_DEPTH 512 / 100 MB / XXE blocked"]
    ir["ir<br>lower to Ir<br>Interner Rodeo / verify"]
    engine["engine<br>pure eval on slices<br>19 models + vm + simd"]
    session["session<br>PmmlEnv + Session + Batch<br>Cpu ExecutionProvider"]
    ffi["ffi<br>C ABI opaque handles"]
    python["python<br>pyo3 0.22 extension-module"]

    base --> xml
    base --> ir
    base --> engine
    base --> session
    xml --> ir
    ir --> engine
    ir --> session
    engine --> session
    session --> ffi
    session --> python

    style base fill:#0b7285,stroke:#083d4a,color:#fff
    style xml fill:#36404a,stroke:#222,color:#fff
    style ir fill:#36404a,stroke:#222,color:#fff
    style engine fill:#36404a,stroke:#222,color:#fff
    style session fill:#e8590c,stroke:#a8450a,color:#fff
```

Single crate `pmmlruntime` (`Cargo.toml` `resolver=2`, `edition=2021`, `rust-version=1.78`, `Apache-2.0`):

```
crates/pmmlruntime/src/
├─ base/         # zero-cost types, arena, errors — no XML, no IR
│  ├─ value.rs   # Value / FieldId / SymbolId
│  ├─ field.rs   # DataType / OpType / MiningFunction / ResultFeature
│  ├─ arena.rs   # BumpArena
│  └─ error.rs   # PmmlError
├─ xml/          # cold only
│  ├─ reader.rs  # PmmlReader — hardened quick-xml
│  └─ unmarshal.rs # -> RawPmml (304 PMML elements, 1:1 with pmml.xsd)
├─ ir/           # lower + verify + Interner
│  ├─ ir.rs      # Ir, FieldMeta, Op, ModelIr (19 variants)
│  ├─ lower.rs   # RawPmml -> Ir (topo sort DerivedField, FieldId/SymbolId)
│  ├─ verify.rs  # verify_raw / verify_ir
│  └─ intern.rs  # Rodeo interner (cold only)
├─ engine/       # pure evaluation
│  ├─ predicate.rs, mining_schema.rs, output.rs, targets.rs, simd.rs
│  ├─ models/    # 19 evaluators (tree, regression, mining, ...)
│  └─ transform/ # vm.rs, builtin.rs, mapvalues.rs, discretize.rs
├─ session/      # primary API
│  ├─ env.rs     # PmmlEnv (Arc inner)
│  ├─ options.rs # SessionOptions
│  ├─ session.rs # Session (+ with_value_buffer)
│  ├─ batch.rs   # Batch / BatchCtx / BatchResult
│  ├─ arrow.rs   # RecordBatch helpers
│  ├─ input.rs   # string_to_value
│  └─ providers/ # cpu.rs — unified Cpu provider
├─ ffi/          # C ABI
└─ python/       # pyo3 placeholder (feature-gated)
```

## 2. Data & control flow — cold vs hot

```mermaid
flowchart LR
    subgraph Cold["Cold path - once per model"]
        A["bytes: slice u8"] --> B["xml::unmarshal<br>quick-xml 0.37<br>depth 512 / 100 MB / DTD blocked"]
        B --> C["RawPmml<br>owned Strings/Vecs<br>304 elements"]
        C --> D["ir::verify_raw"]
        D --> E["ir::lower<br>Rodeo FieldId/SymbolId<br>topo sort DerivedField<br>flat Vec NodeIr"]
        E --> F["ir::verify_ir"]
        F --> G["Arc Ir<br>immutable"]
    end
    subgraph Hot["Hot path - per row / per batch"]
        G --> H["Session::run dyn Batch"]
        H --> I["with_value_buffer<br>64 stack 1KB<br>large thread_local Vec"]
        I --> J["Value FieldId<br>Missing-initialized"]
        J --> K["Batch::materialize_row<br>HashMap get or Arrow col_map"]
        K --> L["ExecutionProvider::eval_row<br>eval_derived_fields VM<br>evaluate_model"]
        L --> M["output + targets<br>predictedValue / probability"]
        M --> N["HashMap String Value"]
    end
```

Key split: `RawPmml` is dropped after `lower`. `Ir` is `Arc` and never mutated. `Session` is `Send+Sync` and holds only `Arc<Ir>` + lookup caches.

### End-to-end timeline

```mermaid
sequenceDiagram
    participant U as User
    participant S as Session
    participant X as xml
    participant I as ir
    participant P as Cpu provider
    participant E as engine

    U->>S: Session::from_bytes(env, bytes, opts)
    S->>X: unmarshal(bytes)
    X-->>S: RawPmml
    S->>I: verify_raw(raw)
    S->>I: lower(raw)
    I-->>S: Ir
    S->>I: verify_ir(ir)
    S-->>U: Session with Arc Ir

    U->>S: run(hashmap as dyn Batch)
    S->>S: with_value_buffer(needed)
    S->>P: eval_batch(ir, batch, ctx)
    P->>E: eval_derived_fields(derived, values)
    P->>E: evaluate_model(model, values)
    E-->>P: Value predicted
    P->>P: build_output + targets
    P-->>S: BatchResult Rows
    S-->>U: predictedValue
```

## 3. Value representation — why `Value` slices

```mermaid
classDiagram
    class FieldId {
        +u32 id
        +as_usize() usize
    }
    class SymbolId {
        +u32 id
        +interned discrete
    }
    class Value {
        <<enum>>
        Continuous(f64)
        Discrete(SymbolId)
        Missing
        +is_missing() bool
        +as_f64() Option
        +approx_eq() bool
    }
    FieldId --> Value : indexes values at FieldId
    SymbolId --> Value : Discrete payload
```

* `FieldId(u32)` assigned by `Interner` cold — dense `0..num_fields`. Hot path is `values[fid.as_usize()]` single bounds check.
* `SymbolId(u32)` for every discrete string. Forward map `String -> SymbolId` cold; dense `Vec<String>` for reverse (cache-line friendly).
* `Value::Missing` is explicit, not `Option<Value>` — avoids double wrap, keeps `Copy`, keeps branchless `Missing` propagation (`Op::JumpIfMissing`).

## 4. Session construction

```mermaid
flowchart TD
    Env["PmmlEnv::new<br>Arc EnvInner"] --> FromBytes
    FromBytes["Session::from_bytes"] --> Unmarshal["xml::unmarshal bytes"]
    FromFile["Session::from_file<br>read file -> from_bytes"] --> Unmarshal
    Unmarshal --> VerifyRaw["verify_raw"]
    VerifyRaw --> Lower["lower RawPmml to Ir"]
    Lower --> VerifyIr["verify_ir"]
    VerifyIr --> Build["Session::from_ir<br>name_to_id AHashMap<br>symbol maps<br>output_fields cached"]
    Build --> Provider["CpuProvider::new"]
    Provider --> SessionDone["Session Send Sync"]
```

`max_field_id = max(FieldId)+1 max(16)`. `needed = max(max_field_id, num_fields+4).max(16)` passed to `with_value_buffer`.

## 5. Batch abstraction — one method, two layouts

```mermaid
flowchart TD
    BatchTrait["Batch trait<br>Send Sync object-safe"] --> RowMajor["RowMajor<br>HashMap String Value<br>Vec HashMap / slice / single"]
    BatchTrait --> Columnar["Columnar<br>RecordBatch<br>Float64Array / StringArray"]
    RowMajor --> CtxRow["BatchCtx new<br>no col_map"]
    Columnar --> CtxCol["BatchCtx for_record_batch<br>col_map Vec FieldId col_idx"]
    CtxRow --> MatRow["materialize_row loops HashMap"]
    CtxCol --> MatCol["materialize_row loops col_map<br>is_null -> Missing"]
    MatRow --> Values["mut Value slice"]
    MatCol --> Values
```

Why both? Arrow wins at `100k` (61 ns/row) but loses for single row (schema + conversion >1 µs) and needs schema agreement. RowMajor is ergonomic for `HashMap`/`dict` and required for `Association` `Collection`. Provider picks; `Session::run` accepts either.

```mermaid
flowchart LR
    subgraph Run["Session::run dyn Batch"]
        A["batch empty -> empty Rows"] --> B["detect format Columnar?"]
        B -->|yes RecordBatch| C["BatchCtx for_record_batch"]
        B -->|no| D["BatchCtx new"]
        C --> E["provider.eval_batch"]
        D --> E
        E --> F["BatchResult Rows"]
        F --> G["into_single / into_rows / into_record_batch"]
    end
```

### `BatchResult`

* `Rows(Vec<HashMap<String,Value>>)` — always for `Session::run`, regardless of input layout. Most callers use `into_single` / `into_rows`.
* `Columnar(RecordBatch)` — only when explicitly converting via `into_record_batch(schema)`.

## 6. Execution provider — unified `Cpu`

```mermaid
flowchart TD
    Trait["ExecutionProvider trait"] --> Cpu["CpuProvider unified"]
    Cpu --> Serial{"batch len lt 256<br>or rows lt threads*4?"}
    Serial -->|yes| SerialPath["serial loop<br>with_value_buffer per row"]
    Serial -->|no| ParPath["rayon par_chunks 256<br>scoped threads"]
    SerialPath --> EvalRow["eval_row"]
    ParPath --> EvalRow
    EvalRow --> Derived["eval_derived_fields topo Vec Op"]
    Derived --> Model["evaluate_model 19 arms"]
    Model --> Out["output + targets"]
    Out --> Rows["BatchResult Rows"]
```

* `CpuProvider` is the only provider today; `preferred_format = Columnar` hint but handles both.
* `eval_row` is single-row VM: derived fields → model → output/targets.
* `eval_batch` auto-shards. Threshold `lt 256` avoids `rayon` spawn cost (~100 µs > 400 ns × 256 rows). Previously split `cpu_serial.rs`/`cpu_batched.rs` — now merged into `providers/cpu.rs`.
* `rayon` global pool (future: per-`PmmlEnv` pool).

## 7. Concurrency & memory

```mermaid
flowchart TB
    subgraph Shared["Shared Send Sync"]
        SessionBox["Session Arc Ir + AHashMap + opts"]
    end
    subgraph Thread["Per-thread no Sync needed"]
        Stack["Stack Value 64 x 16B = 1KB<br>L1-hot if needed le 64"]
        TV["THREAD_VALUES RefCell Vec Value<br>reused, never shrinks"]
        Arena["THREAD_ARENA BumpArena<br>per chunk owned Send"]
        Lag["LAG_BUFFER HashMap FieldId VecDeque cap 128"]
    end
    SessionBox -->|self run concurrent| Stack
    SessionBox --> TV
    SessionBox --> Arena
    SessionBox --> Lag
```

* `Session: Send+Sync`. `run(&self)` never takes `&mut` — it borrows `Arc<Ir>` and builds `BatchCtx` on stack, then `with_value_buffer` gives each thread its own `mut Value` slice.
* `STACK_VALUES_THRESHOLD = 64` covers ~90% fixtures (Iris 3, Diabetes 8, Shopping 22). Larger models spill to `thread_local Vec`.
* `BumpArena` is `Send` (owns `Bump`) moved into rayon threads; `miri` clean, no leak.
* `LAG_BUFFER` is `thread_local` so `Lag` doesn't cross batches.

```mermaid
sequenceDiagram
    participant A as Thread 1
    participant B as Thread 2
    participant S as Session

    A->>S: run(batchA as dyn Batch)
    B->>S: run(batchB as dyn Batch)
    Note over A: with_value_buffer -> stack or THREAD_VALUES tid1
    Note over B: with_value_buffer -> stack or THREAD_VALUES tid2
    A->>S: materialize_row via col_map or hashmap
    B->>S: materialize_row
    A->>A: eval_row own mut Value slice
    B->>B: eval_row own mut Value slice
```

## 8. IR — what `lower` produces

```mermaid
flowchart TD
    Raw["RawPmml xml strings"] --> Intern["Interner<br>field_names<br>symbol_names"]
    Intern --> Meta["FieldMeta per DataField + MiningField<br>outlier/invalid/missing treatments"]
    Meta --> Derive["DerivedFieldIr topo DAG<br>bytecode Vec Op"]
    Derive --> ModelSel{"Which model present?"}
    ModelSel --> Tree["TreeIr flat Vec NodeIr root 0"]
    ModelSel --> Reg["RegressionIr"]
    ModelSel --> Mining["MiningIr Segmentation"]
    ModelSel --> Other["Scorecard / Clustering / NaiveBayes<br>kNN / SVM / NeuralNetwork<br>GR / Association / RuleSet<br>Anomaly / Baseline / Text<br>TimeSeries / GP / Sequence / BayesianNetwork"]
    Tree --> Verify["verify_ir invariants"]
    Reg --> Verify
    Mining --> Verify
    Other --> Verify
    Verify --> IrDone["Ir<br>data_dictionary derived_fields model"]
```

* `TreeModel` flattened `Vec<NodeIr>` — root at 0, branchless traversal, `SmallVec<[Box<PredicateIr>;4]>`.
* `DerivedFieldIr.bytecode: Vec<Op>` evaluated by `engine::transform::vm::eval` in topo order; `Op::JumpIfMissing` for `IF`.
* All `FieldId`/`SymbolId` in `Ir` are interned via single `get_or_intern_field`.

## 9. Engine — dispatch

```mermaid
flowchart TD
    Values["mut Value slice"] --> MiningSchema["apply_mining_schema<br>outlier + invalid + missing"]
    MiningSchema --> Derived["eval_derived_fields<br>vm eval DAG"]
    Derived --> Predicate["eval_predicate<br>True / Simple / SimpleSet / Compound"]
    Predicate --> Model{"ModelIr"}
    Model --> T["Tree flat Vec traversal"]
    Model --> R["Regression coeff pow -> normalization"]
    Model --> M["Mining segments weighted vote"]
    Model --> S["Scorecard reasonCode"]
    Model --> C["Clustering nearest"]
    Model --> N["NaiveBayes PairCounts / Gaussian"]
    Model --> K["kNN InlineTable distance"]
    Model --> Etc["... 12 more"]
    T --> Targets["apply_targets rescale/cast"]
    R --> Targets
    M --> Targets
    Targets --> Output["build_output 26 ResultFeature<br>4 unsupported -> Missing"]
    Output --> Map["HashMap String Value<br>predictedValue + target + outputs"]
```

* `vm.rs` handles `PushField`, `PushConst`, `CallBuiltin`, `JumpIfMissing`, `MapValues`/`Multi`, `Discretize`, `NormContinuous`/`Discrete`, `Lag`.
* `builtin.rs` maps 80+ PMML function names → `BuiltinId`.
* `simd.rs` (`feature=simd`) provides `wide::f64x4` fast path for single-table `Regression` on `RecordBatch` ≥4 rows.

## 10. Storage & serialization boundaries

| Boundary | Format | Notes |
|---|---|---|
| XML in | `quick-xml 0.37` pull reader | `trim_text(true)`, `expand_empty_elements`, depth 512, 100 MB, DTD ignored → XXE safe |
| RawPmml | owned `String`/`Vec` | cold only, dropped after `lower` |
| Ir | `Arc<Ir>` immutable | flat nodes, topo derived, dense symbol vec |
| Arrow in/out | `arrow 53` `RecordBatch` | `Float64Array`/`StringArray` zero-copy, `TableLocator` → empty batch, `csv_str_to_record_batch` via `arrow::csv` |
| Python | `pyo3 0.22` | `extension-module`, `allow_threads` for run (planned `InferenceSession`) |
| C | `ffi` opaque `*mut PmmlEnv/Session` | `PmmlStatusCode Ok=0 Error=1`, `Send+Sync`, null-tolerant release |

## 11. Performance — targets vs measured (i7-12700, release)

| Path | Target | Measured | Technique |
|---|---|---|---|
| Cold `from_bytes` (Iris 2.9 KB, 5 nodes) | ≤80 µs | 68 µs | quick-xml + lower + verify |
| Single `run` | ≤800 ns | 402 ns | stack `Value[64]` + `AHashMap` 3× + branchless tree |
| Batch 1k row-major | ≤350 µs | 336 µs (2.97M rows/s) | serial loop, `with_value_buffer` reuse |
| Batch 1k columnar | ≤250 µs | 249 µs (4.0M rows/s) | `RecordBatch` `col_map`, no HashMap |
| Batch 100k columnar batched | 61 ns/row | 61 ns/row (16.5M rows/s) | `rayon par_chunks(256)` + thread_local |

`STACK_VALUES_THRESHOLD=64` → `64×16B=1KB` on caller frame.

## 12. Invariants — break these and `miri`/`fuzz` will tell you

* `Ir.field_names` contains every `FieldId` in `active_fields` + `target_field` + `DerivedFieldIr`.
* `symbol_names` ↔ `symbol_names_vec` agree: `vec[max_id+1]` dense, `HashMap` for lookup.
* `DerivedFieldIr` DAG is topo sorted — `eval_derived_fields` assumes order, no cycle check hot.
* `Value::Missing` is a value, not absence — `Op::JumpIfMissing` is the only branching on it; equality/comparison on `Missing` → `false`.
* `PmmlError::UnsupportedMarkup` only for `ModelComposition`/`CenterFields`; all 19 models are supported.
* `max_field_id = max(FieldId)+1 max(16)`; `needed = max(max_field_id, num_fields+4)`; out-of-bounds `FieldId` ignored, never panic.
* Hardenings: `MAX_DEPTH 512`, `MAX_XML_SIZE 100MB`, `LAG_BUFFER` cap 128 — `cargo test --test hardening` + `cargo fuzz` cover.

## 13. Extension points

```mermaid
flowchart LR
    A["New model"] --> B["ModelIr::New"]
    B --> C["xml::unmarshal RawNewModel"]
    C --> D["ir::lower arm"]
    D --> E["engine/models/new.rs"]
    E --> F["Session::from_ir output_fields"]
    F --> G["provider eval_row dispatch"]
    G --> H["verify_raw not Unsupported"]
```

* **New `BuiltinId`** — add variant `ir::BuiltinId`, map in `builtin_by_name`, dispatch in `eval_builtin` (`statrs`/`libm`/`chrono`).
* **New `ResultFeature`** — `base::ResultFeature::FromStr` + `is_unsupported`, `engine::output::build_output` match, `Session` mapping.
* **New provider** — implement `ExecutionProvider {eval_row, eval_batch, preferred_format}`, register in `Session::from_ir` via `SessionOptions`.

## 14. Trade-offs & rejected alternatives

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| XML | `quick-xml 0.37` pull | `serde`/`XJC` generated | XSD 4490 lines, mixed Attr/Elem, `Extension` vendor payloads; serde can't express ordering; quick-xml gives hardening + 68 µs |
| Interning | `lasso::Rodeo` cold only | `Rodeo` hot | `AHashMap::get` zero-alloc already 3×; `Rodeo` only helps Python `&str` without `String` |
| Batch | `Batch` trait both layouts | Arrow only | single row `HashMap` 402 ns < Arrow >1 µs + schema friction; `dict`/`Collection` natural |
| Providers | single `Cpu` auto-serial vs `rayon` | `CpuSerial` + `CpuBatched` split | threshold lt 256 avoids spawn cost; merged `cpu.rs` is simpler |
| Arena | `bumpalo::Bump` `thread_local` + `Arc<Ir>` | `moka` Guava `LoadingCache` | `Ir` immutable, no invalidation; bump reset per chunk |
| Crate layout | single `pmmlruntime` + `base/xml/ir/engine/session` mods | 9-crate workspace | `cargo add pmmlruntime` one doc page, <26k LOC |

See `README.md` for user-facing API and `cargo doc` for per-item invariants.
