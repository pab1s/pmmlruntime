# PMML Runtime v1 Tree — Design Spec (ONNX-by-design)

**Fecha:** 2026-08-23
**Repo:** `pab1s/jpmml-evaluator-rs` (`develop` <- `feat/v1-tree-*`)
**Branch plan:** `feat/v1-tree-skeleton` (infra) -> `feat/v1-tree-eval` + `feat/v1-tree-sess` + `feat/v1-tree-bench` (worktrees)
**Licencia:** MIT OR Apache-2.0 (green-field desde spec DMG, no transpilación AGPL)
**Spec:** PMML 4.4 `pmml.xsd:4490` líneas, `GeneralStructure:932`, `jpmml-evaluator-features.md`
**Modelo v1:** `TreeModel` (DecisionTree) único kernel, infra FINAL escalable a 11+ modelos
**Batch default:** `arrow::RecordBatch` (zero-copy PyArrow)
**Equipo:** 8 agentes (4 WT ×2), wall 14-16d, solo 27d

---

## 1. Resumen

PMML Runtime es un runtime Rust super-performante para PMML 4.4 con ergonomics ONNX Runtime: `PmmlEnv -> SessionOptions -> Session -> run()`, `GraphOptimizationLevel 0/1/2`, `ExecutionProvider (CpuSerial/CpuBatched)`, arena `bumpalo`, `FieldId u32` + `SymbolId`, bytecode VM para `DerivedField`, y batch columnar Arrow+Rayon. v1 implementa infra final completa pero solo `TreeModel` como modelo ejecutable; añadir Regression/Mining es solo registrar un `ModelEvaluator` sin tocar Session/IR.

Fork de `onnxruntime` C++ fue descartado: semántica tabular vs tensor y coste 500k LOC C++ vs 15k Rust green-field. Se forkean las *ideas* (Session, Providers, MemoryPattern, kernel fusion) no el código.

---

## 2. Arquitectura

```
PMML XML --quick-xml hardened--> PMML (raw) --lower/verify--> IR (FieldId, SymbolId, DAG, bytecode)
            pmml-xml                pmml-core      pmml-ir
                                                  |
                                              Session::new(env, bytes, opts)  pmml-session
                                                  |
   HashMap<FieldName,Value> --run()--> CpuSerialProvider::evaluate() --> HashMap<Target,Value>
   RecordBatch          --run_batch()-->  mining_schema + vm + tree   --> RecordBatch
                                                  |
                                          C ABI pmml_runtime.h  pmml-ffi
                                          PyO3 InferenceSession pmml-python (stub v1)
                                          clap CLI              pmml-cli
```

**Crate mapping desde repo existente (git mv):**
- `pmml-model-rs` -> `pmml-core`
- `pmml-evaluator-xml` -> `pmml-xml`
- `pmml-evaluator-core` -> `pmml-ir`
- `pmml-model-evaluators` -> `pmml-evaluator`
- `pmml-evaluator-rs` -> `pmml-session`
- `pmml-evaluator-serde` -> `pmml-ffi`
- `pmml-evaluator-testing` -> `pmml-bench`
- `pmml-jni-bridge` -> `pmml-python`
- + `pmml-cli` nuevo

**Escalado v2:** `pmml-evaluator/src/models/regression.rs` + `mining.rs` registrados en `ModelIr::Regression|Mining`. Session no cambia.

---

## 3. PMML 4.4 Coverage v1

- **GeneralStructure:** `Header`, `DataDictionary` (DATATYPE strict minus `dateDaysSince[0]`), `MiningSchema` (OPTYPE strict, `missingValueReplacement`, `invalidValueTreatment`, `outlierTreatment`), `TransformationDictionary`, `Targets`, `Output` (`predictedValue`/`probability`/`transformedValue` v1), `ModelVerification`, `Extension` ignorada.
- **Transformaciones:** `DerivedField` DAG topo-sort, `EXPRESSION` (`Constant`, `FieldRef`, `Apply`->bytecode, `MapValues`, `Discretize`, `NormContinuous/Discrete`, `InlineTable`), `DefineFunction` inline, `Builtins` 100 subset.
- **TreeModel:** `Node` flat `Vec<NodeIr>`, `Predicate` (`SimplePredicate` con `Equal/NotEqual/LessThan/...`, `SimpleSetPredicate` con `isIn/isNotIn`, `CompoundPredicate And/Or/Xor/Surrogate`), `missingValueStrategy` (`lastPrediction`/`nullPrediction`/`defaultChild`), `noTrueChildStrategy` (`returnNullPrediction`/`returnLastPrediction`), `splitCharacteristic`.
- **Rechaza con `UnsupportedMarkupException`:** `AnomalyDetection`, `Baseline`, `BayesianNetwork`, `GaussianProcess`, `Sequence`, `Text`, `TimeSeries`, `MiningModel/Segmentation/LocalTransformations` deprecated, `Clustering/CenterFields`, `TableLocator`.

---

## 4. Componentes Detallados

### 4.1 pmml-core
- `value.rs: Value { Continuous(f64), Discrete(SymbolId), Missing }`, `SymbolId(u32)`, `FieldId(u32)`, `FieldValue` helpers, `approx_eq` eps 1e-9.
- `field.rs: DataType`, `OpType`, `MiningFunction`, `ResultFeature` enums (serde-free, `FromStr`).
- `error.rs: PmmlError::UnsupportedMarkup|InvalidValue|MissingField` via `thiserror`.
- `arena.rs: thread_local! Arena(bumpalo::Bump)` + `SmallVec<[(FieldId, Value); 16]>`.

### 4.2 pmml-xml
- `reader.rs: quick-xml 0.37` pull reader, `expand_empty_elements`, `check_end_names`, max_depth 128, no DTD, `memchr` fast `find '<'`.
- `unmarshal.rs: fn unmarshal(bytes) -> RawPMML` manual `match tag` (no serde), parsa `DataField`, `MiningField`, `DerivedField`, `TreeModel`+`Node` recursivo.

### 4.3 pmml-ir
- `intern.rs: Rodeo` interner `FieldName->FieldId`, `value string->SymbolId`.
- `ir.rs: struct Ir { data_dictionary: Vec<FieldMeta>, mining_schema: MiningSchemaIr, derived_dag: Vec<DerivedFieldIr {field_id, bytecode: Vec<Op>}, model: ModelIr }`.
- `lower.rs: fn lower(raw: RawPMML, intern: &mut Interner) -> Ir` hace FieldId assignment, DAG topo, bytecode gen, `DefineFunction` inline.
- `verify.rs: fn verify(ir: &Ir) -> Result<(), Vec<PmmlError>>` implementa `UnsupportedInspector`.

### 4.4 pmml-evaluator
- `transform/vm.rs: enum Op { PushField(FieldId), PushConst(Value), CallBuiltin(BuiltinId, u8), JumpIfMissing }` + `eval(&[Op], &mut [Value])` stack 32.
- `transform/builtin.rs: BUILTINS: &[Builtin]` 100 funcs `libm`/`statrs`.
- `transform/{mapvalues,discretize}.rs`
- `mining_schema.rs: fn apply(&MiningSchemaIr, &HashMap<FieldId,Value>, &mut [Value])` branchless `select`.
- `targets.rs, output.rs`
- `models/mod.rs: trait ModelEvaluator`, `models/tree.rs` flat traversal, `tree_lower.rs`.

### 4.5 pmml-session
- `env.rs: struct PmmlEnv { pool: rayon::ThreadPool, logger }` `Send+Sync`.
- `options.rs: GraphOptLevel {DisableAll(0), EnableBasic(1), EnableAll(2)}` + `SessionOptionsBuilder`.
- `session.rs: struct Session { env: Arc<Env>, ir: Arc<Ir>, provider: Box<dyn ExecutionProvider> }` `run(&self, map)`, `run_batch(&self, RecordBatch)`.
- `providers/{mod.rs,cpu_serial.rs,cpu_batched.rs}` trait + impl.
- `pmml-ffi`: C ABI `PmmlCreateSession`, `PmmlSessionRun`, `PmmlReleaseSession`, `cbindgen` header.
- `pmml-cli`: `run`, `inspect`, `verify` subcommands via `arrow::csv`.

---

## 5. Performance Pillars v1

1. **FieldId+SymbolId:** `FieldName` HashMap lookup solo en `lower()` (frío). Hot `values[field_id as usize]` array O(1), `Discrete` compare `u32==u32`.
2. **Zero alloc per row:** `thread_local! Bump`, `SmallVec` inputs, reused `Vec<Value>` scratch `values.len() == N_active + N_derived`.
3. **Bytecode VM:** `Apply` no `match function_name` por fila; `eval` loop sobre `Op` (5ns/op) vs Jalisco `Functions` switch.
4. **Branchless mining_schema:** `select(missing_mask, fallback, value)` no `if invalid`.
5. **Flat Tree:** `Vec<NodeIr>` contiguo, prefetch, `match predicate` table jump, no `Box<dyn Predicate>`.

Batch v1 ya expone `run_batch(RecordBatch)` aunque serial; v2 lo vuelve `par_iter` chunks 1k + SIMD `f64x4` Regression.

---

## 6. API

```rust
let env = PmmlEnv::new();
let opts = SessionOptions::new().graph_optimization_level(1).intra_threads(4);
let sess = Session::new(&env, "tree.pmml", &opts)?; // from_bytes también
let out = sess.run(hashmap!{"age"=>33.0, "income"=>Missing})?;
let batch: RecordBatch = sess.run_batch(csv_to_arrow("in.csv"))?;

// C
PmmlSession* s; PmmlCreateSession(env, path, opts, &s); PmmlSessionRun(s, inputs, &outputs);
```

---

## 7. Testing & Gates

- **Unit:** `value coerce`, `outlier`, `builtin` proptest.
- **Parity 46 fixtures:** `bench/pmml/*.pmml` + `testing_batch.csv` vs `jpmml-evaluator` java oracle, `insta` snapshot eps 1e-9, Tree subset `IrisTree, TreeSample, Ming*`.
- **Amplified:** `sklearn.tree.DecisionTreeClassifier` -> `sklearn2pmml` 100 pmmls, differential.
- **Fuzz:** `cargo fuzz` xml reader + `proptest` vm.
- **Bench:** `criterion` single 1..100k, batch 1k/10k vs `mvn -Dbenchmark` en `BENCHMARK.md`.
- **Gates:** `cargo check --workspace` 0 errors, `clippy pedantic` 0 warnings, `miri` clean, `cargo asm` no duplicated bounds, `0 skipped`, `Level1 >=2x JPMML single`, `cold <10ms Iris`.

---

## 8. Fases & Agentes

Ver `docs/PLAN.md` detallado. 8 agentes 4 WT: WT-0 ir(0,1), WT-1 eval(2,3), WT-2 sess(4), WT-3 bench(5), fixer A7 cross.

---

## 9. Referencias

- PMML 4.4 `pmml.xsd:4490`, `GeneralStructure:932`, `jpmml-evaluator-features.md:96`
- Upstream `jpmml/jpmml-evaluator@23d0761 v1.7.7`, Openscoring bench 2021
- Repo actual `pab1s/jpmml-evaluator-rs` `chore/migration-plan` (8 crate placeholders) -> `feat/v1-tree-*` worktrees
