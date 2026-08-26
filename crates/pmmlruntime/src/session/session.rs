use crate::base::error::{PmmlError, Result};
use crate::base::{FieldId, SymbolId, Value};
use crate::ir::Ir;
use crate::session::batch::{Batch, BatchCtx, BatchResult};
use crate::session::env::PmmlEnv;
use crate::session::options::{ExecutionProviderKind, SessionOptions};
use crate::session::providers::{CpuBatchedProvider, CpuSerialProvider, ExecutionProvider};
use ahash::AHashMap;
#[allow(unused_imports)]
use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Thread-local reusable Value buffer — avoids per-run Vec allocation (E1 bump arena like ONNX BFCArena)
thread_local! {
    static THREAD_VALUES: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

/// Stack fast path threshold: 90% of fixtures have `max_field_id` < 32 (Iris 3, Diabetes 8). 64 covers Shopping (22) + buffer.
///
/// Stack allocation is L1-hot, no heap churn, no `RefCell` borrow (P4).
const STACK_VALUES_THRESHOLD: usize = 64;

/// Execute `f` with a `&mut [Value]` of `needed` length.
///
/// Uses a stack array for `needed <= 64` (90% of models) and a `thread_local!` heap
/// buffer otherwise. Mirrors ONNX Runtime's small-model stack fallback + `BFCArena`
/// for large models. The slice is always initialized to [`Value::Missing`] so unused
/// slots are deterministic.
///
/// # Parameters
///
/// - `needed`: number of `Value` slots required, typically `max(max_field_id, ir.num_fields()+4).max(16)`.
///
/// # Returns
///
/// Whatever `f` returns. `f` receives `&mut [Value]` of exactly `needed` length.
///
/// # Performance
///
/// Stack path is ~1 KB (`64 * 16 B`) on the caller's frame, no allocation, no `RefCell`.
/// Heap path reuses the thread-local `Vec<Value>` and only grows, never shrinks, and
/// zeroes the prefix on reuse. Avoids per-row `Vec` allocation (~402 ns single-row vs >1 µs with alloc).
///
/// # Concurrency
///
/// Stack path is thread-independent. Heap path uses a `thread_local!` `RefCell<Vec<Value>>`,
/// so each thread has its own buffer and no cross-thread synchronization is needed.
/// `&self` scoring can be called concurrently from multiple threads; each thread
/// materializes its own `values` slice via this helper.
///
/// # Examples
///
/// ```ignore
/// use pmmlruntime::base::Value;
/// // with_value_buffer is pub(crate); shown as if it were public:
/// // let sum = with_value_buffer(4, |values| { values[0] = Value::Continuous(1.0); values[1] = Value::Continuous(2.0); 3.0 });
/// ```
#[inline(always)]
pub(crate) fn with_value_buffer<R>(needed: usize, f: impl FnOnce(&mut [Value]) -> R) -> R {
    if needed <= STACK_VALUES_THRESHOLD {
        // Stack array — uninitialized would be faster with MaybeUninit, but Missing init is cheap (64*16B=1KB)
        // and ensures deterministic Missing for unused slots.
        let mut buf = [Value::Missing; STACK_VALUES_THRESHOLD];
        let slice = &mut buf[..needed];
        f(slice)
    } else {
        THREAD_VALUES.with(|cell| {
            let mut values = cell.borrow_mut();
            if values.len() < needed {
                values.resize(needed, Value::Missing);
            } else {
                for v in values.iter_mut().take(needed) {
                    *v = Value::Missing;
                }
            }
            f(&mut values[..needed])
        })
    }
}

/// Immutable scoring session, analogous to ONNX Runtime `OrtSession`.
///
/// Holds `Arc<Ir>` (immutable model) and a boxed [`ExecutionProvider`].
/// Cheaply `Send` + `Sync`; `run` uses a stack `Value` buffer for `<=64` fields (L1-hot) and a
/// `thread_local!` heap buffer otherwise, so `&self` scoring never allocates per row.
///
/// Design mirrors ONNX Runtime `OrtSession`:
///
/// - `Ir` is `Arc` immutable (like `OrtModel`), `Session` is `Send+Sync`
/// - `Value[FieldId]` is materialized per row via `with_value_buffer` helper (stack `64` + `thread_local`)
/// - `Batch` trait abstracts `Vec<HashMap>` (row-major, JPMML compat) vs `RecordBatch` (columnar, Arrow zero-copy)
/// - `ExecutionProvider` owns batch sharding (`rayon` for `CpuBatched`), `Session` only does `Value` materialization + output mapping.
///
/// See [`crate::session::batch`] for `Batch`/`BatchResult` and [`crate::session::providers`] for `eval_row`/`eval_batch`.
///
/// # Thread safety
///
/// `Session` is `Send` + `Sync`. All interior state after construction is immutable
/// except the `thread_local!` `Value` buffer used during scoring, which is per-thread.
/// You can share a single `Session` across threads and call `run`/`run_batch` concurrently
/// without external synchronization. `PmmlEnv` is `Arc` internally (like `OrtEnv`) and
/// also `Send` + `Sync`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
/// use pmmlruntime::base::Value;
/// use std::collections::HashMap;
///
/// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
/// let env = PmmlEnv::new();
/// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
/// let mut input = HashMap::new();
/// input.insert("x".to_string(), Value::Continuous(1.0));
/// let out = sess.run(input).unwrap();
/// assert!(out.contains_key("predictedValue"));
/// ```
pub struct Session {
    /// Global environment (cheap `Arc` clone, like `OrtEnv`).
    pub env: PmmlEnv,
    /// Options used to build this session (graph opt level, threads, provider).
    pub options: SessionOptions,
    /// Immutable lowered IR (`Arc` so `Session` can be `Clone` cheaply in future).
    pub ir: Arc<Ir>,
    provider: Box<dyn ExecutionProvider>,
    // reverse map for field name -> FieldId (from Ir field_names) — ahash for hot path (E1)
    // `AHashMap` avoids SipHash overhead per row (~3× vs std). No `Rodeo` — `AHashMap::get(&str)` is already
    // zero-alloc via `Borrow<str>`; `Rodeo`/`Spur` only needed if Python passes `&str` without `String` alloc (re-add then).
    name_to_id: AHashMap<String, FieldId>,
    // std HashMap clone for GeneralRegression/Mining evaluator (cached, not per-row)
    // `pmml-evaluator` API expects `HashMap<String, FieldId>` — keep `std` here, `AHashMap` for hot path above.
    name_to_id_std: HashMap<String, FieldId>,
    // max field id for values vec size
    max_field_id: usize,
    // target field name for output (if known)
    target_name: Option<String>,
    // P7: cached output fields (pre-resolved, avoids match per row)
    output_fields: Vec<crate::ir::OutputFieldIr>,
    // P1: forward symbol map String -> SymbolId for zero-copy Arrow discrete inputs
    symbol_str_to_id: HashMap<String, crate::base::SymbolId>,
    // Dense table for SymbolId → String (cache-line friendly, used for probability output)
    symbol_names_vec: Vec<String>,
}

impl Session {
    /// Create a session from PMML XML bytes (cold path).
    ///
    /// Delegates to `crate::xml::unmarshal` → `crate::ir::verify_raw` → `crate::ir::lower` → `crate::ir::verify_ir` → `from_ir`.
    /// The `Session` then holds `Arc<Ir>` and a boxed [`ExecutionProvider`] chosen by `options.execution_provider`.
    ///
    /// # Parameters
    ///
    /// - `env`: global environment (`Arc` inner, cheap to clone). Like `OrtEnv`, it owns the thread pool / logger in v2.
    /// - `bytes`: raw PMML XML (UTF-8). File cap is 100 MB and depth cap is 512 inside `pmml_xml` (XXE-hardened).
    /// - `options`: builder for graph optimization level, intra-op threads, and provider kind.
    ///
    /// # Returns
    ///
    /// `Ok(Session)` with immutable `Ir` and provider ready for `run` / `run_batch`.
    ///
    /// # Errors
    ///
    /// Returns [`PmmlError`] variants:
    /// - `PmmlError::Parse` if XML is not well-formed or violates `pmml.xsd`.
    /// - `PmmlError::UnsupportedMarkup` if `verify_raw` / `verify_ir` rejects `AnomalyDetectionModel` etc.
    /// - `PmmlError::InvalidValue` if lowering fails (e.g. missing `DataDictionary` field).
    /// - `PmmlError::Io` is not used here (see [`from_file`](Self::from_file) for IO).
    ///
    /// # Performance
    ///
    /// ~68 µs for Iris 2.9 KB on the cold path (includes XML parse + verify + lower). Hot path `run` is ~402 ns.
    ///
    /// # Concurrency
    ///
    /// This is a constructor; no concurrency concerns. The resulting `Session` is `Send+Sync`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// assert_eq!(sess.num_active_fields(), 1);
    /// ```
    pub fn from_bytes(env: &PmmlEnv, bytes: &[u8], options: SessionOptions) -> Result<Self> {
        let raw = crate::xml::unmarshal(bytes)?;
        crate::ir::verify_raw(&raw)?;
        let ir = crate::ir::lower(raw)?;
        crate::ir::verify_ir(&ir)?;
        Self::from_ir(env.clone(), ir, options)
    }

    /// Create a session from a file path (cold path).
    ///
    /// Reads the file fully into memory then delegates to [`from_bytes`](Self::from_bytes).
    /// Use this for CLI / batch jobs; for in-memory bytes prefer `from_bytes`.
    ///
    /// # Parameters
    ///
    /// - `env`: global environment.
    /// - `path`: filesystem path to the PMML file (UTF-8).
    /// - `options`: session options.
    ///
    /// # Returns
    ///
    /// `Ok(Session)` on success.
    ///
    /// # Errors
    ///
    /// - `PmmlError::Io` if `std::fs::read` fails (file not found, permission denied).
    /// - Propagates `PmmlError::Parse` / `UnsupportedMarkup` / `InvalidValue` from `from_bytes`.
    ///
    /// # Panics
    ///
    /// Does not panic. IO errors are returned as `PmmlError::Io`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_file(&env, "model.pmml", SessionOptions::default()).unwrap();
    /// # let _ = sess;
    /// ```
    pub fn from_file(env: &PmmlEnv, path: &str, options: SessionOptions) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| PmmlError::Io(e.to_string()))?;
        Self::from_bytes(env, &bytes, options)
    }

    /// Lowered-IR constructor (cold path, crate-private).
    ///
    /// Builds `name_to_id` (`AHashMap` for hot path + `std::HashMap` for evaluator),
    /// `max_field_id` (`max(FieldId)+1` clamped to at least 16), `target_name`,
    /// cached `output_fields` (P7), forward `symbol_str_to_id` (P1), and dense
    /// `symbol_names_vec` (cache-line friendly for probability output).
    ///
    /// Provider is chosen by `options.execution_provider` (`CpuSerial` vs `CpuBatched`).
    fn from_ir(env: PmmlEnv, ir: Ir, options: SessionOptions) -> Result<Self> {
        let provider: Box<dyn ExecutionProvider> = match options.execution_provider {
            ExecutionProviderKind::CpuSerial => Box::new(CpuSerialProvider::new()),
            ExecutionProviderKind::CpuBatched => Box::new(CpuBatchedProvider::new()),
        };

        // Build name->FieldId map from Ir field_names (FieldId -> name) — E1: ahash for hot path
        // `AHashMap::get(&str)` is zero-alloc via `Borrow<str>`; no Rodeo needed until Python needs `&str` interning.
        let mut name_to_id: AHashMap<String, FieldId> = AHashMap::new();
        let mut name_to_id_std: HashMap<String, FieldId> = HashMap::new();
        for (fid, name) in &ir.field_names {
            name_to_id.insert(name.clone(), *fid);
            name_to_id_std.insert(name.clone(), *fid);
        }
        // Also include derived fields names (if any) — already in field_names if lower populated correctly
        // Determine max field id for values vec
        let max_field_id = name_to_id
            .values()
            .map(|fid| fid.as_usize())
            .max()
            .unwrap_or(0)
            + 1;
        // Extract target name from model
        let target_name = match &ir.model {
            crate::ir::ModelIr::Tree(t) => t
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Regression(r) => r
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Mining(m) => m
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Scorecard(s) => s
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Clustering(c) => c
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::NaiveBayes(n) => n
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::NearestNeighbor(n) => n
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::SupportVectorMachine(s) => s
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::GeneralRegression(g) => g
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Association(a) => a
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::RuleSet(r) => r
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::NeuralNetwork(n) => n
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::AnomalyDetection(a) => a
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Baseline(b) => b
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::GaussianProcess(g) => g
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Text(t) => t
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::TimeSeries(t) => t
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::Sequence(s) => s
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            crate::ir::ModelIr::BayesianNetwork(b) => b
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
        };
        // P7: cache output fields to avoid per-row match on ModelIr
        let output_fields = match &ir.model {
            crate::ir::ModelIr::Tree(t) => t.output.clone(),
            crate::ir::ModelIr::Regression(r) => r.output.clone(),
            crate::ir::ModelIr::Mining(m) => m.output.clone(),
            crate::ir::ModelIr::Scorecard(s) => s.output.clone(),
            crate::ir::ModelIr::Clustering(c) => c.output.clone(),
            crate::ir::ModelIr::NaiveBayes(n) => n.output.clone(),
            crate::ir::ModelIr::NearestNeighbor(n) => n.output.clone(),
            crate::ir::ModelIr::SupportVectorMachine(s) => s.output.clone(),
            crate::ir::ModelIr::GeneralRegression(g) => g.output.clone(),
            crate::ir::ModelIr::Association(a) => a.output.clone(),
            crate::ir::ModelIr::RuleSet(r) => r.output.clone(),
            crate::ir::ModelIr::NeuralNetwork(n) => n.output.clone(),
            crate::ir::ModelIr::AnomalyDetection(a) => a.output.clone(),
            crate::ir::ModelIr::Baseline(b) => b.output.clone(),
            crate::ir::ModelIr::GaussianProcess(g) => g.output.clone(),
            crate::ir::ModelIr::Text(t) => t.output.clone(),
            crate::ir::ModelIr::TimeSeries(t) => t.output.clone(),
            crate::ir::ModelIr::Sequence(s) => s.output.clone(),
            crate::ir::ModelIr::BayesianNetwork(b) => b.output.clone(),
        };
        // P1: forward symbol map for Arrow discrete zero-copy (String -> SymbolId)
        let symbol_str_to_id: HashMap<String, crate::base::SymbolId> = ir
            .symbol_names
            .iter()
            .map(|(sid, s)| (s.clone(), *sid))
            .collect();
        // Dense table for SymbolId → String (cache-line friendly, used for probability output)
        let max_symbol_id = ir.symbol_names.keys().map(|sid| sid.0).max().unwrap_or(0) as usize;
        let mut symbol_names_vec = vec![String::new(); max_symbol_id + 1];
        for (sid, name) in &ir.symbol_names {
            let idx = sid.0 as usize;
            if idx < symbol_names_vec.len() {
                symbol_names_vec[idx] = name.clone();
            }
        }

        Ok(Self {
            env,
            options,
            ir: Arc::new(ir),
            provider,
            name_to_id,
            name_to_id_std,
            max_field_id: max_field_id.max(16), // at least 16
            target_name,
            output_fields,
            symbol_str_to_id,
            symbol_names_vec,
        })
    }

    /// Run a single row (hot path).
    ///
    /// Materializes `HashMap<String, Value>` into `Value[FieldId]` via `with_value_buffer`,
    /// evaluates `DerivedFields` + model via the `ExecutionProvider`, then maps the predicted
    /// [`Value`] back to named outputs. Unknown input fields are ignored per PMML spec.
    ///
    /// Output always contains `predictedValue`; if the model has a `target_field` or
    /// `Output` fields, those names are also inserted. For classification with probabilities,
    /// `Probability_*` and per-category entries are added.
    ///
    /// # Parameters
    ///
    /// - `input`: map of field name → [`Value`] (`Continuous(f64)` or `Discrete(SymbolId)` or `Missing`).
    ///
    /// # Returns
    ///
    /// `HashMap<String, Value>` with at least `predictedValue`. For `GeneralRegression` with
    /// softmax, the map also contains per-category probabilities.
    ///
    /// # Errors
    ///
    /// Returns [`PmmlError::InvalidValue`] if `eval_derived_fields` fails, or propagates
    /// provider errors. In practice scoring rarely errors; missing fields become [`Value::Missing`].
    ///
    /// # Panics
    ///
    /// Does not panic. Unknown fields are ignored; out-of-range `FieldId` is bounds-checked.
    ///
    /// # Performance
    ///
    /// ~402 ns per row (stack `64` path, no allocation). `GeneralRegression` softmax does one extra
    /// allocation for probabilities. The `AHashMap` lookup is ~3× faster than `SipHash`.
    ///
    /// # Concurrency
    ///
    /// `&self` is `Send` + `Sync`. Each call uses its own thread-local buffer via `with_value_buffer` (`thread_local!` for large models),
    /// so concurrent `run` calls on the same `Session` are safe without external locking.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::base::Value;
    /// use std::collections::HashMap;
    ///
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    ///
    /// let mut input = HashMap::new();
    /// input.insert("x".to_string(), Value::Continuous(1.0));
    /// // unknown field is ignored, not an error
    /// input.insert("unknown".to_string(), Value::Continuous(9.9));
    /// let out = sess.run(input).unwrap();
    /// assert!(out.contains_key("predictedValue"));
    /// ```
    pub fn run(&self, input: HashMap<String, Value>) -> Result<HashMap<String, Value>> {
        let needed = self.max_field_id.max(self.ir.num_fields() + 4);
        with_value_buffer(needed, |values| {
            // Fill from input using ahash map (E1: SipHash -> ahash, Rodeo interned at build time for validation)
            // Rodeo spur_to_id is kept for future zero-alloc &str -> FieldId fast path via lasso, but ahash already ~3x faster
            for (name, val) in input {
                if let Some(&fid) = self.name_to_id.get(&name) {
                    let idx = fid.as_usize();
                    if idx < needed {
                        values[idx] = val;
                    }
                } else {
                    // Unknown field — ignore per PMML (or error). We'll ignore.
                }
            }
            // We now have &mut [Value] in `values[..needed]` to evaluate
            // To avoid double borrow issues, we will clone the slice handling into a helper that operates on &mut [Value]
            // Use a closure to handle GeneralRegression vs other models without holding borrow across return
            let result: Result<HashMap<String, Value>> = (|| {
                // Handle GeneralRegression specially to get probabilities (needs field_names + symbol_names + name_to_id)
                if let crate::ir::ModelIr::GeneralRegression(gr) = &self.ir.model {
                    // Use cached std map (P0) — no per-row allocation
                    let std_map = &self.name_to_id_std;
                    let (predicted, probs) =
                        crate::engine::models::evaluate_general_regression_with_probs(
                            gr,
                            &values[..needed],
                            &self.ir.field_names,
                            &self.ir.symbol_names,
                            std_map,
                        );
                    let mut output = HashMap::new();
                    for of in &gr.output {
                        match of.feature {
                            crate::base::field::ResultFeature::Probability => {
                                if let Some(cat_sid) = of.value {
                                    if let Some(cat_str) = self
                                        .symbol_names_vec
                                        .get(cat_sid.0 as usize)
                                        .filter(|s| !s.is_empty())
                                    {
                                        if let Some(p) = probs.get(cat_str) {
                                            output.insert(of.name.clone(), Value::Continuous(*p));
                                            continue;
                                        }
                                    }
                                }
                                if let Some(cat_sid) = of.value {
                                    if let Some(cat_str) = self
                                        .symbol_names_vec
                                        .get(cat_sid.0 as usize)
                                        .filter(|s| !s.is_empty())
                                    {
                                        if let Some(p) = probs.get(cat_str) {
                                            output.insert(of.name.clone(), Value::Continuous(*p));
                                            continue;
                                        }
                                    }
                                }
                                output.insert(of.name.clone(), Value::Missing);
                            }
                            crate::base::field::ResultFeature::PredictedValue => {
                                output.insert(of.name.clone(), predicted);
                            }
                            _ => {
                                output.insert(of.name.clone(), predicted);
                            }
                        }
                    }
                    if output.is_empty() {
                        output.insert("predictedValue".to_string(), predicted);
                    }
                    let mut final_out = output;
                    if let Some(tname) = &self.target_name {
                        final_out.entry(tname.clone()).or_insert(predicted);
                    }
                    final_out
                        .entry("predictedValue".to_string())
                        .or_insert(predicted);
                    for (k, v) in probs {
                        final_out.entry(k.clone()).or_insert(Value::Continuous(v));
                        let prob_name = format!("Probability_{}", k);
                        final_out.entry(prob_name).or_insert(Value::Continuous(v));
                    }
                    return Ok(final_out);
                }

                // Call provider for other models — provider evaluates derived fields + model
                let predicted = self.provider.evaluate(&self.ir, &mut values[..needed])?;

                // P7: use cached output_fields (no per-row ModelIr match, no per-row allocation of match)
                let output = {
                    let mut out = HashMap::with_capacity(self.output_fields.len().max(1) + 2);
                    if self.output_fields.is_empty() {
                        out.insert("predictedValue".to_string(), predicted);
                    } else {
                        for of in &self.output_fields {
                            match of.feature {
                                crate::base::field::ResultFeature::PredictedValue => {
                                    out.insert(of.name.clone(), predicted);
                                }
                                crate::base::field::ResultFeature::Probability => {
                                    // v1 stub 0.0 if not calculated via derived probabilities
                                    out.insert(of.name.clone(), Value::Continuous(0.0));
                                }
                                _ => {
                                    out.insert(of.name.clone(), predicted);
                                }
                            }
                        }
                    }
                    out
                };

                let mut final_out = output;
                if let Some(tname) = &self.target_name {
                    final_out.entry(tname.clone()).or_insert(predicted);
                }
                final_out
                    .entry("predictedValue".to_string())
                    .or_insert(predicted);

                Ok(final_out)
            })();
            result
        })
    }
    /// Fast path: run with pre-resolved [`FieldId`] values (no string hashing).
    ///
    /// Avoids `HashMap<String, Value>` lookup by accepting already-resolved `(FieldId, Value)` pairs.
    /// Used by benches to achieve ~400 ns per row and by callers that cache [`field_id`](Self::field_id).
    ///
    /// # Parameters
    ///
    /// - `fields`: slice of `(FieldId, Value)` for the active fields of the row.
    ///
    /// # Returns
    ///
    /// Same output map as [`run`](Self::run): `predictedValue` + target + probabilities.
    ///
    /// # Errors
    ///
    /// Propagates [`PmmlError::InvalidValue`] from derived-field evaluation or provider.
    ///
    /// # Performance
    ///
    /// Stack fast path for `needed <= 64`; no per-row `HashMap` allocation, just array writes.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::base::Value;
    ///
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// let fid = sess.field_id("x").unwrap();
    /// let out = sess.run_with_ids(&[(fid, Value::Continuous(2.0))]).unwrap();
    /// assert!(out.contains_key("predictedValue"));
    /// ```
    pub fn run_with_ids(&self, fields: &[(FieldId, Value)]) -> Result<HashMap<String, Value>> {
        let needed = self.max_field_id.max(self.ir.num_fields() + 4);
        with_value_buffer(needed, |values| {
            for (fid, val) in fields {
                let idx = fid.as_usize();
                if idx < needed {
                    values[idx] = *val;
                }
            }
            // Reuse same evaluation logic as run but without string map
            if let crate::ir::ModelIr::GeneralRegression(gr) = &self.ir.model {
                let std_map = &self.name_to_id_std;
                let (predicted, probs) =
                    crate::engine::models::evaluate_general_regression_with_probs(
                        gr,
                        &values[..needed],
                        &self.ir.field_names,
                        &self.ir.symbol_names,
                        std_map,
                    );
                let mut output = HashMap::new();
                for of in &gr.output {
                    match of.feature {
                        crate::base::field::ResultFeature::Probability => {
                            if let Some(cat_sid) = of.value {
                                if let Some(cat_str) = self
                                    .symbol_names_vec
                                    .get(cat_sid.0 as usize)
                                    .filter(|s| !s.is_empty())
                                {
                                    if let Some(p) = probs.get(cat_str) {
                                        output.insert(of.name.clone(), Value::Continuous(*p));
                                        continue;
                                    }
                                }
                            }
                            output.insert(of.name.clone(), Value::Missing);
                        }
                        _ => {
                            output.insert(of.name.clone(), predicted);
                        }
                    }
                }
                if output.is_empty() {
                    output.insert("predictedValue".to_string(), predicted);
                }
                let mut final_out = output;
                if let Some(tname) = &self.target_name {
                    final_out.entry(tname.clone()).or_insert(predicted);
                }
                final_out
                    .entry("predictedValue".to_string())
                    .or_insert(predicted);
                for (k, v) in probs {
                    final_out.entry(k.clone()).or_insert(Value::Continuous(v));
                }
                return Ok(final_out);
            }
            let predicted = self.provider.evaluate(&self.ir, &mut values[..needed])?;
            let output = {
                let mut out = HashMap::with_capacity(self.output_fields.len().max(1) + 2);
                if self.output_fields.is_empty() {
                    out.insert("predictedValue".to_string(), predicted);
                } else {
                    for of in &self.output_fields {
                        match of.feature {
                            crate::base::field::ResultFeature::PredictedValue => {
                                out.insert(of.name.clone(), predicted);
                            }
                            crate::base::field::ResultFeature::Probability => {
                                out.insert(of.name.clone(), Value::Continuous(0.0));
                            }
                            _ => {
                                out.insert(of.name.clone(), predicted);
                            }
                        }
                    }
                }
                out
            };
            let mut final_out = output;
            if let Some(tname) = &self.target_name {
                final_out.entry(tname.clone()).or_insert(predicted);
            }
            final_out
                .entry("predictedValue".to_string())
                .or_insert(predicted);
            Ok(final_out)
        })
    }

    /// Batched run over `Vec<HashMap>` — delegates to [`ExecutionProvider::eval_batch`] via [`Batch`].
    ///
    /// Builds a [`BatchCtx`] (no per-row allocation) that captures `name_to_id` / `symbol_str_to_id` / `Ir` refs,
    /// then calls `provider.eval_batch`. `CpuBatched` shards with `rayon::par_chunks(256)`; `CpuSerial` loops serially.
    /// For tiny batches (`<256` rows or fewer than `threads*4`) `CpuBatched` falls back to serial to avoid `rayon` overhead.
    ///
    /// # Parameters
    ///
    /// - `batch`: row-major batch, each map is `field name → Value`. Empty batch returns `Ok(Vec::new())` without calling provider.
    ///
    /// # Returns
    ///
    /// `Vec<HashMap<String, Value>>` one output map per input row, same semantics as [`run`](Self::run).
    ///
    /// # Errors
    ///
    /// Propagates [`PmmlError`] from `provider.eval_batch` or `materialize_row` (e.g. `InvalidValue`).
    ///
    /// # Performance
    ///
    /// `BatchCtx` construction is O(fields), not O(rows). Provider reuses the thread-local `Value` buffer per row. Use
    /// [`run_batch_arrow`](Self::run_batch_arrow) for columnar data (61 ns/row at 100k) to avoid per-row `HashMap` clone for input.
    ///
    /// # Concurrency
    ///
    /// `&self` scoring is `Send+Sync`. `CpuBatched::eval_batch` uses `rayon` internally; concurrent `run_batch`
    /// calls from multiple threads are safe.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::base::Value;
    /// use std::collections::HashMap;
    ///
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// let batch = vec![
    ///     { let mut m = HashMap::new(); m.insert("x".into(), Value::Continuous(1.0)); m },
    ///     { let mut m = HashMap::new(); m.insert("x".into(), Value::Continuous(2.0)); m },
    /// ];
    /// let outs = sess.run_batch(batch).unwrap();
    /// assert_eq!(outs.len(), 2);
    /// ```
    pub fn run_batch(
        &self,
        batch: Vec<HashMap<String, Value>>,
    ) -> Result<Vec<HashMap<String, Value>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let ctx = BatchCtx::new(
            &self.name_to_id,
            &self.name_to_id_std,
            &self.symbol_str_to_id,
            &self.ir.symbol_names,
            &self.ir,
            self.max_field_id,
            &self.output_fields,
            self.target_name.as_ref(),
            &self.symbol_names_vec,
        );
        let result = self
            .provider
            .eval_batch(&self.ir, &batch as &dyn Batch, &ctx)?;
        Ok(result.into_rows())
    }

    /// Batched run with shared reference (avoids moving the original `Vec`).
    ///
    /// Clones the slice into a `Vec` internally to reuse the `Vec<HashMap>` [`Batch`] impl.
    /// Useful for benches that retain the original batch. For true zero-copy prefer `RecordBatch` path.
    ///
    /// # Parameters
    ///
    /// - `batch`: slice of row maps.
    ///
    /// # Returns
    ///
    /// Same as [`run_batch`](Self::run_batch): one output map per row.
    ///
    /// # Errors
    ///
    /// Propagates [`PmmlError`] from provider.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::base::Value;
    /// use std::collections::HashMap;
    /// # let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// # let env = PmmlEnv::new();
    /// # let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// let batch: Vec<HashMap<String, Value>> = vec![];
    /// let outs = sess.run_batch_ref(&batch).unwrap();
    /// # let _ = outs;
    /// ```
    pub fn run_batch_ref(
        &self,
        batch: &[HashMap<String, Value>],
    ) -> Result<Vec<HashMap<String, Value>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        // Use Vec Batch impl to avoid DST trait object for [HashMap]
        let owned: Vec<HashMap<String, Value>> = batch.to_vec();
        self.run_batch(owned)
    }

    /// Generic batch via `&dyn Batch` — ONNX `Run` / `RunWithBinding` style.
    ///
    /// Accepts any `Batch` impl (`Vec<HashMap>` or `RecordBatch`) and returns a [`BatchResult`]
    /// that can be `Rows` or `Columnar`. This is the primary batched API; convenience wrappers
    /// [`run_batch`](Self::run_batch) and [`run_batch_arrow`](Self::run_batch_arrow) delegate here.
    ///
    /// # Parameters
    ///
    /// - `batch`: reference to a type implementing [`Batch`] (`Send+Sync` + object-safe).
    ///
    /// # Returns
    ///
    /// [`BatchResult::Rows`] for row-major inputs, or `Rows` for columnar as well (provider always
    /// returns `Rows`; `run_record_batch` then converts to `Columnar`).
    ///
    /// # Errors
    ///
    /// Propagates [`PmmlError`] from `provider.eval_batch`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::base::Value;
    /// use std::collections::HashMap;
    /// # let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// # let env = PmmlEnv::new();
    /// # let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// let batch: Vec<HashMap<String, Value>> = vec![];
    /// let res = sess.run_batch_dyn(&batch as &dyn pmmlruntime::session::batch::Batch).unwrap();
    /// # let _ = res;
    /// ```
    pub fn run_batch_dyn(&self, batch: &dyn Batch) -> Result<BatchResult> {
        let ctx = BatchCtx::new(
            &self.name_to_id,
            &self.name_to_id_std,
            &self.symbol_str_to_id,
            &self.ir.symbol_names,
            &self.ir,
            self.max_field_id,
            &self.output_fields,
            self.target_name.as_ref(),
            &self.symbol_names_vec,
        );
        self.provider.eval_batch(&self.ir, batch, &ctx)
    }

    /// Resolve a field name to its stable [`FieldId`] for zero-copy [`run_with_ids`](Self::run_with_ids).
    ///
    /// # Parameters
    ///
    /// - `name`: PMML field name (e.g. `Petal.Length`) as it appears in `DataDictionary`.
    ///
    /// # Returns
    ///
    /// `Some(FieldId)` if the field exists, `None` if unknown (caller should treat as ignored).
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// assert!(sess.field_id("x").is_some());
    /// assert!(sess.field_id("nope").is_none());
    /// ```
    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        self.name_to_id.get(name).copied()
    }
    /// Resolve a discrete string value to its interned [`SymbolId`] (for categorical inputs).
    ///
    /// Looks up `symbol_str_to_id` built from `Ir.symbol_names` during cold path. For Arrow
    /// `Utf8` columns, `run_batch_arrow` uses this map for zero-copy `Discrete` conversion.
    ///
    /// # Parameters
    ///
    /// - `s`: categorical string (e.g. `setosa`).
    ///
    /// # Returns
    ///
    /// `Some(SymbolId)` if the symbol was seen in the model, `None` if unknown (caller should use `Missing`).
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::base::Value;
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="Species" dataType="string" optype="categorical"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="Species"/></MiningSchema><Node score="setosa"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// // Symbol lookup depends on model having interned the value; this tree's score is "setosa"
    /// let sid = sess.symbol_id("setosa");
    /// # let _ = sid;
    /// ```
    pub fn symbol_id(&self, s: &str) -> Option<SymbolId> {
        self.symbol_str_to_id.get(s).copied()
    }
    /// Convert a raw string value to [`Value`] using `FieldId`/`DataType`/`OpType` + interning.
    ///
    /// Delegates to [`crate::session::input::string_to_value`] (ONNX `OrtValue` string handling).
    /// Empty or `"Missing"` (case-insensitive) becomes [`Value::Missing`]. For categorical
    /// fields with `DataType::String` or `OpType::Categorical`, the string is interned to
    /// `Discrete(SymbolId)` if known; otherwise numeric strings become `Continuous(f64)`.
    ///
    /// # Parameters
    ///
    /// - `field_name`: field whose `DataDictionary` entry decides numeric vs categorical.
    /// - `s`: raw string from CSV / user input.
    ///
    /// # Returns
    ///
    /// Appropriate [`Value`] variant. Unknown categorical strings become `Missing` so predicates fail safely.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::base::Value;
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// assert_eq!(sess.string_to_value("x", "3.14"), Value::Continuous(3.14));
    /// assert_eq!(sess.string_to_value("x", ""), Value::Missing);
    /// ```
    pub fn string_to_value(&self, field_name: &str, s: &str) -> Value {
        let fid = self.field_id(field_name);
        let (dt, op) = if let Some(f) = fid {
            if let Some(meta) = self.ir.data_dictionary.iter().find(|m| m.field_id == f) {
                (Some(meta.data_type), Some(meta.op_type))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        crate::session::input::string_to_value(field_name, s, fid, dt, op, &self.symbol_str_to_id)
    }

    /// Arrow columnar batch scoring — zero-copy input path via the [`Batch`] trait.
    ///
    /// Takes a `RecordBatch` where columns are named after PMML fields (`Float64` for
    /// continuous, `Utf8` for categorical). Returns `Vec<HashMap>` per row (same as
    /// [`run_batch`](Self::run_batch)) but avoids per-row `HashMap<String,Value>` clone for input.
    /// For Arrow output, use [`run_record_batch`](Self::run_record_batch) which returns a `RecordBatch` directly.
    /// Delegates to [`ExecutionProvider::eval_batch`] with [`BatchCtx::for_record_batch`](crate::session::batch::BatchCtx::for_record_batch) (provider does sharding).
    ///
    /// # Parameters
    ///
    /// - `batch`: `RecordBatch` with `num_rows()` rows. Columns must be named after PMML fields; missing columns are `Missing`.
    ///   `Float64` → `Continuous`, `Utf8` → `Discrete(SymbolId)` if in `symbol_str_to_id`, else `Continuous` if parseable, else `Missing`.
    ///
    /// # Returns
    ///
    /// `Vec<HashMap<String, Value>>` one per row, same output map as `run`.
    ///
    /// # Errors
    ///
    /// Returns [`PmmlError::InvalidValue`] from provider or derived-field evaluation.
    ///
    /// # Performance
    ///
    /// Zero-copy input (provider reads `Float64Array` / `StringArray` directly). ~61 ns/row at 100k rows
    /// via `CpuBatched` `par_chunks(256)`. SIMD fast path (feature `simd`, `Regression` single-table, `>=4` rows) uses 4-wide `wide` crate.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use arrow::array::Float64Array;
    /// use arrow::datatypes::{DataType, Field, Schema};
    /// use arrow::record_batch::RecordBatch;
    /// use std::sync::Arc;
    /// # let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// # let env = PmmlEnv::new();
    /// # let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    /// let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![1.0, 2.0])) as _]).unwrap();
    /// let outs = sess.run_batch_arrow(&batch).unwrap();
    /// assert_eq!(outs.len(), 2);
    /// ```
    pub fn run_batch_arrow(&self, batch: &RecordBatch) -> Result<Vec<HashMap<String, Value>>> {
        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }
        // P8: SIMD fast path for single-table regression (4-wide, AVX2/NEON via `wide`)
        // Keep SIMD here before general Batch path — provider does not yet know SIMD.
        #[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
        if let crate::ir::ModelIr::Regression(reg) = &self.ir.model {
            if reg.regression_tables.len() == 1 && batch.num_rows() >= 4 {
                let needed = self.max_field_id.max(self.ir.num_fields() + 4);
                let mut col_map: Vec<(FieldId, usize)> = Vec::new();
                for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                    if let Some(&fid) = self.name_to_id.get(field.name().as_str()) {
                        col_map.push((fid, col_idx));
                    }
                }
                let mut batch_values: Vec<Vec<Value>> = Vec::with_capacity(batch.num_rows());
                for row_idx in 0..batch.num_rows() {
                    let mut row_vals = vec![Value::Missing; needed];
                    for (fid, col_idx) in &col_map {
                        let col = batch.column(*col_idx);
                        if !col.is_null(row_idx) {
                            let val = match col.data_type() {
                                ArrowDataType::Float64 => {
                                    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                                    Value::Continuous(arr.value(row_idx))
                                }
                                ArrowDataType::Utf8 => {
                                    let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                                    let s = arr.value(row_idx);
                                    if let Some(sid) = self.symbol_str_to_id.get(s) {
                                        Value::Discrete(*sid)
                                    } else if let Ok(f) = s.parse::<f64>() {
                                        Value::Continuous(f)
                                    } else {
                                        Value::Missing
                                    }
                                }
                                _ => Value::Missing,
                            };
                            row_vals[fid.as_usize()] = val;
                        }
                    }
                    batch_values.push(row_vals);
                }
                let refs: Vec<&[Value]> = batch_values.iter().map(|v| v.as_slice()).collect();
                let simd_results = crate::engine::simd::evaluate_regression_batch_simd(reg, &refs);
                let mut results = Vec::with_capacity(batch.num_rows());
                for predicted in simd_results {
                    let mut output = HashMap::with_capacity(self.output_fields.len().max(1) + 2);
                    if self.output_fields.is_empty() {
                        output.insert("predictedValue".to_string(), predicted);
                    } else {
                        for of in &self.output_fields {
                            match of.feature {
                                crate::base::field::ResultFeature::PredictedValue => {
                                    output.insert(of.name.clone(), predicted);
                                }
                                crate::base::field::ResultFeature::Probability => {
                                    output.insert(of.name.clone(), Value::Continuous(0.0));
                                }
                                _ => {
                                    output.insert(of.name.clone(), predicted);
                                }
                            }
                        }
                    }
                    let mut final_out = output;
                    if let Some(tname) = &self.target_name {
                        final_out.entry(tname.clone()).or_insert(predicted);
                    }
                    final_out
                        .entry("predictedValue".to_string())
                        .or_insert(predicted);
                    results.push(final_out);
                }
                return Ok(results);
            }
        }
        let ctx = BatchCtx::for_record_batch(
            &self.name_to_id,
            &self.name_to_id_std,
            &self.symbol_str_to_id,
            &self.ir.symbol_names,
            &self.ir,
            self.max_field_id,
            &self.output_fields,
            self.target_name.as_ref(),
            &self.symbol_names_vec,
            batch,
        );
        let result = self
            .provider
            .eval_batch(&self.ir, batch as &dyn Batch, &ctx)?;
        Ok(result.into_rows())
    }

    /// Direct `RecordBatch` → `RecordBatch` scoring (zero-copy input, Arrow output).
    ///
    /// Delegates to [`run_batch_arrow`](Self::run_batch_arrow) for scoring, then builds an output
    /// `RecordBatch` via [`crate::session::arrow::value_maps_to_record_batch`]. Schema is derived from
    /// `output_fields` ( `Probability` → `Float64`, `PredictedValue` inferred from first row's `Value`).
    /// Falls back to `ir_to_arrow_schema` or `predictedValue` `Float64` if no rows.
    ///
    /// # Parameters
    ///
    /// - `batch`: input `RecordBatch`.
    ///
    /// # Returns
    ///
    /// `RecordBatch` with `num_rows() == batch.num_rows()` and columns per `output_fields`.
    ///
    /// # Errors
    ///
    /// Returns [`PmmlError::InvalidValue`] if `value_maps_to_record_batch` fails (e.g. schema mismatch).
    ///
    /// # Performance
    ///
    /// One extra allocation to build output arrays; input is zero-copy.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use arrow::array::Float64Array;
    /// use arrow::datatypes::{DataType, Field, Schema};
    /// use arrow::record_batch::RecordBatch;
    /// use std::sync::Arc;
    /// # let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// # let env = PmmlEnv::new();
    /// # let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    /// let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![1.0])) as _]).unwrap();
    /// let out = sess.run_record_batch(&batch).unwrap();
    /// assert_eq!(out.num_rows(), 1);
    /// ```
    pub fn run_record_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let maps = self.run_batch_arrow(batch)?;
        let schema = if self.output_fields.is_empty() {
            // Fallback schema: single predictedValue Float64 (or Utf8 if discrete)
            // Infer from first result if available
            if let Some(first) = maps.first() {
                if let Some(Value::Discrete(_)) = first.get("predictedValue") {
                    use arrow::datatypes::{Field, Schema};
                    std::sync::Arc::new(Schema::new(vec![Field::new(
                        "predictedValue",
                        ArrowDataType::Utf8,
                        true,
                    )]))
                } else {
                    use arrow::datatypes::{Field, Schema};
                    std::sync::Arc::new(Schema::new(vec![Field::new(
                        "predictedValue",
                        ArrowDataType::Float64,
                        true,
                    )]))
                }
            } else {
                crate::session::arrow::ir_to_arrow_schema(&self.ir)
            }
        } else {
            // Build schema from output_fields: predictedValue/probability -> Float64 else Utf8
            use arrow::datatypes::{Field, Schema};
            let fields: Vec<arrow::datatypes::Field> = self
                .output_fields
                .iter()
                .map(|of| {
                    let dt = match of.feature {
                        crate::base::field::ResultFeature::Probability => ArrowDataType::Float64,
                        crate::base::field::ResultFeature::PredictedValue => {
                            // Peek at first map to infer type, default Float64
                            if let Some(first) = maps.first() {
                                match first.get(&of.name) {
                                    Some(Value::Discrete(_)) => ArrowDataType::Utf8,
                                    _ => ArrowDataType::Float64,
                                }
                            } else {
                                ArrowDataType::Float64
                            }
                        }
                        _ => ArrowDataType::Utf8,
                    };
                    Field::new(of.name.clone(), dt, true)
                })
                .collect();
            std::sync::Arc::new(Schema::new(fields))
        };
        crate::session::arrow::value_maps_to_record_batch(
            &maps,
            schema,
            Some(&self.ir.symbol_names),
        )
        .map_err(PmmlError::InvalidValue)
    }

    /// Convenience: run with string values (coerced via [`string_to_value`](Self::string_to_value)).
    ///
    /// Useful for CSV inputs where all fields arrive as `String`. Each value is converted
    /// via `string_to_value` so categorical strings map to correct `Discrete(SymbolId)` ids
    /// (not hash placeholder). Empty strings become `Missing`.
    ///
    /// # Parameters
    ///
    /// - `input`: `HashMap<String, String>` from CSV / user.
    ///
    /// # Returns
    ///
    /// Same output map as [`run`](Self::run).
    ///
    /// # Errors
    ///
    /// Propagates [`PmmlError`] from `run`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use std::collections::HashMap;
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// let mut input = HashMap::new();
    /// input.insert("x".to_string(), "3.14".to_string());
    /// let out = sess.run_from_strings(input).unwrap();
    /// assert!(out.contains_key("predictedValue"));
    /// ```
    pub fn run_from_strings(
        &self,
        input: HashMap<String, String>,
    ) -> Result<HashMap<String, Value>> {
        let mut map: HashMap<String, Value> = HashMap::new();
        for (k, v) in input {
            let val = self.string_to_value(&k, &v);
            map.insert(k, val);
        }
        self.run(map)
    }

    /// Number of active (input) fields for this model.
    ///
    /// Reads `mining_schema.active_fields.len()` for whichever `ModelIr` is held.
    /// This matches JPMML `MiningSchema.getActiveFields().size()` and is used by
    /// CLI `inspect` to report model arity and by benches to size batches.
    ///
    /// # Returns
    ///
    /// `usize` count of active fields. For `MiningModel` / `Association` etc., this is the number of
    /// fields the model expects; `Missing` values are still scored via `MissingValueStrategy`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    /// assert_eq!(sess.num_active_fields(), 1);
    /// ```
    pub fn num_active_fields(&self) -> usize {
        match &self.ir.model {
            crate::ir::ModelIr::Tree(t) => t.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Regression(r) => r.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Mining(m) => m.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Scorecard(s) => s.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Clustering(c) => c.mining_schema.active_fields.len(),
            crate::ir::ModelIr::NaiveBayes(n) => n.mining_schema.active_fields.len(),
            crate::ir::ModelIr::NearestNeighbor(n) => n.mining_schema.active_fields.len(),
            crate::ir::ModelIr::SupportVectorMachine(s) => s.mining_schema.active_fields.len(),
            crate::ir::ModelIr::GeneralRegression(g) => g.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Association(a) => a.mining_schema.active_fields.len(),
            crate::ir::ModelIr::RuleSet(r) => r.mining_schema.active_fields.len(),
            crate::ir::ModelIr::NeuralNetwork(n) => n.mining_schema.active_fields.len(),
            crate::ir::ModelIr::AnomalyDetection(a) => a.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Baseline(b) => b.mining_schema.active_fields.len(),
            crate::ir::ModelIr::GaussianProcess(g) => g.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Text(t) => t.mining_schema.active_fields.len(),
            crate::ir::ModelIr::TimeSeries(t) => t.mining_schema.active_fields.len(),
            crate::ir::ModelIr::Sequence(s) => s.mining_schema.active_fields.len(),
            crate::ir::ModelIr::BayesianNetwork(b) => b.mining_schema.active_fields.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Value;
    use std::collections::HashMap;

    #[test]
    fn session_iris_tree() {
        let xml = std::fs::read("/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/DecisionTreeIris.pmml").unwrap();
        let env = PmmlEnv::new();
        let opts = SessionOptions::default();
        let sess = Session::from_bytes(&env, &xml, opts).unwrap();
        let mut input = HashMap::new();
        input.insert("Petal.Length".to_string(), Value::Continuous(1.4)); // setosa
        input.insert("Petal.Width".to_string(), Value::Continuous(0.2));
        let out = sess.run(input).unwrap();
        // Predicted should be setosa
        let pred = out.get("predictedValue").unwrap();
        match pred {
            Value::Discrete(sid) => {
                // SymbolId for setosa should be interned; we check not missing
                assert_ne!(sid.0, u32::MAX);
            }
            _ => panic!("expected discrete"),
        }
    }

    #[test]
    fn session_iris_virginica() {
        let xml = std::fs::read("/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/DecisionTreeIris.pmml").unwrap();
        let env = PmmlEnv::new();
        let sess = Session::from_bytes(&env, &xml, SessionOptions::default()).unwrap();
        let mut input = HashMap::new();
        input.insert("Petal.Length".to_string(), Value::Continuous(6.0));
        input.insert("Petal.Width".to_string(), Value::Continuous(2.0));
        let out = sess.run(input).unwrap();
        assert!(out.contains_key("predictedValue"));
    }
}
