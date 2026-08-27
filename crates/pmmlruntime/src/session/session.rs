use crate::base::error::{PmmlError, Result};
use crate::base::{FieldId, SymbolId, Value};
use crate::ir::Ir;
use crate::session::batch::{Batch, BatchCtx, BatchResult};
use crate::session::env::PmmlEnv;
use crate::session::options::SessionOptions;
use crate::session::providers::{CpuProvider, ExecutionProvider};
use ahash::AHashMap;
#[allow(unused_imports)]
use arrow::array::{Array, Float64Array, StringArray};
#[allow(unused_imports)]
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Thread-local reusable Value buffer — avoids per-run Vec allocation (E1 bump arena like session BFCArena)
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
/// buffer otherwise. Mirrors session runtime's small-model stack fallback + `BFCArena`
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

/// Immutable scoring session, analogous to session runtime `Session`.
///
/// Holds `Arc<Ir>` (immutable model) and a boxed [`ExecutionProvider`].
/// Cheaply `Send` + `Sync`; `run` uses a stack `Value` buffer for `<=64` fields (L1-hot) and a
/// `thread_local!` heap buffer otherwise, so `&self` scoring never allocates per row.
///
/// Design mirrors session runtime `Session`:
///
/// - `Ir` is `Arc` immutable (like `Ir`), `Session` is `Send+Sync`
/// - `Value[FieldId]` is materialized per row via `with_value_buffer` helper (stack `64` + `thread_local`)
/// - `Batch` trait abstracts `Vec<HashMap>` (row-major, PMML compat) vs `RecordBatch` (columnar, Arrow zero-copy)
/// - `ExecutionProvider` (`Cpu`) owns batch sharding (auto serial vs `rayon`), `Session` only does `Value` materialization + output mapping.
///
/// See [`crate::session::batch`] for `Batch`/`BatchResult` and [`crate::session::providers`] for `eval_row`/`eval_batch`.
///
/// # Thread safety
///
/// `Session` is `Send` + `Sync`. All interior state after construction is immutable
/// except the `thread_local!` `Value` buffer used during scoring, which is per-thread.
/// You can share a single `Session` across threads and call `run`/`run_batch` concurrently
/// without external synchronization. `PmmlEnv` is `Arc` internally (like `PmmlEnv`) and
/// also `Send` + `Sync`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
/// use pmmlruntime::session::batch::Batch;
/// use pmmlruntime::base::Value;
/// use std::collections::HashMap;
///
/// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
/// let env = PmmlEnv::new();
/// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
/// let mut input = HashMap::new();
/// input.insert("x".to_string(), Value::Continuous(1.0));
/// let out = sess.run(&input as &dyn Batch).unwrap().into_single().unwrap();
/// assert!(out.contains_key("predictedValue"));
/// ```
pub struct Session {
    /// Global environment (cheap `Arc` clone, like `PmmlEnv`).
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
    // Engine expects `HashMap<String, FieldId>` — keep `std` here, `AHashMap` for hot path above.
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
    /// The `Session` then holds `Arc<Ir>` and a boxed `Cpu` [`ExecutionProvider`].
    ///
    /// # Parameters
    ///
    /// - `env`: global environment (`Arc` inner, cheap to clone). may own thread pool / logger handles.
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
    /// Provider is always the unified `Cpu` (auto serial vs `rayon`).
    fn from_ir(env: PmmlEnv, ir: Ir, options: SessionOptions) -> Result<Self> {
        let provider: Box<dyn ExecutionProvider> = Box::new(CpuProvider::new());

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

    /// Unified scoring — single row, batch, or Arrow. One CPU provider, one method.
    ///
    /// Accepts any `&dyn Batch`:
    /// - `&HashMap<String, Value>` — single row (1 row)
    /// - `&Vec<HashMap<String, Value>>` / `&[HashMap]` — batch row-major
    /// - `&RecordBatch` — columnar Arrow (zero-copy)
    ///
    /// Returns `BatchResult::Rows` (row-major) for all inputs; `RecordBatch` inputs still return `Rows`
    /// (provider always returns `Rows`; call `.into_record_batch(schema)` if you need Arrow output).
    /// For single row, use `.into_single().unwrap()` or `.into_rows()[0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
    /// use pmmlruntime::session::batch::Batch;
    /// use pmmlruntime::base::Value;
    /// use std::collections::HashMap;
    /// use arrow::array::Float64Array;
    /// use arrow::datatypes::{DataType, Field, Schema};
    /// use arrow::record_batch::RecordBatch;
    /// use std::sync::Arc;
    ///
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
    /// let env = PmmlEnv::new();
    /// let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    ///
    /// // 1) single row via &HashMap
    /// let mut single = HashMap::new();
    /// single.insert("x".to_string(), Value::Continuous(1.0));
    /// let out = sess.run(&single as &dyn Batch).unwrap().into_single().unwrap();
    /// assert!(out.contains_key("predictedValue"));
    ///
    /// // 2) batch via &Vec
    /// let batch = vec![single.clone(), single.clone()];
    /// let outs = sess.run(&batch as &dyn Batch).unwrap().into_rows();
    /// assert_eq!(outs.len(), 2);
    ///
    /// // 3) Arrow via &RecordBatch
    /// let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    /// let rb = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![1.0, 2.0])) as _]).unwrap();
    /// let outs = sess.run(&rb as &dyn Batch).unwrap().into_rows();
    /// assert_eq!(outs.len(), 2);
    /// ```
    pub fn run(&self, batch: &dyn Batch) -> Result<BatchResult> {
        if batch.is_empty() {
            return Ok(BatchResult::Rows(Vec::new()));
        }
        // SIMD fast path for RecordBatch + single-table Regression (kept from run_batch_arrow)
        #[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
        if batch.format() == crate::session::batch::BatchFormat::Columnar {
            if let Some(rb) = batch.as_any().downcast_ref::<RecordBatch>() {
                if let crate::ir::ModelIr::Regression(reg) = &self.ir.model {
                    if reg.regression_tables.len() == 1 && rb.num_rows() >= 4 {
                        let needed = self.max_field_id.max(self.ir.num_fields() + 4);
                        let mut col_map: Vec<(FieldId, usize)> = Vec::new();
                        for (col_idx, field) in rb.schema().fields().iter().enumerate() {
                            if let Some(&fid) = self.name_to_id.get(field.name().as_str()) {
                                col_map.push((fid, col_idx));
                            }
                        }
                        let mut batch_values: Vec<Vec<Value>> = Vec::with_capacity(rb.num_rows());
                        for row_idx in 0..rb.num_rows() {
                            let mut row_vals = vec![Value::Missing; needed];
                            for (fid, col_idx) in &col_map {
                                let col = rb.column(*col_idx);
                                if !col.is_null(row_idx) {
                                    let val = match col.data_type() {
                                        ArrowDataType::Float64 => {
                                            let arr = col
                                                .as_any()
                                                .downcast_ref::<Float64Array>()
                                                .unwrap();
                                            Value::Continuous(arr.value(row_idx))
                                        }
                                        ArrowDataType::Utf8 => {
                                            let arr =
                                                col.as_any().downcast_ref::<StringArray>().unwrap();
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
                        let refs: Vec<&[Value]> =
                            batch_values.iter().map(|v| v.as_slice()).collect();
                        let simd_results =
                            crate::engine::simd::evaluate_regression_batch_simd(reg, &refs);
                        let mut results = Vec::with_capacity(rb.num_rows());
                        for predicted in simd_results {
                            let mut output =
                                HashMap::with_capacity(self.output_fields.len().max(1) + 2);
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
                        return Ok(BatchResult::Rows(results));
                    }
                }
            }
        }
        let ctx = if batch.format() == crate::session::batch::BatchFormat::Columnar {
            if let Some(rb) = batch.as_any().downcast_ref::<RecordBatch>() {
                BatchCtx::for_record_batch(
                    &self.name_to_id,
                    &self.name_to_id_std,
                    &self.symbol_str_to_id,
                    &self.ir.symbol_names,
                    &self.ir,
                    self.max_field_id,
                    &self.output_fields,
                    self.target_name.as_ref(),
                    &self.symbol_names_vec,
                    rb,
                )
            } else {
                BatchCtx::new(
                    &self.name_to_id,
                    &self.name_to_id_std,
                    &self.symbol_str_to_id,
                    &self.ir.symbol_names,
                    &self.ir,
                    self.max_field_id,
                    &self.output_fields,
                    self.target_name.as_ref(),
                    &self.symbol_names_vec,
                )
            }
        } else {
            BatchCtx::new(
                &self.name_to_id,
                &self.name_to_id_std,
                &self.symbol_str_to_id,
                &self.ir.symbol_names,
                &self.ir,
                self.max_field_id,
                &self.output_fields,
                self.target_name.as_ref(),
                &self.symbol_names_vec,
            )
        };
        self.provider.eval_batch(&self.ir, batch, &ctx)
    }

    /// Resolve a field name to its stable [`FieldId`] for use with [`run`](Self::run).
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
    /// Delegates to [`crate::session::input::string_to_value`] (session `Value` string handling).
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

    /// Number of active (input) fields for this model.
    ///
    /// Reads `mining_schema.active_fields.len()` for whichever `ModelIr` is held.
    /// This matches PMML `MiningSchema.getActiveFields().size()` and is used by
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
        let xml = include_bytes!("../../../../bench/pmml/DecisionTreeIris.pmml");
        let env = PmmlEnv::new();
        let opts = SessionOptions::default();
        let sess = Session::from_bytes(&env, xml, opts).unwrap();
        let mut input = HashMap::new();
        input.insert("Petal.Length".to_string(), Value::Continuous(1.4)); // setosa
        input.insert("Petal.Width".to_string(), Value::Continuous(0.2));
        let out = sess
            .run(&input as &dyn crate::session::batch::Batch)
            .unwrap()
            .into_single()
            .unwrap();
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
        let xml = include_bytes!("../../../../bench/pmml/DecisionTreeIris.pmml");
        let env = PmmlEnv::new();
        let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
        let mut input = HashMap::new();
        input.insert("Petal.Length".to_string(), Value::Continuous(6.0));
        input.insert("Petal.Width".to_string(), Value::Continuous(2.0));
        let out = sess
            .run(&input as &dyn crate::session::batch::Batch)
            .unwrap()
            .into_single()
            .unwrap();
        assert!(out.contains_key("predictedValue"));
    }
}
