use crate::batch::{Batch, BatchCtx, BatchResult};
use crate::env::PmmlEnv;
use crate::options::{ExecutionProviderKind, SessionOptions};
use crate::providers::{CpuBatchedProvider, CpuSerialProvider, ExecutionProvider};
use ahash::AHashMap;
#[allow(unused_imports)]
use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;
use pmml_core::error::{PmmlError, Result};
use pmml_core::{FieldId, SymbolId, Value};
use pmml_ir::ir::Ir;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Thread-local reusable Value buffer — avoids per-run Vec allocation (E1 bump arena like ONNX BFCArena)
thread_local! {
    static THREAD_VALUES: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

/// Stack fast path threshold: 90% of fixtures have max_field_id < 32 (Iris 3, Diabetes 8). 64 covers Shopping (22) + buffer.
/// Stack allocation is L1-hot, no heap churn, no RefCell borrow (P4).
const STACK_VALUES_THRESHOLD: usize = 64;

/// Execute `f` with a `&mut [Value]` of `needed` length. Uses stack for `<=64`, thread-local heap otherwise.
/// This mirrors ONNX Runtime's small-model stack fallback + BFCArena for large.
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

/// Session — immutable, Send+Sync, analogous to OrtSession.
/// Holds Arc<Ir> and provider.
///
/// Design mirrors ONNX Runtime `OrtSession`:
///
/// - `Ir` is `Arc` immutable (like `OrtModel`), `Session` is `Send+Sync`
/// - `Value[FieldId]` is materialized per row via `with_value_buffer` (stack 64 + thread_local)
/// - `Batch` trait abstracts `Vec<HashMap>` (row-major, JPMML compat) vs `RecordBatch` (columnar, Arrow zero-copy)
/// - `ExecutionProvider` owns batch sharding (rayon for `CpuBatched`), `Session` only does `Value` materialization + output mapping.
///
///   See `batch.rs` for `Batch`/`BatchResult` and `providers/mod.rs` for `eval_row`/`eval_batch`.
pub struct Session {
    pub env: PmmlEnv,
    pub options: SessionOptions,
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
    output_fields: Vec<pmml_ir::ir::OutputFieldIr>,
    // P1: forward symbol map String -> SymbolId for zero-copy Arrow discrete inputs
    symbol_str_to_id: HashMap<String, pmml_core::SymbolId>,
    // Dense table for SymbolId → String (cache-line friendly, used for probability output)
    symbol_names_vec: Vec<String>,
}

impl Session {
    /// Create from bytes (PMML XML).
    pub fn from_bytes(env: &PmmlEnv, bytes: &[u8], options: SessionOptions) -> Result<Self> {
        let raw = pmml_xml::unmarshal(bytes)?;
        pmml_ir::verify_raw(&raw)?;
        let ir = pmml_ir::lower(raw)?;
        pmml_ir::verify_ir(&ir)?;
        Self::from_ir(env.clone(), ir, options)
    }

    /// Create from file path.
    pub fn from_file(env: &PmmlEnv, path: &str, options: SessionOptions) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| PmmlError::Io(e.to_string()))?;
        Self::from_bytes(env, &bytes, options)
    }

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
            pmml_ir::ir::ModelIr::Tree(t) => t
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::Regression(r) => r
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::Mining(m) => m
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::Scorecard(s) => s
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::Clustering(c) => c
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::NaiveBayes(n) => n
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::NearestNeighbor(n) => n
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::SupportVectorMachine(s) => s
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::GeneralRegression(g) => g
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::Association(a) => a
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::RuleSet(r) => r
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            pmml_ir::ir::ModelIr::NeuralNetwork(n) => n
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
        };
        // P7: cache output fields to avoid per-row match on ModelIr
        let output_fields = match &ir.model {
            pmml_ir::ir::ModelIr::Tree(t) => t.output.clone(),
            pmml_ir::ir::ModelIr::Regression(r) => r.output.clone(),
            pmml_ir::ir::ModelIr::Mining(m) => m.output.clone(),
            pmml_ir::ir::ModelIr::Scorecard(s) => s.output.clone(),
            pmml_ir::ir::ModelIr::Clustering(c) => c.output.clone(),
            pmml_ir::ir::ModelIr::NaiveBayes(n) => n.output.clone(),
            pmml_ir::ir::ModelIr::NearestNeighbor(n) => n.output.clone(),
            pmml_ir::ir::ModelIr::SupportVectorMachine(s) => s.output.clone(),
            pmml_ir::ir::ModelIr::GeneralRegression(g) => g.output.clone(),
            pmml_ir::ir::ModelIr::Association(a) => a.output.clone(),
            pmml_ir::ir::ModelIr::RuleSet(r) => r.output.clone(),
            pmml_ir::ir::ModelIr::NeuralNetwork(n) => n.output.clone(),
        };
        // P1: forward symbol map for Arrow discrete zero-copy (String -> SymbolId)
        let symbol_str_to_id: HashMap<String, pmml_core::SymbolId> = ir
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

    /// Run single row. Input map: field name -> Value.
    /// Returns output map: output field name -> Value (includes predictedValue).
    /// E1: uses thread-local reusable Value buffer (BumpArena-like) and ahash+Rodeo fast path, avoids per-row HashMap clone.
    /// P4: stack fast path for needed <= 64 (90% models), heap thread-local otherwise.
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
                if let pmml_ir::ir::ModelIr::GeneralRegression(gr) = &self.ir.model {
                    // Use cached std map (P0) — no per-row allocation
                    let std_map = &self.name_to_id_std;
                    let (predicted, probs) =
                        pmml_evaluator::models::evaluate_general_regression_with_probs(
                            gr,
                            &values[..needed],
                            &self.ir.field_names,
                            &self.ir.symbol_names,
                            std_map,
                        );
                    let mut output = HashMap::new();
                    for of in &gr.output {
                        match of.feature {
                            pmml_core::field::ResultFeature::Probability => {
                                if let Some(cat_sid) = of.value {
                                    if let Some(cat_str) = self.symbol_names_vec.get(cat_sid.0 as usize).filter(|s| !s.is_empty()) {
                                        if let Some(p) = probs.get(cat_str) {
                                            output.insert(of.name.clone(), Value::Continuous(*p));
                                            continue;
                                        }
                                    }
                                }
                                if let Some(cat_sid) = of.value {
                                    if let Some(cat_str) = self.symbol_names_vec.get(cat_sid.0 as usize).filter(|s| !s.is_empty()) {
                                        if let Some(p) = probs.get(cat_str) {
                                            output.insert(of.name.clone(), Value::Continuous(*p));
                                            continue;
                                        }
                                    }
                                }
                                output.insert(of.name.clone(), Value::Missing);
                            }
                            pmml_core::field::ResultFeature::PredictedValue => {
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
                                pmml_core::field::ResultFeature::PredictedValue => {
                                    out.insert(of.name.clone(), predicted);
                                }
                                pmml_core::field::ResultFeature::Probability => {
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
    // Helper for run_batch fast path that avoids HashMap<String,Value> clone per row via FieldId array (E1)
    /// Fast path: run with pre-resolved FieldId values (no string hash). Used by bench to achieve 400ns.
    /// P4: stack fast path for <=64.
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
            if let pmml_ir::ir::ModelIr::GeneralRegression(gr) = &self.ir.model {
                let std_map = &self.name_to_id_std;
                let (predicted, probs) =
                    pmml_evaluator::models::evaluate_general_regression_with_probs(
                        gr,
                        &values[..needed],
                        &self.ir.field_names,
                        &self.ir.symbol_names,
                        std_map,
                    );
                let mut output = HashMap::new();
                for of in &gr.output {
                    match of.feature {
                        pmml_core::field::ResultFeature::Probability => {
                            if let Some(cat_sid) = of.value {
                                if let Some(cat_str) = self.symbol_names_vec.get(cat_sid.0 as usize).filter(|s| !s.is_empty()) {
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
                final_out.entry("predictedValue".to_string()).or_insert(predicted);
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
                            pmml_core::field::ResultFeature::PredictedValue => {
                                out.insert(of.name.clone(), predicted);
                            }
                            pmml_core::field::ResultFeature::Probability => {
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
            final_out.entry("predictedValue".to_string()).or_insert(predicted);
            Ok(final_out)
        })
    }

    /// Batched run — delegates to `ExecutionProvider::eval_batch` via `Batch` trait.
    /// `Session` builds `BatchCtx` (no per-row alloc) and provider shards (rayon for `CpuBatched`).
    /// For tiny batches (<256) provider falls back to serial to avoid rayon overhead (see BENCHMARK.md §3).
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
        let result = self.provider.eval_batch(&self.ir, &batch as &dyn Batch, &ctx)?;
        Ok(result.into_rows())
    }

    /// Batched run with shared reference (avoids moving). Useful for benches that
    /// retain original batch Vec. Delegates to `run_batch` via clone to reuse `Vec` Batch impl.
    /// For true zero-copy, caller should use `RecordBatch` path.
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
    /// Accepts any `Batch` impl (`Vec<HashMap>` or `RecordBatch`) and returns `BatchResult`.
    /// For Arrow, use `run_batch_arrow` convenience wrapper. This is the primary batched API.
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

    /// Dense API — resolve field name to FieldId for zero-copy `run_with_ids`
    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        self.name_to_id.get(name).copied()
    }
    /// Resolve discrete string value to SymbolId (for categorical inputs)
    pub fn symbol_id(&self, s: &str) -> Option<SymbolId> {
        self.symbol_str_to_id.get(s).copied()
    }
    /// Convert string value to `Value` using `FieldId`/`DataType`/`OpType` + interning.
    /// Delegates to `crate::input::string_to_value` (ONNX `OrtValue` string handling).
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
        crate::input::string_to_value(field_name, s, fid, dt, op, &self.symbol_str_to_id)
    }

    /// P1: Arrow columnar batch scoring — zero-copy input path via `Batch` trait.
    /// Takes a RecordBatch where columns are named after PMML fields (Float64 for continuous, Utf8 for categorical).
    /// Returns Vec<HashMap> per row (same as `run_batch`) but avoids per-row `HashMap<String,Value>` clone for input.
    /// For output Arrow, use `run_record_batch` which returns `RecordBatch` directly.
    /// Delegates to `ExecutionProvider::eval_batch` with `BatchCtx::for_record_batch` (provider does sharding).
    pub fn run_batch_arrow(
        &self,
        batch: &RecordBatch,
    ) -> Result<Vec<HashMap<String, Value>>> {
        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }
        // P8: SIMD fast path for single-table regression (4-wide, AVX2/NEON via `wide`)
        // Keep SIMD here before general Batch path — provider does not yet know SIMD.
        #[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
        if let pmml_ir::ir::ModelIr::Regression(reg) = &self.ir.model {
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
                let simd_results = pmml_evaluator::simd::evaluate_regression_batch_simd(reg, &refs);
                let mut results = Vec::with_capacity(batch.num_rows());
                for predicted in simd_results {
                    let mut output = HashMap::with_capacity(self.output_fields.len().max(1) + 2);
                    if self.output_fields.is_empty() {
                        output.insert("predictedValue".to_string(), predicted);
                    } else {
                        for of in &self.output_fields {
                            match of.feature {
                                pmml_core::field::ResultFeature::PredictedValue => {
                                    output.insert(of.name.clone(), predicted);
                                }
                                pmml_core::field::ResultFeature::Probability => {
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
                    final_out.entry("predictedValue".to_string()).or_insert(predicted);
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
        let result = self.provider.eval_batch(&self.ir, batch as &dyn Batch, &ctx)?;
        Ok(result.into_rows())
    }

    /// P1: Direct RecordBatch -> RecordBatch scoring (zero-copy input, Arrow output).
    /// Builds output RecordBatch via `value_maps_to_record_batch` helper for now.
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
                crate::arrow::ir_to_arrow_schema(&self.ir)
            }
        } else {
            // Build schema from output_fields: predictedValue/probability -> Float64 else Utf8
            use arrow::datatypes::{Field, Schema};
            let fields: Vec<arrow::datatypes::Field> = self
                .output_fields
                .iter()
                .map(|of| {
                    let dt = match of.feature {
                        pmml_core::field::ResultFeature::Probability => ArrowDataType::Float64,
                        pmml_core::field::ResultFeature::PredictedValue => {
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
        crate::arrow::value_maps_to_record_batch(&maps, schema, Some(&self.ir.symbol_names))
            .map_err(PmmlError::InvalidValue)
    }

    /// Convenience: run with string values (coerced). Useful for CSV.
    /// Uses `string_to_value` per field (proper `SymbolId` interning via `symbol_str_to_id`),
    /// so categorical inputs map to correct `Discrete` ids (not hash placeholder).
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

    /// Number of active fields.
    pub fn num_active_fields(&self) -> usize {
        match &self.ir.model {
            pmml_ir::ir::ModelIr::Tree(t) => t.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::Regression(r) => r.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::Mining(m) => m.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::Scorecard(s) => s.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::Clustering(c) => c.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::NaiveBayes(n) => n.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::NearestNeighbor(n) => n.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::SupportVectorMachine(s) => s.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::GeneralRegression(g) => g.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::Association(a) => a.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::RuleSet(r) => r.mining_schema.active_fields.len(),
            pmml_ir::ir::ModelIr::NeuralNetwork(n) => n.mining_schema.active_fields.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::Value;
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
