//! `CpuSerialProvider` — single-threaded execution, no `rayon`.
//!
//! Mirrors `Session::run` single-row path and is the default for `SessionOptions::default()`.
//! It is also the fallback for `CpuBatched` when `n < 256`.

use super::ExecutionProvider;
use crate::base::{Result, Value};
use crate::ir::{Ir, ModelIr};
use crate::session::batch::{Batch, BatchCtx, BatchResult};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Serial CPU execution provider — evaluates rows one by one on the calling thread.
///
/// Caches a `HashMap<String, FieldId>` reverse map from `Ir.field_names` for
/// `MiningModel` / `GeneralRegression` which need `field_names` + `symbol_names` + `name_to_id`.
/// The map is built once via `OnceLock` and is `Send+Sync` because `Self` is.
///
/// # Performance
///
/// No threading overhead; ~402 ns per row for Tree Iris. Use [`CpuBatchedProvider`](super::CpuBatchedProvider)
/// for `>256` rows where `rayon` sharding pays off.
///
/// # Concurrency
///
/// `Self` is `Send+Sync`; `eval_row` is `&self` and borrows `Ir` immutably. Concurrent calls
/// on the same provider are safe, but they execute serially on each caller's thread.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::providers::{CpuSerialProvider, ExecutionProvider};
/// let p = CpuSerialProvider::new();
/// assert_eq!(p.name(), "CPU");
/// ```
pub struct CpuSerialProvider {
    // P0: cache reverse map FieldId->name->FieldId (built once per Session/Ir)
    // Note: Session now caches `name_to_id_std` and passes via `BatchCtx` for Mining/GeneralRegression,
    // but provider keeps this for `eval_row` calls (single) where `Ir` is available.
    cached_map: OnceLock<HashMap<String, crate::base::FieldId>>,
}

impl CpuSerialProvider {
    /// Create a new serial provider (no allocation until first `eval_row` for `MiningModel`).
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::providers::{CpuSerialProvider, ExecutionProvider};
    /// let p = CpuSerialProvider::new();
    /// let p2 = CpuSerialProvider::default();
    /// assert_eq!(p.name(), p2.name());
    /// ```
    pub fn new() -> Self {
        Self {
            cached_map: OnceLock::new(),
        }
    }
    /// Get or initialize the cached `name -> FieldId` map from `Ir.field_names`.
    fn get_or_init_map<'a>(&'a self, ir: &Ir) -> &'a HashMap<String, crate::base::FieldId> {
        self.cached_map.get_or_init(|| {
            let mut m = HashMap::new();
            for (fid, name) in &ir.field_names {
                m.insert(name.clone(), *fid);
            }
            m
        })
    }
}

impl Default for CpuSerialProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionProvider for CpuSerialProvider {
    fn name(&self) -> &str {
        "CPU"
    }

    /// Evaluate a batch serially, one row at a time.
    ///
    /// For each `row_idx` it calls `with_value_buffer(needed, |values| { batch.materialize_row(...); eval_row or GeneralRegression })`
    /// and collects `HashMap<String, Value>` outputs. No `rayon`. For `GeneralRegression` it
    /// also emits `Probability_*` keys.
    ///
    /// # Parameters
    ///
    /// - `ir`: model.
    /// - `batch`: `&dyn Batch` (row-major or columnar).
    /// - `ctx`: `BatchCtx` with `max_field_id`, `output_fields`, `symbol_names_vec` etc.
    ///
    /// # Returns
    ///
    /// `BatchResult::Rows` preserving input order.
    ///
    /// # Errors
    ///
    /// Propagates `PmmlError` from `materialize_row` or `eval_row`.
    fn eval_batch(&self, ir: &Ir, batch: &dyn Batch, ctx: &BatchCtx) -> Result<BatchResult> {
        use crate::base::field::ResultFeature;
        use crate::session::batch::BatchResult;
        if batch.is_empty() {
            return Ok(BatchResult::Rows(Vec::new()));
        }
        // Threshold: for tiny batches (<256) overhead of batch path not worth, but serial is always serial anyway.
        // We still use serial loop.
        let needed = ctx.max_field_id.max(ir.num_fields() + 4);
        let mut results = Vec::with_capacity(batch.len());
        // Reusable buffer via thread_local or stack — use same as Session::with_value_buffer but here we need per-row.
        // For serial, we can allocate Vec<Value> per row reuse via thread_local to avoid stack overflow for large needed.
        // Use crate::session::with_value_buffer if pub, else replicate.
        for row_idx in 0..batch.len() {
            let out = crate::session::with_value_buffer(
                needed,
                |values| -> Result<HashMap<String, Value>> {
                    batch.materialize_row(row_idx, values, ctx)?;
                    // Handle GeneralRegression specially to get probs for output
                    if let ModelIr::GeneralRegression(gr) = &ir.model {
                        let (predicted, probs) =
                            crate::engine::models::evaluate_general_regression_with_probs(
                                gr,
                                &values[..needed],
                                &ir.field_names,
                                &ir.symbol_names,
                                ctx.name_to_id_std,
                            );
                        let mut output = HashMap::with_capacity(ctx.output_fields.len().max(1) + 2);
                        for of in ctx.output_fields {
                            match of.feature {
                                ResultFeature::Probability => {
                                    if let Some(cat_sid) = of.value {
                                        if let Some(cat_str) = ctx
                                            .symbol_names_vec
                                            .get(cat_sid.0 as usize)
                                            .filter(|s| !s.is_empty())
                                        {
                                            if let Some(p) = probs.get(cat_str) {
                                                output
                                                    .insert(of.name.clone(), Value::Continuous(*p));
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
                        if let Some(tname) = ctx.target_name {
                            final_out.entry(tname.clone()).or_insert(predicted);
                        }
                        final_out
                            .entry("predictedValue".to_string())
                            .or_insert(predicted);
                        for (k, v) in &probs {
                            final_out.entry(k.clone()).or_insert(Value::Continuous(*v));
                            let prob_name = format!("Probability_{}", k);
                            final_out.entry(prob_name).or_insert(Value::Continuous(*v));
                        }
                        Ok(final_out)
                    } else {
                        let predicted = self.eval_row(ir, values)?;
                        let mut output = HashMap::with_capacity(ctx.output_fields.len().max(1) + 2);
                        if ctx.output_fields.is_empty() {
                            output.insert("predictedValue".to_string(), predicted);
                        } else {
                            for of in ctx.output_fields {
                                match of.feature {
                                    ResultFeature::PredictedValue => {
                                        output.insert(of.name.clone(), predicted);
                                    }
                                    ResultFeature::Probability => {
                                        output.insert(of.name.clone(), Value::Continuous(0.0));
                                    }
                                    _ => {
                                        output.insert(of.name.clone(), predicted);
                                    }
                                }
                            }
                        }
                        let mut final_out = output;
                        if let Some(tname) = ctx.target_name {
                            final_out.entry(tname.clone()).or_insert(predicted);
                        }
                        final_out
                            .entry("predictedValue".to_string())
                            .or_insert(predicted);
                        Ok(final_out)
                    }
                },
            )?;
            results.push(out);
        }
        Ok(BatchResult::Rows(results))
    }

    /// Evaluate a single row's `values` → predicted `Value`.
    ///
    /// First evaluates `DerivedFields` via `crate::engine::eval_derived_fields` (if any),
    /// then dispatches to the per-model evaluator (`evaluate_tree`, `evaluate_regression`, `evaluate_mining`, etc.).
    /// For `MiningModel` / `GeneralRegression` it passes the cached `HashMap<String, FieldId>` and symbol maps.
    ///
    /// # Parameters
    ///
    /// - `ir`: model containing `derived_fields` and `ModelIr` variant.
    /// - `values`: `&mut [Value]` indexed by `FieldId`.
    ///
    /// # Returns
    ///
    /// Predicted `Value`.
    ///
    /// # Errors
    ///
    /// Returns `PmmlError::InvalidValue` if derived field evaluation fails.
    fn eval_row(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
        // Install symbol map for string/date builtins (per-row thread-local, cheap for <50 entries)
        // This is needed because vm decodes Discrete SymbolId via thread_local SYMBOL_STR_MAP
        // For GeneralRegression and other models, derived fields may use string functions
        if !ir.symbol_names.is_empty() {
            crate::engine::transform::vm::vm_set_symbol_map(ir.symbol_names.clone());
        }
        if !ir.derived_fields.is_empty() {
            crate::engine::eval_derived_fields(&ir.derived_fields, values)
                .map_err(crate::base::error::PmmlError::InvalidValue)?;
        }
        let predicted = match &ir.model {
            ModelIr::Tree(tree) => crate::engine::models::evaluate_tree(tree, values),
            ModelIr::Regression(reg) => crate::engine::models::evaluate_regression(reg, values),
            ModelIr::Mining(mining) => {
                let name_to_id = self.get_or_init_map(ir);
                crate::engine::models::evaluate_mining(
                    mining,
                    values,
                    &ir.field_names,
                    &ir.symbol_names,
                    name_to_id,
                )
            }
            ModelIr::Scorecard(sc) => crate::engine::models::evaluate_scorecard(sc, values),
            ModelIr::Clustering(cl) => crate::engine::models::evaluate_clustering(cl, values),
            ModelIr::NaiveBayes(nb) => crate::engine::models::evaluate_naive_bayes(nb, values),
            ModelIr::NearestNeighbor(nn) => crate::engine::models::evaluate_nearest_neighbor(
                nn,
                values,
                Some(&ir.field_names),
                Some(&ir.symbol_names),
            ),
            ModelIr::SupportVectorMachine(svm) => {
                crate::engine::models::evaluate_support_vector_machine(svm, values)
            }
            ModelIr::GeneralRegression(gr) => {
                let name_to_id = self.get_or_init_map(ir);
                crate::engine::models::evaluate_general_regression(
                    gr,
                    values,
                    &ir.field_names,
                    &ir.symbol_names,
                    name_to_id,
                )
            }
            ModelIr::NeuralNetwork(nn) => {
                crate::engine::models::evaluate_neural_network(nn, values)
            }
            ModelIr::Association(a) => crate::engine::models::evaluate_association(a, values),
            ModelIr::RuleSet(r) => crate::engine::models::evaluate_rule_set(r, values),
            ModelIr::AnomalyDetection(ad) => {
                crate::engine::models::evaluate_anomaly_detection(ad, values)
            }
            ModelIr::Baseline(b) => crate::engine::models::evaluate_baseline(b, values),
            ModelIr::GaussianProcess(gp) => {
                crate::engine::models::evaluate_gaussian_process(gp, values)
            }
            ModelIr::Text(t) => {
                crate::engine::models::evaluate_text(t, values, Some(&ir.symbol_names))
            }
            ModelIr::TimeSeries(ts) => crate::engine::models::evaluate_time_series(ts, values),
            ModelIr::Sequence(s) => crate::engine::models::evaluate_sequence(s, values),
            ModelIr::BayesianNetwork(bn) => {
                crate::engine::models::evaluate_bayesian_network(bn, values)
            }
        };
        Ok(predicted)
    }
}
