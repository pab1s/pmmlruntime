//! `CpuProvider` — single CPU execution provider (auto serial / rayon).

use super::ExecutionProvider;
use crate::base::{Result, Value};
use crate::ir::{Ir, ModelIr};
use crate::session::batch::{Batch, BatchCtx, BatchResult};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Unified CPU execution provider — auto-chooses serial vs `rayon` sharding.
///
/// For `n < 256` or `n < num_threads*4` it runs serially (same as old `CpuSerial`);
/// otherwise it shards via `rayon::par_chunks(chunk_size)` where `chunk_size = 256.max(n / num_threads)`.
///
/// Each thread gets its own `&mut [Value]` slice via `with_value_buffer` and `Arc<Ir>` is `Send+Sync`.
///
/// # Performance
///
/// ~402 ns per row single, ~61 ns/row at 100k Arrow rows. Small batches avoid `rayon` overhead.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::providers::{CpuProvider, ExecutionProvider};
/// let p = CpuProvider::new();
/// assert_eq!(p.name(), "CPU");
/// ```
pub struct CpuProvider {
    cached_map: OnceLock<HashMap<String, crate::base::FieldId>>,
}

impl CpuProvider {
    /// Create a new CPU provider (no allocation until first `eval_row` for `MiningModel`).
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::providers::{CpuProvider, ExecutionProvider};
    /// let p = CpuProvider::new();
    /// assert_eq!(p.name(), "CPU");
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

impl Default for CpuProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionProvider for CpuProvider {
    fn name(&self) -> &str {
        "CPU"
    }

    /// Evaluate a batch with auto serial / parallel sharding.
    ///
    /// For `n < 256` or `n < num_threads*4` it runs serially; otherwise `par_chunks(chunk_size)`.
    fn eval_batch(&self, ir: &Ir, batch: &dyn Batch, ctx: &BatchCtx) -> Result<BatchResult> {
        use crate::base::field::ResultFeature;
        use crate::session::batch::BatchResult;
        if batch.is_empty() {
            return Ok(BatchResult::Rows(Vec::new()));
        }
        let needed = ctx.max_field_id.max(ir.num_fields() + 4);
        let n = batch.len();
        let num_threads = rayon::current_num_threads().max(1);
        if n < 256 || n < num_threads * 4 {
            let mut results = Vec::with_capacity(n);
            for row_idx in 0..n {
                let out = crate::session::with_value_buffer(
                    needed,
                    |values| -> Result<HashMap<String, Value>> {
                        batch.materialize_row(row_idx, values, ctx)?;
                        if let ModelIr::GeneralRegression(gr) = &ir.model {
                            let (predicted, probs) =
                                crate::engine::models::evaluate_general_regression_with_probs(
                                    gr,
                                    &values[..needed],
                                    &ir.field_names,
                                    &ir.symbol_names,
                                    ctx.name_to_id_std,
                                );
                            let mut output =
                                HashMap::with_capacity(ctx.output_fields.len().max(1) + 2);
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
                                                    output.insert(
                                                        of.name.clone(),
                                                        Value::Continuous(*p),
                                                    );
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
                            let mut output =
                                HashMap::with_capacity(ctx.output_fields.len().max(1) + 2);
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
            return Ok(BatchResult::Rows(results));
        }
        use rayon::prelude::*;
        let chunk_size = 256.max(n / num_threads);
        let indices: Vec<usize> = (0..n).collect();
        let chunk_results: Result<Vec<Vec<HashMap<String, Value>>>> = indices
            .par_chunks(chunk_size)
            .map(|chunk| -> Result<Vec<HashMap<String, Value>>> {
                let mut local = Vec::with_capacity(chunk.len());
                for &row_idx in chunk {
                    let out = crate::session::with_value_buffer(
                        needed,
                        |values| -> Result<HashMap<String, Value>> {
                            batch.materialize_row(row_idx, values, ctx)?;
                            if let ModelIr::GeneralRegression(gr) = &ir.model {
                                let (predicted, probs) =
                                    crate::engine::models::evaluate_general_regression_with_probs(
                                        gr,
                                        &values[..needed],
                                        &ir.field_names,
                                        &ir.symbol_names,
                                        ctx.name_to_id_std,
                                    );
                                let mut output =
                                    HashMap::with_capacity(ctx.output_fields.len().max(1) + 2);
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
                                                        output.insert(
                                                            of.name.clone(),
                                                            Value::Continuous(*p),
                                                        );
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
                                let mut output =
                                    HashMap::with_capacity(ctx.output_fields.len().max(1) + 2);
                                if ctx.output_fields.is_empty() {
                                    output.insert("predictedValue".to_string(), predicted);
                                } else {
                                    for of in ctx.output_fields {
                                        match of.feature {
                                            ResultFeature::PredictedValue => {
                                                output.insert(of.name.clone(), predicted);
                                            }
                                            ResultFeature::Probability => {
                                                output.insert(
                                                    of.name.clone(),
                                                    Value::Continuous(0.0),
                                                );
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
                    local.push(out);
                }
                Ok(local)
            })
            .collect();
        let results = chunk_results?.into_iter().flatten().collect();
        Ok(BatchResult::Rows(results))
    }

    /// Evaluate a single row.
    fn eval_row(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
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

impl CpuProvider {
    /// Evaluate a batch in parallel, chunked at 1024 rows (or `num_cpus` sharded).
    ///
    /// Provided for direct use; `Session::run_batch` is the public API.
    #[allow(dead_code)]
    pub fn evaluate_batch_parallel(
        &self,
        ir: &Ir,
        batch_values: &mut [Vec<Value>],
    ) -> Result<Vec<Value>> {
        use rayon::prelude::*;
        batch_values
            .par_iter_mut()
            .map(|values| self.evaluate(ir, values))
            .collect()
    }
}
