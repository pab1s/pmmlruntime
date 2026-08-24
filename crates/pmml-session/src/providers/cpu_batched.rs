use super::ExecutionProvider;
use pmml_core::{Result, Value};
use pmml_ir::ir::{Ir, ModelIr};

/// CpuBatched — Rayon-ready, shares logic with CpuSerial but designed for `par_iter`.
///
/// Each thread gets its own `&mut [Value]` slice (cloned from template) and `Arc<Ir>` is `Send+Sync`.
/// Batched scoring shards by `batch.len() / num_cpus` with chunk size 1k (see `Session::run_batch`).
pub struct CpuBatchedProvider;

impl ExecutionProvider for CpuBatchedProvider {
    fn name(&self) -> &str {
        "CPU_BATCHED"
    }

    fn evaluate(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
        // Derived fields first (if any) — per-row, thread-local values
        if !ir.derived_fields.is_empty() {
            pmml_evaluator::eval_derived_fields(&ir.derived_fields, values)
                .map_err(pmml_core::error::PmmlError::InvalidValue)?;
        }
        let predicted = match &ir.model {
            ModelIr::Tree(tree) => pmml_evaluator::models::evaluate_tree(tree, values),
            ModelIr::Regression(reg) => pmml_evaluator::models::evaluate_regression(reg, values),
            ModelIr::Mining(mining) => {
                let mut name_to_id: std::collections::HashMap<String, pmml_core::FieldId> =
                    std::collections::HashMap::new();
                for (fid, name) in &ir.field_names {
                    name_to_id.insert(name.clone(), *fid);
                }
                pmml_evaluator::models::evaluate_mining(
                    mining,
                    values,
                    &ir.field_names,
                    &ir.symbol_names,
                    &name_to_id,
                )
            }
            ModelIr::Scorecard(sc) => pmml_evaluator::models::evaluate_scorecard(sc, values),
            ModelIr::Clustering(cl) => pmml_evaluator::models::evaluate_clustering(cl, values),
            ModelIr::NaiveBayes(nb) => pmml_evaluator::models::evaluate_naive_bayes(nb, values),
            ModelIr::NearestNeighbor(nn) => pmml_evaluator::models::evaluate_nearest_neighbor(
                nn,
                values,
                Some(&ir.field_names),
                Some(&ir.symbol_names),
            ),
            ModelIr::SupportVectorMachine(svm) => {
                pmml_evaluator::models::evaluate_support_vector_machine(svm, values)
            }
            ModelIr::GeneralRegression(gr) => {
                let mut name_to_id: std::collections::HashMap<String, pmml_core::FieldId> =
                    std::collections::HashMap::new();
                for (fid, name) in &ir.field_names {
                    name_to_id.insert(name.clone(), *fid);
                }
                pmml_evaluator::models::evaluate_general_regression(
                    gr,
                    values,
                    &ir.field_names,
                    &ir.symbol_names,
                    &name_to_id,
                )
            }
            ModelIr::NeuralNetwork(nn) => {
                pmml_evaluator::models::evaluate_neural_network(nn, values)
            }
            ModelIr::Association(a) => pmml_evaluator::models::evaluate_association(a, values),
            ModelIr::RuleSet(r) => pmml_evaluator::models::evaluate_rule_set(r, values),
        };
        Ok(predicted)
    }
}

impl CpuBatchedProvider {
    /// Evaluate a batch in parallel, chunked at 1024 rows (or num_cpus sharded).
    /// Provided for direct use; `Session::run_batch` is the public API and handles
    /// input conversion + rayon sharding. This helper shows the intended inner loop.
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
