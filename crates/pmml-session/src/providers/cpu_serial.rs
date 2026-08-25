use super::ExecutionProvider;
use crate::batch::{Batch, BatchCtx, BatchResult};
use pmml_core::{Result, Value};
use pmml_ir::ir::{Ir, ModelIr};
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct CpuSerialProvider {
    // P0: cache reverse map FieldId->name->FieldId (built once per Session/Ir)
    // Note: Session now caches `name_to_id_std` and passes via `BatchCtx` for Mining/GeneralRegression,
    // but provider keeps this for `eval_row` calls (single) where `Ir` is available.
    cached_map: OnceLock<HashMap<String, pmml_core::FieldId>>,
}

impl CpuSerialProvider {
    pub fn new() -> Self {
        Self {
            cached_map: OnceLock::new(),
        }
    }
    fn get_or_init_map<'a>(&'a self, ir: &Ir) -> &'a HashMap<String, pmml_core::FieldId> {
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

    fn eval_batch(&self, ir: &Ir, batch: &dyn Batch, ctx: &BatchCtx) -> Result<BatchResult> {
        use crate::batch::BatchResult;
        use pmml_core::field::ResultFeature;
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
                            pmml_evaluator::models::evaluate_general_regression_with_probs(
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

    fn eval_row(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
        if !ir.derived_fields.is_empty() {
            pmml_evaluator::eval_derived_fields(&ir.derived_fields, values)
                .map_err(pmml_core::error::PmmlError::InvalidValue)?;
        }
        let predicted = match &ir.model {
            ModelIr::Tree(tree) => pmml_evaluator::models::evaluate_tree(tree, values),
            ModelIr::Regression(reg) => pmml_evaluator::models::evaluate_regression(reg, values),
            ModelIr::Mining(mining) => {
                let name_to_id = self.get_or_init_map(ir);
                pmml_evaluator::models::evaluate_mining(
                    mining,
                    values,
                    &ir.field_names,
                    &ir.symbol_names,
                    name_to_id,
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
                let name_to_id = self.get_or_init_map(ir);
                pmml_evaluator::models::evaluate_general_regression(
                    gr,
                    values,
                    &ir.field_names,
                    &ir.symbol_names,
                    name_to_id,
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
