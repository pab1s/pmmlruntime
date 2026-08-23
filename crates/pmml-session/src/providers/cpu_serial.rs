use super::ExecutionProvider;
use pmml_core::{Result, Value};
use pmml_ir::ir::{Ir, ModelIr};
use std::collections::HashMap;

pub struct CpuSerialProvider;

impl ExecutionProvider for CpuSerialProvider {
    fn name(&self) -> &str {
        "CPU"
    }

    fn evaluate(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
        if !ir.derived_fields.is_empty() {
            pmml_evaluator::eval_derived_fields(&ir.derived_fields, values)
                .map_err(pmml_core::error::PmmlError::InvalidValue)?;
        }
        let predicted = match &ir.model {
            ModelIr::Tree(tree) => pmml_evaluator::models::evaluate_tree(tree, values),
            ModelIr::Regression(reg) => pmml_evaluator::models::evaluate_regression(reg, values),
            ModelIr::Mining(mining) => {
                let mut name_to_id: HashMap<String, pmml_core::FieldId> = HashMap::new();
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
            ModelIr::NearestNeighbor(nn) => {
                pmml_evaluator::models::evaluate_nearest_neighbor(nn, values)
            }
            ModelIr::SupportVectorMachine(svm) => {
                pmml_evaluator::models::evaluate_support_vector_machine(svm, values)
            }
            ModelIr::GeneralRegression(gr) => {
                pmml_evaluator::models::evaluate_general_regression(gr, values)
            }
            ModelIr::Association(a) => pmml_evaluator::models::evaluate_association(a, values),
            ModelIr::RuleSet(r) => pmml_evaluator::models::evaluate_rule_set(r, values),
            ModelIr::NeuralNetwork(_) => {
                return Err(pmml_core::error::PmmlError::UnsupportedMarkup(
                    "NeuralNetwork not yet fully supported (stub)".into(),
                ))
            }
        };
        Ok(predicted)
    }
}
