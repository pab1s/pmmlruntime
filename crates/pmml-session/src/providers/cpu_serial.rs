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
                // Build name_to_id map from ir.field_names (FieldId->String invert)
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
        };
        Ok(predicted)
    }
}
