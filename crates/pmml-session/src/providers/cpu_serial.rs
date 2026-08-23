use super::ExecutionProvider;
use pmml_core::{Result, Value};
use pmml_ir::ir::{Ir, ModelIr};

pub struct CpuSerialProvider;

impl ExecutionProvider for CpuSerialProvider {
    fn name(&self) -> &str {
        "CPU"
    }

    fn evaluate(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
        // 1. DerivedFields (bytecode) — v1: noop if empty
        if !ir.derived_fields.is_empty() {
            pmml_evaluator::eval_derived_fields(&ir.derived_fields, values)
                .map_err(pmml_core::error::PmmlError::InvalidValue)?;
        }
        // 2. Model
        let predicted = match &ir.model {
            ModelIr::Tree(tree) => pmml_evaluator::models::evaluate_tree(tree, values),
            _ => return Err(pmml_core::error::PmmlError::UnsupportedMarkup("only TreeModel supported in v1".into())),
        };
        Ok(predicted)
    }
}
