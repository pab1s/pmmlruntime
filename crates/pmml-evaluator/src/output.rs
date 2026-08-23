//! Output — ResultFeature handling (stub v1: predictedValue only).

use pmml_core::Value;
use pmml_ir::ir::OutputFieldIr;
use std::collections::HashMap;

pub fn build_output(
    output_fields: &[OutputFieldIr],
    predicted: Value,
    _probabilities: &HashMap<String, f64>,
) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    // Always include predictedValue under field "predictedValue" or target name
    // For v1, we expose predicted under key "predictedValue" and also copy to each output field with feature predictedValue
    for of in output_fields {
        match of.feature {
            pmml_core::field::ResultFeature::PredictedValue => {
                out.insert(of.name.clone(), predicted);
            }
            pmml_core::field::ResultFeature::Probability => {
                // v1: stub 0.0 if not calculated
                out.insert(of.name.clone(), Value::Continuous(0.0));
            }
            _ => {
                out.insert(of.name.clone(), predicted);
            }
        }
    }
    if out.is_empty() {
        out.insert("predictedValue".to_string(), predicted);
    }
    out
}
