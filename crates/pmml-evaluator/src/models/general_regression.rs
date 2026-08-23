use pmml_core::Value;
use pmml_ir::ir::GeneralRegressionIr;

/// Stub GeneralRegression evaluator — for v1, return Missing.
/// Real implementation would handle ParamMatrix, PPMatrix, etc.
pub fn evaluate_general_regression(_gr: &GeneralRegressionIr, _values: &[Value]) -> Value {
    Value::Missing
}
