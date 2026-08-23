use pmml_core::Value;
use pmml_ir::ir::AssociationIr;

pub fn evaluate_association(_assoc: &AssociationIr, _values: &[Value]) -> Value {
    // Association rules: for v1, return Missing
    Value::Missing
}
