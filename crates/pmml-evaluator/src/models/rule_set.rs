use pmml_core::Value;
use pmml_ir::ir::RuleSetIr;

pub fn evaluate_rule_set(_rs: &RuleSetIr, _values: &[Value]) -> Value {
    // RuleSet: for v1, return Missing
    Value::Missing
}
