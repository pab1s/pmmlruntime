//! Discretize — binning continuous to discrete (stub v1).

use pmml_core::Value;

pub fn eval_discretize(_value: Value, _bins: &[(f64, f64, bool, bool)]) -> Value {
    Value::Missing
}
