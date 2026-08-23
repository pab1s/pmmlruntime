//! Targets post-processing — rescale (stub v1).

use pmml_core::Value;
use pmml_ir::ir::TargetIr;

pub fn apply_targets(_targets: &[TargetIr], value: Value) -> Value {
    // v1: no rescale; return as is
    value
}
