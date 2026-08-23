//! Bytecode VM for DerivedField expressions — Level 1 graph optimization.
//! v1: only handles empty derived_fields (no-op). Full Apply/MapValues in v1.1.

use pmml_core::Value;
use pmml_ir::ir::{DerivedFieldIr, Op};

/// Evaluate a slice of derived fields in DAG order, mutating `values` array.
/// `values` is indexed by FieldId (as_usize). Caller ensures len = num_fields.
pub fn eval_derived_fields(fields: &[DerivedFieldIr], values: &mut [Value]) -> Result<(), String> {
    for df in fields {
        let v = eval_bytecode(&df.bytecode, values)?;
        let idx = df.field_id.as_usize();
        if idx < values.len() {
            values[idx] = v;
        }
    }
    Ok(())
}

fn eval_bytecode(bytecode: &[Op], values: &[Value]) -> Result<Value, String> {
    if bytecode.is_empty() {
        return Ok(Value::Missing);
    }
    // Minimal stack VM — v1 only supports PushField, PushConst, CallBuiltin.
    // Full impl will handle 100 builtins; for now return Missing if unknown.
    let mut stack: Vec<Value> = Vec::with_capacity(8);
    for op in bytecode {
        match op {
            Op::PushField(fid) => {
                let idx = fid.as_usize();
                let v = if idx < values.len() { values[idx] } else { Value::Missing };
                stack.push(v);
            }
            Op::PushConst(c) => {
                let v = match c {
                    pmml_ir::ir::SymbolIdOrContinuous::Continuous(f) => Value::Continuous(*f),
                    pmml_ir::ir::SymbolIdOrContinuous::Symbol(s) => Value::Discrete(*s),
                    pmml_ir::ir::SymbolIdOrContinuous::Missing => Value::Missing,
                };
                stack.push(v);
            }
            Op::CallBuiltin(_, _) => {
                // stub: pop args, push Missing
                if stack.is_empty() {
                    stack.push(Value::Missing);
                } else {
                    let _ = stack.pop();
                    stack.push(Value::Missing);
                }
            }
            Op::JumpIfMissing { .. } => {}
            Op::MapValues { .. } => {
                if stack.is_empty() { stack.push(Value::Missing); }
            }
        }
    }
    Ok(stack.pop().unwrap_or(Value::Missing))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_bytecode() {
        let mut vals = vec![Value::Continuous(1.0)];
        let res = eval_derived_fields(&[], &mut vals);
        assert!(res.is_ok());
    }
}
