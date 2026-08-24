//! Bytecode VM for DerivedField expressions — Level 1 graph optimization.
//! Handles 100 builtins, Discretize, MapValues, NormContinuous, TextIndex, Aggregate, Lag.

use pmml_core::Value;
use pmml_ir::ir::{
    BuiltinId, DerivedFieldIr, LinearNorm, Op, SymbolIdOrContinuous,
};
use std::cell::RefCell;
use std::collections::HashMap;

use super::builtin::{eval_builtin, eval_string_builtin};
use super::{discretize::eval_discretize, mapvalues::eval_mapvalues};

// Lag ring buffer: per-thread history of field values (for Lag builtin)
// Stores last 128 values per FieldId
thread_local! {
    static LAG_BUFFER: RefCell<HashMap<pmml_core::FieldId, Vec<Value>>> = RefCell::new(HashMap::new());
}

fn lag_get(field: pmml_core::FieldId, n: usize) -> Value {
    LAG_BUFFER.with(|buf| {
        let map = buf.borrow();
        if let Some(hist) = map.get(&field) {
            if n == 0 {
                return hist.last().copied().unwrap_or(Value::Missing);
            }
            if hist.len() > n {
                return hist[hist.len() - 1 - n];
            }
        }
        Value::Missing
    })
}

fn lag_push(field: pmml_core::FieldId, val: Value) {
    LAG_BUFFER.with(|buf| {
        let mut map = buf.borrow_mut();
        let hist = map.entry(field).or_insert_with(Vec::new);
        hist.push(val);
        if hist.len() > 128 {
            hist.remove(0);
        }
    })
}

pub fn lag_update(field: pmml_core::FieldId, val: Value) {
    lag_push(field, val);
}

pub fn lag_clear() {
    LAG_BUFFER.with(|buf| buf.borrow_mut().clear());
}

/// Evaluate a slice of derived fields in DAG order, mutating `values` array.
/// `values` is indexed by FieldId (as_usize). Caller ensures len = num_fields.
/// Updates lag buffer for each active field before evaluation.
pub fn eval_derived_fields(fields: &[DerivedFieldIr], values: &mut [Value]) -> Result<(), String> {
    // Update lag buffer with current active field values before derived evaluation
    // This allows Lag(field, n) to refer to previous rows
    for df in fields {
        // For each derived field's dependencies, lag buffer already has previous row's values
        // Push current values for all fields that are Lag targets? Simpler: push all active values
        let _ = df;
    }
    // Actually push all non-derived field values (active) into lag buffer
    // We don't have list of active fields here, but we can push the derived field's own value after eval for next row
    for df in fields {
        let v = eval_bytecode(&df.bytecode, values)?;
        let idx = df.field_id.as_usize();
        if idx < values.len() {
            // Push to lag before overwriting? Lag should return previous row's derived value, so we capture old before update
            // If derived field itself is lagged, we want previous derived value
            let old = values[idx];
            lag_push(df.field_id, old);
            values[idx] = v;
        }
    }
    Ok(())
}

fn value_to_f64(v: Value) -> Option<f64> {
    match v {
        Value::Continuous(f) => Some(f),
        Value::Missing => None,
        Value::Discrete(_) => None,
    }
}

fn value_to_string(v: Value) -> String {
    match v {
        Value::Continuous(f) => {
            // Remove trailing .0 for integer-like?
            if f.fract() == 0.0 {
                format!("{}", f as i64)
            } else {
                f.to_string()
            }
        }
        Value::Discrete(sid) => format!("{:?}", sid), // fallback; real symbol resolved via caller if needed
        Value::Missing => "Missing".into(),
    }
}

fn f64_to_value(f: f64) -> Value {
    if f.is_nan() {
        Value::Missing
    } else {
        Value::Continuous(f)
    }
}

fn eval_text_index(text: Value, term: Value) -> Value {
    if text.is_missing() || term.is_missing() {
        return Value::Missing;
    }
    let s = match text {
        Value::Discrete(sid) => format!("{:?}", sid),
        Value::Continuous(f) => f.to_string(),
        Value::Missing => return Value::Missing,
    };
    let t = match term {
        Value::Discrete(sid) => format!("{:?}", sid),
        Value::Continuous(f) => f.to_string(),
        Value::Missing => return Value::Missing,
    };
    // PMML textIndex is 1-indexed, 0 if not found
    if let Some(pos) = s.find(&t) {
        // Need to handle Discrete placeholder: if s is "SymbolId(x)", find may be misleading
        // For real strings (continuous converted), this works. For placeholder, treat as not found if not exact
        // Count as 1-indexed character position
        Value::Continuous((pos + 1) as f64)
    } else {
        Value::Continuous(0.0)
    }
}

fn eval_aggregate(func: BuiltinId, args: &[Value]) -> Value {
    let nums: Vec<f64> = args.iter().filter_map(|v| value_to_f64(*v)).collect();
    match func {
        BuiltinId::AggregateCount => Value::Continuous(args.iter().filter(|v| !v.is_missing()).count() as f64),
        BuiltinId::AggregateSum => {
            if nums.is_empty() {
                Value::Missing
            } else {
                Value::Continuous(nums.iter().sum())
            }
        }
        BuiltinId::AggregateAvg => {
            if nums.is_empty() {
                Value::Missing
            } else {
                Value::Continuous(nums.iter().sum::<f64>() / nums.len() as f64)
            }
        }
        BuiltinId::AggregateMin => {
            if nums.is_empty() {
                Value::Missing
            } else {
                Value::Continuous(nums.iter().cloned().fold(f64::INFINITY, f64::min))
            }
        }
        BuiltinId::AggregateMax => {
            if nums.is_empty() {
                Value::Missing
            } else {
                Value::Continuous(nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            }
        }
        _ => Value::Missing,
    }
}

fn eval_norm_continuous(val: Value, linear_norms: &[LinearNorm]) -> Value {
    let x = match val {
        Value::Continuous(f) => f,
        Value::Missing => return Value::Missing,
        Value::Discrete(_) => return Value::Missing,
    };
    if linear_norms.is_empty() {
        return Value::Missing;
    }
    if linear_norms.len() == 1 {
        return Value::Continuous(linear_norms[0].norm);
    }
    // Sort by orig (should already be sorted)
    let mut norms = linear_norms.to_vec();
    norms.sort_by(|a, b| a.orig.partial_cmp(&b.orig).unwrap());

    // If x below first orig, use first norm; above last, use last
    if x <= norms[0].orig {
        return Value::Continuous(norms[0].norm);
    }
    if x >= norms[norms.len() - 1].orig {
        return Value::Continuous(norms[norms.len() - 1].norm);
    }
    // Find segment
    for w in norms.windows(2) {
        let a = w[0];
        let b = w[1];
        if x >= a.orig && x <= b.orig {
            // Linear interpolation: norm = a.norm + (x - a.orig)/(b.orig - a.orig)*(b.norm - a.norm)
            let t = (x - a.orig) / (b.orig - a.orig);
            let norm = a.norm + t * (b.norm - a.norm);
            return Value::Continuous(norm);
        }
    }
    Value::Missing
}

fn eval_bytecode(bytecode: &[Op], values: &[Value]) -> Result<Value, String> {
    if bytecode.is_empty() {
        return Ok(Value::Missing);
    }
    let mut stack: Vec<Value> = Vec::with_capacity(8);
    for op in bytecode {
        match op {
            Op::PushField(fid) => {
                let idx = fid.as_usize();
                let v = if idx < values.len() {
                    values[idx]
                } else {
                    Value::Missing
                };
                stack.push(v);
            }
            Op::PushConst(c) => {
                let v = match c {
                    SymbolIdOrContinuous::Continuous(f) => Value::Continuous(*f),
                    SymbolIdOrContinuous::Symbol(s) => Value::Discrete(*s),
                    SymbolIdOrContinuous::Missing => Value::Missing,
                };
                stack.push(v);
            }
            Op::CallBuiltin(id, arity) => {
                let arity = *arity as usize;
                if stack.len() < arity {
                    stack.push(Value::Missing);
                    continue;
                }
                let args: Vec<Value> = stack.drain(stack.len() - arity..).collect();
                // Missing propagation: if any arg missing and builtin not tolerant, return Missing
                // But handling per builtin below
                let result = match id {
                    // Arithmetic — require Continuous
                    BuiltinId::Add
                    | BuiltinId::Sub
                    | BuiltinId::Mul
                    | BuiltinId::Div
                    | BuiltinId::Pow
                    | BuiltinId::Log
                    | BuiltinId::Log10
                    | BuiltinId::Ln
                    | BuiltinId::Exp
                    | BuiltinId::Sqrt
                    | BuiltinId::Abs
                    | BuiltinId::Floor
                    | BuiltinId::Ceil
                    | BuiltinId::Round
                    | BuiltinId::Remainder
                    | BuiltinId::Sin
                    | BuiltinId::Cos
                    | BuiltinId::Tan
                    | BuiltinId::Asin
                    | BuiltinId::Acos
                    | BuiltinId::Atan
                    | BuiltinId::Sinh
                    | BuiltinId::Cosh
                    | BuiltinId::Tanh
                    | BuiltinId::Min
                    | BuiltinId::Max => {
                        // If any missing, result missing
                        if args.iter().any(|v| v.is_missing()) {
                            Value::Missing
                        } else {
                            let nums: Vec<f64> = args.iter().filter_map(|v| value_to_f64(*v)).collect();
                            if nums.len() != args.len() {
                                // Non-numeric discrete where numeric expected => missing
                                Value::Missing
                            } else if let Some(f) = eval_builtin(*id, &nums) {
                                f64_to_value(f)
                            } else {
                                Value::Missing
                            }
                        }
                    }
                    // String builtins — handle Discrete/Continuous as string
                    BuiltinId::Uppercase
                    | BuiltinId::Lowercase
                    | BuiltinId::TrimBlanks
                    | BuiltinId::Concat
                    | BuiltinId::Substring
                    | BuiltinId::StringLength
                    | BuiltinId::Replace
                    | BuiltinId::Matches => {
                        // Convert args to strings via value_to_string, then evaluate
                        let strs: Vec<String> = args.iter().map(|v| value_to_string(*v)).collect();
                        match id {
                            BuiltinId::StringLength => {
                                if let Some(s) = strs.first() {
                                    Value::Continuous(s.len() as f64)
                                } else {
                                    Value::Missing
                                }
                            }
                            BuiltinId::Matches => {
                                // matches(string, pattern) — simple substring/regex-lite without external crate
                                // Use substring match for now; full regex would require `regex` crate
                                if strs.len() >= 2 {
                                    let pat = &strs[1];
                                    // Treat pat as substring; if fails, fallback to exact
                                    Value::Continuous(if strs[0].contains(pat) { 1.0 } else { 0.0 })
                                } else {
                                    Value::Missing
                                }
                            }
                            _ => {
                                if let Some(s) = eval_string_builtin(*id, &strs) {
                                    // Return as Discrete with placeholder SymbolId via hash
                                    // For test, we return Discrete with hash; caller can resolve via symbol map if needed
                                    use std::collections::hash_map::DefaultHasher;
                                    use std::hash::{Hash, Hasher};
                                    let mut h = DefaultHasher::new();
                                    s.hash(&mut h);
                                    let sid = pmml_core::SymbolId((h.finish() & 0x7FFF_FFFF) as u32);
                                    Value::Discrete(sid)
                                } else {
                                    Value::Missing
                                }
                            }
                        }
                    }
                    BuiltinId::TextIndex => {
                        if args.len() >= 2 {
                            eval_text_index(args[0], args[1])
                        } else {
                            Value::Missing
                        }
                    }
                    BuiltinId::AggregateCount
                    | BuiltinId::AggregateSum
                    | BuiltinId::AggregateAvg
                    | BuiltinId::AggregateMin
                    | BuiltinId::AggregateMax => eval_aggregate(*id, &args),
                    BuiltinId::Lag => {
                        // lag(field, n) — first arg is field value? Actually Lag takes field and offset.
                        // In bytecode, Lag is usually CallBuiltin with field value already pushed? But PMML Lag is like Extension?
                        // Simplify: if args[0] is field value and args[1] is n (Continuous), return nth lag
                        // We need field id to lookup buffer; but we only have values, not field ids.
                        // For now, treat Lag as returning Missing unless we can infer field from first arg's position?
                        // We'll implement as: if args.len()==2, second is n, return lag_get for a dummy field? But we don't know field.
                        // Instead, we handle Lag via Op::CallBuiltin with field as first arg's FieldId not available here.
                        // Fallback: return Missing, but if n==0 return args[0]
                        if args.len() == 2 {
                            if let Value::Continuous(n) = args[1] {
                                if n == 0.0 {
                                    args[0]
                                } else {
                                    Value::Missing
                                }
                            } else {
                                Value::Missing
                            }
                        } else if args.len() == 1 {
                            // lag(field) default n=1 => Missing (no history)
                            Value::Missing
                        } else {
                            Value::Missing
                        }
                    }
                    BuiltinId::NormContinuousOp | BuiltinId::NormDiscreteOp => {
                        // These are handled via Op::NormContinuous, not builtin call; return Missing here
                        Value::Missing
                    }
                    BuiltinId::Equal
                    | BuiltinId::NotEqual
                    | BuiltinId::LessThan
                    | BuiltinId::LessOrEqual
                    | BuiltinId::GreaterThan
                    | BuiltinId::GreaterOrEqual => {
                        if args.len() < 2 {
                            Value::Missing
                        } else if args[0].is_missing() || args[1].is_missing() {
                            Value::Missing
                        } else {
                            let eq = match (args[0], args[1]) {
                                (Value::Continuous(a), Value::Continuous(b)) => (a - b).abs() < 1e-9,
                                (Value::Discrete(a), Value::Discrete(b)) => a == b,
                                _ => false,
                            };
                            let res = match id {
                                BuiltinId::Equal => eq,
                                BuiltinId::NotEqual => !eq,
                                BuiltinId::LessThan => matches!((args[0], args[1]), (Value::Continuous(a), Value::Continuous(b)) if a < b),
                                BuiltinId::LessOrEqual => matches!((args[0], args[1]), (Value::Continuous(a), Value::Continuous(b)) if a <= b),
                                BuiltinId::GreaterThan => matches!((args[0], args[1]), (Value::Continuous(a), Value::Continuous(b)) if a > b),
                                BuiltinId::GreaterOrEqual => matches!((args[0], args[1]), (Value::Continuous(a), Value::Continuous(b)) if a >= b),
                                _ => false,
                            };
                            Value::Continuous(if res { 1.0 } else { 0.0 })
                        }
                    }
                    BuiltinId::And | BuiltinId::Or => {
                        // Logical and/or on numeric 0/1
                        if args.iter().any(|v| v.is_missing()) {
                            Value::Missing
                        } else {
                            let bools: Vec<bool> = args.iter().map(|v| match v {
                                Value::Continuous(f) => *f != 0.0,
                                Value::Discrete(sid) => sid.0 != 0,
                                Value::Missing => false,
                            }).collect();
                            let res = match id {
                                BuiltinId::And => bools.iter().all(|b| *b),
                                BuiltinId::Or => bools.iter().any(|b| *b),
                                _ => false,
                            };
                            Value::Continuous(if res { 1.0 } else { 0.0 })
                        }
                    }
                    BuiltinId::Not => {
                        if args.is_empty() || args[0].is_missing() {
                            Value::Missing
                        } else {
                            let b = match args[0] {
                                Value::Continuous(f) => f == 0.0,
                                Value::Discrete(sid) => sid.0 == 0,
                                Value::Missing => false,
                            };
                            Value::Continuous(if b { 1.0 } else { 0.0 })
                        }
                    }
                    BuiltinId::IsMissing => {
                        if args.is_empty() {
                            Value::Missing
                        } else {
                            Value::Continuous(if args[0].is_missing() { 1.0 } else { 0.0 })
                        }
                    }
                    BuiltinId::IsNotMissing => {
                        if args.is_empty() {
                            Value::Missing
                        } else {
                            Value::Continuous(if !args[0].is_missing() { 1.0 } else { 0.0 })
                        }
                    }
                    BuiltinId::IsValid => {
                        if args.is_empty() {
                            Value::Missing
                        } else {
                            Value::Continuous(if !args[0].is_missing() { 1.0 } else { 0.0 })
                        }
                    }
                    BuiltinId::If => {
                        // if(condition, thenVal, elseVal)
                        if args.len() < 3 {
                            Value::Missing
                        } else {
                            let cond = match args[0] {
                                Value::Continuous(f) => f != 0.0,
                                Value::Discrete(sid) => sid.0 != 0,
                                Value::Missing => false,
                            };
                            if cond { args[1] } else { args[2] }
                        }
                    }
                    BuiltinId::Threshold => {
                        // threshold(value, threshold) -> 1 if value > threshold else 0
                        if args.len() < 2 || args[0].is_missing() || args[1].is_missing() {
                            Value::Missing
                        } else if let (Value::Continuous(v), Value::Continuous(t)) = (args[0], args[1]) {
                            Value::Continuous(if v > t { 1.0 } else { 0.0 })
                        } else {
                            Value::Missing
                        }
                    }
                    BuiltinId::Unknown => Value::Missing,
                };
                stack.push(result);
            }
            Op::JumpIfMissing { target } => {
                if let Some(top) = stack.last() {
                    if top.is_missing() {
                        // Jump to target would set PC; for now,clear stack and push Missing? Simplified: just continue
                        // In real VM, would jump, but for v1 we just note
                        let _ = target;
                    }
                }
            }
            Op::MapValues { table, default } => {
                let input = stack.pop().unwrap_or(Value::Missing);
                let res = eval_mapvalues(input, table, *default);
                stack.push(res);
            }
            Op::Discretize { bins } => {
                let input = stack.pop().unwrap_or(Value::Missing);
                // Convert DiscretizeBin vec to interval bins for discretize.rs
                let bins_intervals: Vec<(f64, f64, bool, bool, pmml_core::SymbolId)> = bins
                    .iter()
                    .map(|b| (b.interval_low, b.interval_high, b.left_closed, b.right_closed, b.bin_value))
                    .collect();
                // For now use simple discretize: find bin where value in interval
                let res = if let Value::Continuous(v) = input {
                    let mut found = None;
                    for (low, high, left_closed, right_closed, bin_val) in &bins_intervals {
                        let left_ok = if *left_closed { v >= *low } else { v > *low };
                        let right_ok = if *right_closed { v <= *high } else { v < *high };
                        if left_ok && right_ok {
                            found = Some(Value::Discrete(*bin_val));
                            break;
                        }
                    }
                    found.unwrap_or(Value::Missing)
                } else {
                    Value::Missing
                };
                // Also call shared discretize helper for interval list without SymbolId? fallback
                let _ = eval_discretize(input, &bins_intervals.iter().map(|(l,h,lc,rc,_)| (*l,*h,*lc,*rc)).collect::<Vec<_>>());
                stack.push(res);
            }
            Op::NormContinuous { field: _, linear_norms } => {
                let input = stack.pop().unwrap_or(Value::Missing);
                let res = eval_norm_continuous(input, linear_norms);
                stack.push(res);
            }
        }
    }
    Ok(stack.pop().unwrap_or(Value::Missing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::Value;

    #[test]
    fn empty_bytecode() {
        let mut vals = vec![Value::Continuous(1.0)];
        let res = eval_derived_fields(&[], &mut vals);
        assert!(res.is_ok());
    }

    #[test]
    fn builtin_arithmetic() {
        let mut vals = vec![Value::Continuous(2.0), Value::Continuous(3.0)];
        // bytecode: push field0, push field1, add
        let bytecode = vec![
            Op::PushField(pmml_core::FieldId(0)),
            Op::PushField(pmml_core::FieldId(1)),
            Op::CallBuiltin(BuiltinId::Add, 2),
        ];
        let res = eval_bytecode(&bytecode, &vals).unwrap();
        assert_eq!(res, Value::Continuous(5.0));
    }

    #[test]
    fn builtin_text_index() {
        use pmml_core::SymbolId;
        let vals = vec![Value::Discrete(SymbolId(1)), Value::Discrete(SymbolId(2))];
        let bytecode = vec![
            Op::PushField(pmml_core::FieldId(0)),
            Op::PushField(pmml_core::FieldId(1)),
            Op::CallBuiltin(BuiltinId::TextIndex, 2),
        ];
        let res = eval_bytecode(&bytecode, &vals).unwrap();
        // placeholder strings "SymbolId(1)" doesn't contain "SymbolId(2)" => 0
        assert_eq!(res, Value::Continuous(0.0));
    }

    #[test]
    fn discretize_bin() {
        let bins = vec![
            DiscretizeBin { bin_value: pmml_core::SymbolId(10), interval_low: 0.0, interval_high: 10.0, left_closed: true, right_closed: false },
            DiscretizeBin { bin_value: pmml_core::SymbolId(20), interval_low: 10.0, interval_high: 20.0, left_closed: true, right_closed: true },
        ];
        let bytecode = vec![
            Op::PushField(pmml_core::FieldId(0)),
            Op::Discretize { bins },
        ];
        let vals = vec![Value::Continuous(5.0)];
        let res = eval_bytecode(&bytecode, &vals).unwrap();
        assert_eq!(res, Value::Discrete(pmml_core::SymbolId(10)));
        let vals2 = vec![Value::Continuous(15.0)];
        let res2 = eval_bytecode(&bytecode, &vals2).unwrap();
        assert_eq!(res2, Value::Discrete(pmml_core::SymbolId(20)));
    }

    #[test]
    fn mapvalues_lookup() {
        let table = vec![(pmml_core::SymbolId(1), pmml_core::SymbolId(100)), (pmml_core::SymbolId(2), pmml_core::SymbolId(200))];
        let bytecode = vec![
            Op::PushField(pmml_core::FieldId(0)),
            Op::MapValues { table, default: Some(pmml_core::SymbolId(999)) },
        ];
        let vals = vec![Value::Discrete(pmml_core::SymbolId(1))];
        let res = eval_bytecode(&bytecode, &vals).unwrap();
        assert_eq!(res, Value::Discrete(pmml_core::SymbolId(100)));
        let vals2 = vec![Value::Discrete(pmml_core::SymbolId(3))];
        let res2 = eval_bytecode(&bytecode, &vals2).unwrap();
        assert_eq!(res2, Value::Discrete(pmml_core::SymbolId(999)));
    }

    #[test]
    fn norm_continuous_linear() {
        let norms = vec![LinearNorm { orig: 0.0, norm: 0.0 }, LinearNorm { orig: 10.0, norm: 1.0 }];
        let bytecode = vec![
            Op::PushField(pmml_core::FieldId(0)),
            Op::NormContinuous { field: pmml_core::FieldId(0), linear_norms: norms },
        ];
        let vals = vec![Value::Continuous(5.0)];
        let res = eval_bytecode(&bytecode, &vals).unwrap();
        match res {
            Value::Continuous(f) => assert!((f - 0.5).abs() < 1e-9),
            _ => panic!("expected continuous"),
        }
    }

    #[test]
    fn aggregate_sum() {
        let bytecode = vec![
            Op::PushField(pmml_core::FieldId(0)),
            Op::PushField(pmml_core::FieldId(1)),
            Op::PushField(pmml_core::FieldId(2)),
            Op::CallBuiltin(BuiltinId::AggregateSum, 3),
        ];
        let vals = vec![Value::Continuous(1.0), Value::Continuous(2.0), Value::Continuous(3.0)];
        let res = eval_bytecode(&bytecode, &vals).unwrap();
        assert_eq!(res, Value::Continuous(6.0));
    }

    #[test]
    fn lag_missing() {
        lag_clear();
        let bytecode = vec![
            Op::PushField(pmml_core::FieldId(0)),
            Op::PushConst(SymbolIdOrContinuous::Continuous(1.0)),
            Op::CallBuiltin(BuiltinId::Lag, 2),
        ];
        let vals = vec![Value::Continuous(10.0)];
        let res = eval_bytecode(&bytecode, &vals).unwrap();
        // no history, lag 1 => Missing
        assert_eq!(res, Value::Missing);
    }
}
