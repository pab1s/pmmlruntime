//! Shared predicate evaluation — used by Tree and Mining (deduplicated, P5).
//! Inlined, branch-predictor friendly, matches JPMML semantics for Simple/SimpleSet/Compound.

use pmml_core::{FieldId, Value};
use pmml_ir::ir::{CompoundOperator, PredicateIr, SimpleOperator, SymbolIdOrContinuous};

#[inline(always)]
fn eval_simple(
    field: FieldId,
    operator: SimpleOperator,
    value: &SymbolIdOrContinuous,
    values: &[Value],
) -> bool {
    let idx = field.as_usize();
    let actual = if idx < values.len() {
        values[idx]
    } else {
        Value::Missing
    };
    match operator {
        SimpleOperator::IsMissing => actual.is_missing(),
        SimpleOperator::IsNotMissing => !actual.is_missing(),
        _ => {
            if actual.is_missing() {
                return false;
            }
            match (actual, value) {
                (Value::Continuous(a), SymbolIdOrContinuous::Continuous(b)) => match operator {
                    SimpleOperator::Equal => (a - b).abs() < 1e-9,
                    SimpleOperator::NotEqual => (a - b).abs() >= 1e-9,
                    SimpleOperator::LessThan => a < *b,
                    SimpleOperator::LessOrEqual => a <= *b,
                    SimpleOperator::GreaterThan => a > *b,
                    SimpleOperator::GreaterOrEqual => a >= *b,
                    _ => false,
                },
                (Value::Discrete(sid), SymbolIdOrContinuous::Symbol(s)) => match operator {
                    SimpleOperator::Equal => sid == *s,
                    SimpleOperator::NotEqual => sid != *s,
                    _ => false,
                },
                (Value::Continuous(a), SymbolIdOrContinuous::Symbol(_)) => {
                    let _ = a;
                    false
                }
                (Value::Discrete(_), SymbolIdOrContinuous::Continuous(_)) => false,
                _ => false,
            }
        }
    }
}

#[inline(always)]
pub fn eval_predicate(pred: &PredicateIr, values: &[Value]) -> bool {
    match pred {
        PredicateIr::True => true,
        PredicateIr::Simple {
            field,
            operator,
            value,
        } => eval_simple(*field, *operator, value, values),
        PredicateIr::SimpleSet {
            field,
            is_in,
            array,
        } => {
            let idx = field.as_usize();
            let actual = if idx < values.len() {
                values[idx]
            } else {
                Value::Missing
            };
            if actual.is_missing() {
                return false;
            }
            let mut found = false;
            for v in array {
                let matches = match (actual, v) {
                    (Value::Discrete(sid), SymbolIdOrContinuous::Symbol(s)) => sid == *s,
                    (Value::Continuous(a), SymbolIdOrContinuous::Continuous(b)) => {
                        (a - b).abs() < 1e-9
                    }
                    _ => false,
                };
                if matches {
                    found = true;
                    break;
                }
            }
            if *is_in {
                found
            } else {
                !found
            }
        }
        PredicateIr::Compound {
            operator,
            predicates,
        } => match operator {
            CompoundOperator::And => predicates.iter().all(|p| eval_predicate(&**p, values)),
            CompoundOperator::Or => predicates.iter().any(|p| eval_predicate(&**p, values)),
            CompoundOperator::Xor => {
                let mut c = 0;
                for p in predicates.iter() {
                    if eval_predicate(&**p, values) {
                        c += 1;
                    }
                }
                c == 1
            }
            CompoundOperator::Surrogate => {
                for p in predicates.iter() {
                    let field_missing = match &**p {
                        PredicateIr::Simple { field, .. } => {
                            let idx = field.as_usize();
                            idx < values.len() && values[idx].is_missing()
                        }
                        PredicateIr::SimpleSet { field, .. } => {
                            let idx = field.as_usize();
                            idx < values.len() && values[idx].is_missing()
                        }
                        _ => false,
                    };
                    if field_missing {
                        continue;
                    }
                    if eval_predicate(&**p, values) {
                        return true;
                    }
                    return false;
                }
                false
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::{FieldId, SymbolId, Value};
    use pmml_ir::ir::PredicateIr;

    #[test]
    fn simple_equal() {
        let vals = vec![Value::Continuous(2.0)];
        let pred = PredicateIr::Simple {
            field: FieldId(0),
            operator: SimpleOperator::Equal,
            value: SymbolIdOrContinuous::Continuous(2.0),
        };
        assert!(eval_predicate(&pred, &vals));
    }

    #[test]
    fn simple_set() {
        let vals = vec![Value::Discrete(SymbolId(1))];
        let pred = PredicateIr::SimpleSet {
            field: FieldId(0),
            is_in: true,
            array: vec![
                SymbolIdOrContinuous::Symbol(SymbolId(1)),
                SymbolIdOrContinuous::Symbol(SymbolId(2)),
            ],
        };
        assert!(eval_predicate(&pred, &vals));
    }
}
