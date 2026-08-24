use pmml_core::Value;
use pmml_ir::ir::{PredicateIr, RuleSetIr, SimpleOperator, SymbolIdOrContinuous};

fn eval_predicate(pred: &PredicateIr, values: &[Value]) -> bool {
    match pred {
        PredicateIr::True => true,
        PredicateIr::Simple {
            field,
            operator,
            value,
        } => {
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
                        (Value::Continuous(a), SymbolIdOrContinuous::Continuous(b)) => {
                            match operator {
                                SimpleOperator::Equal => (a - b).abs() < 1e-9,
                                SimpleOperator::NotEqual => (a - b).abs() >= 1e-9,
                                SimpleOperator::LessThan => a < *b,
                                SimpleOperator::LessOrEqual => a <= *b,
                                SimpleOperator::GreaterThan => a > *b,
                                SimpleOperator::GreaterOrEqual => a >= *b,
                                _ => false,
                            }
                        }
                        (Value::Discrete(sid), SymbolIdOrContinuous::Symbol(s)) => match operator {
                            SimpleOperator::Equal => sid == *s,
                            SimpleOperator::NotEqual => sid != *s,
                            _ => false,
                        },
                        _ => false,
                    }
                }
            }
        }
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
            let is_contained = match actual {
                Value::Continuous(a) => array.iter().any(|v| match v {
                    SymbolIdOrContinuous::Continuous(b) => (a - b).abs() < 1e-9,
                    _ => false,
                }),
                Value::Discrete(sid) => array.iter().any(|v| match v {
                    SymbolIdOrContinuous::Symbol(s) => sid == *s,
                    _ => false,
                }),
                Value::Missing => false,
            };
            if *is_in {
                is_contained
            } else {
                !is_contained
            }
        }
        PredicateIr::Compound {
            operator,
            predicates,
        } => match operator {
            pmml_ir::ir::CompoundOperator::And => {
                predicates.iter().all(|p| eval_predicate(p, values))
            }
            pmml_ir::ir::CompoundOperator::Or => {
                predicates.iter().any(|p| eval_predicate(p, values))
            }
            pmml_ir::ir::CompoundOperator::Xor => {
                let true_count = predicates.iter().filter(|p| eval_predicate(p, values)).count();
                true_count == 1
            }
            pmml_ir::ir::CompoundOperator::Surrogate => {
                // Surrogate: true if any predicate true or if missing values cause surrogate to be considered
                // For v1, treat as Or but handle missing as false
                predicates.iter().any(|p| eval_predicate(p, values))
            }
        },
    }
}

pub fn evaluate_rule_set(rs: &RuleSetIr, values: &[Value]) -> Value {
    for rule in &rs.rules {
        if eval_predicate(&rule.predicate, values) {
            return Value::Discrete(rule.score);
        }
    }
    if let Some(default) = rs.default_score {
        return Value::Discrete(default);
    }
    Value::Missing
}
