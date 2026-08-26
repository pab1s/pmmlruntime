//! RuleSetModel evaluation — ordered first-match rule firing.
//!
//! Implements `RuleSetModel` where `RuleSet` holds an ordered list of `SimpleRule`s.
//! Each rule's [`PredicateIr`](crate::ir::PredicateIr) is tested via a local
//! `eval_predicate` (identical semantics to [`crate::engine::predicate::eval_predicate`] but
//! duplicated for bootstrapping; migration to the shared helper is pending) in document
//! order. The first firing rule's `score: SymbolId` is returned as `Discrete`. When no rule
//! fires, `defaultScore` (when present) is returned; otherwise `Missing`.
//!
//! # What belongs here
//!
//! - [`evaluate_rule_set`] — the single public entry point.
//!
//! # Performance
//!
//! `O(rules * predicate_cost)` where each predicate is a `PredicateIr` test; early exit on first match.
//! Typically `rules < 256`.

use crate::base::Value;
use crate::ir::{PredicateIr, RuleSetIr, SimpleOperator, SymbolIdOrContinuous};

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
            crate::ir::CompoundOperator::And => {
                predicates.iter().all(|p| eval_predicate(&**p, values))
            }
            crate::ir::CompoundOperator::Or => {
                predicates.iter().any(|p| eval_predicate(&**p, values))
            }
            crate::ir::CompoundOperator::Xor => {
                let true_count = predicates
                    .iter()
                    .filter(|p| eval_predicate(&**p, values))
                    .count();
                true_count == 1
            }
            crate::ir::CompoundOperator::Surrogate => {
                // Surrogate: true if any predicate true or if missing values cause surrogate to be considered
                //  treat as Or but handle missing as false
                predicates.iter().any(|p| eval_predicate(&**p, values))
            }
        },
    }
}

/// Evaluate a [`RuleSetIr`] against a dense `values` array.
///
/// Tests `rs.rules` in order; returns `Discrete(rule.score)` for the first predicate
/// that holds. When no rule fires, returns `Discrete(rs.default_score)` when `Some`,
/// otherwise `Missing`.
///
/// # Parameters
///
/// - `rs`: Lowered rule set (`RuleSetIr`) with `rules: Vec<SimpleRuleIr>` in PMML order and optional `default_score`.
/// - `values`: Dense `&[Value]` indexed by [`FieldId`](crate::base::FieldId). Out-of-bounds → `Missing`.
///
/// # Returns
///
/// `Discrete(score)` for the winning rule, `Discrete(default_score)` when no rule matches but `Some`, otherwise `Missing`.
///
/// # Panics
///
/// Never panics. All `FieldId` indexing is bounds-checked.
///
/// # Performance
///
/// `O(rules)` predicate evaluations with early exit; each predicate is `O(1)` to `O(array_len)` for `SimpleSet`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, SymbolId, Value};
/// use pmmlruntime::ir::*;
/// use pmmlruntime::engine::models::evaluate_rule_set;
///
/// let f = FieldId(0);
/// let s_yes = SymbolId(1);
/// let s_no = SymbolId(2);
/// let rs = RuleSetIr {
///     function_name: "classification".into(),
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     output: vec![],
///     default_score: Some(s_no),
///     rules: vec![
///         SimpleRuleIr { id: Some("r1".into()), score: s_yes, predicate: PredicateIr::Simple { field: f, operator: SimpleOperator::GreaterThan, value: SymbolIdOrContinuous::Continuous(10.0) } },
///         SimpleRuleIr { id: Some("r2".into()), score: s_no, predicate: PredicateIr::True },
///     ],
/// };
/// assert_eq!(evaluate_rule_set(&rs, &[Value::Continuous(15.0)]), Value::Discrete(s_yes));
/// assert_eq!(evaluate_rule_set(&rs, &[Value::Continuous(5.0)]), Value::Discrete(s_no)); // falls through to r2 (True)
/// let rs_no_default = RuleSetIr { default_score: None, ..rs.clone() };
/// // r2 still fires, but if it were absent the result would be Missing
/// ```
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
