//! Scorecard evaluation — summed `initialScore` plus per-characteristic `partialScore`.
//!
//! A `Scorecard` is a special regression-like model where each `Characteristic`
//! contributes either the `partialScore` of its first matching `Attribute`
//! predicate or its `baselineScore` when no attribute matches. Reason codes are
//! collected but not yet ranked per `reasonCodeAlgorithm`; the score itself is
//! returned as `Continuous`.
//!
//! # What belongs here
//!
//! - [`evaluate_scorecard`] — the single public entry point.
//!
//! # Performance
//!
//! `O(characteristics * attributes)` predicate tests; typically < 50 predicates total.
//! No allocation beyond the `reason_codes` vec (currently unused).

use pmml_core::Value;
use pmml_ir::ir::{PredicateIr, ScorecardIr, SimpleOperator};

/// Evaluate a [`ScorecardIr`] against a dense `values` array.
///
/// Computes `total = initialScore + Σ characteristic_contribution` where each
/// `Characteristic` scans its `Attribute`s in document order and adds the first
/// `partialScore` whose predicate holds (via predicate logic mirroring
/// [`crate::predicate::eval_predicate`]). When none matches, `baselineScore` is added.
/// `reasonCode` propagation is tracked internally but not yet exposed via `Output`
/// (the returned value is the raw `total`).
///
/// # Parameters
///
/// - `scorecard`: Lowered scorecard (`ScorecardIr`) with `initial_score`, `characteristics`, and per-attribute `PredicateIr`.
/// - `values`: Dense `&[Value]` indexed by [`FieldId`](pmml_core::FieldId). Out-of-bounds fields are treated as `Missing`.
///
/// # Returns
///
/// `Continuous(total)` score. `Missing` is never returned (even for missing inputs a baseline applies).
///
/// # Panics
///
/// Never panics. All `FieldId` indexing is bounds-checked.
///
/// # Performance
///
/// `O(C * A)` where `C = characteristics.len()`, `A = attributes per characteristic`. Predicate dispatch
/// is inline and branch-friendly; no heap allocation on the hot path except for the temporary `reason_codes` vec.
///
/// # Examples
///
/// ```
/// use pmml_core::{FieldId, SymbolId, Value};
/// use pmml_ir::ir::*;
/// use pmml_evaluator::models::evaluate_scorecard;
///
/// let f_dept = FieldId(0);
/// let f_age = FieldId(1);
/// let sc = ScorecardIr {
///     function_name: "regression".into(),
///     initial_score: 0.0,
///     use_reason_codes: false,
///     reason_code_algorithm: "pointsAbove".into(),
///     mining_schema: MiningSchemaIr { active_fields: vec![f_dept, f_age], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     characteristics: vec![
///         CharacteristicIr { name: "deptScore".into(), reason_code: None, baseline_score: 10.0, attributes: vec![
///             AttributeIr { partial_score: 5.0, predicate: PredicateIr::Simple { field: f_dept, operator: SimpleOperator::Equal, value: SymbolIdOrContinuous::Symbol(SymbolId(1)) }, reason_code: None },
///             AttributeIr { partial_score: 2.0, predicate: PredicateIr::True, reason_code: None },
///         ]},
///         CharacteristicIr { name: "ageScore".into(), reason_code: None, baseline_score: 0.0, attributes: vec![
///             AttributeIr { partial_score: 3.0, predicate: PredicateIr::Simple { field: f_age, operator: SimpleOperator::GreaterThan, value: SymbolIdOrContinuous::Continuous(30.0) }, reason_code: None },
///         ]},
///     ],
///     output: vec![],
/// };
/// let vals = vec![Value::Discrete(SymbolId(1)), Value::Continuous(35.0)];
/// assert_eq!(evaluate_scorecard(&sc, &vals), Value::Continuous(8.0)); // 5 + 3
/// let vals2 = vec![Value::Discrete(SymbolId(99)), Value::Continuous(20.0)];
/// assert_eq!(evaluate_scorecard(&sc, &vals2), Value::Continuous(2.0)); // 2 + 0 (baseline)
/// ```
pub fn evaluate_scorecard(scorecard: &ScorecardIr, values: &[Value]) -> Value {
    let mut total = scorecard.initial_score;
    let mut reason_codes: Vec<String> = Vec::new();

    for ch in &scorecard.characteristics {
        let mut matched = false;
        for attr in &ch.attributes {
            if eval_predicate(&attr.predicate, values) {
                total += attr.partial_score;
                if let Some(rc) = &attr.reason_code {
                    reason_codes.push(rc.clone());
                } else if let Some(rc) = &ch.reason_code {
                    reason_codes.push(rc.clone());
                }
                matched = true;
                break;
            }
        }
        if !matched {
            total += ch.baseline_score;
            // baseline reason code not added
        }
    }

    // For now, ignore reasonCodeAlgorithm and useReasonCodes, just return total as continuous
    Value::Continuous(total)
}

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
                        (
                            Value::Continuous(a),
                            pmml_ir::ir::SymbolIdOrContinuous::Continuous(b),
                        ) => match operator {
                            SimpleOperator::Equal => (a - b).abs() < 1e-9,
                            SimpleOperator::NotEqual => (a - b).abs() >= 1e-9,
                            SimpleOperator::LessThan => a < *b,
                            SimpleOperator::LessOrEqual => a <= *b,
                            SimpleOperator::GreaterThan => a > *b,
                            SimpleOperator::GreaterOrEqual => a >= *b,
                            _ => false,
                        },
                        (Value::Discrete(sid), pmml_ir::ir::SymbolIdOrContinuous::Symbol(s)) => {
                            match operator {
                                SimpleOperator::Equal => sid == *s,
                                SimpleOperator::NotEqual => sid != *s,
                                _ => false,
                            }
                        }
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
            let mut found = false;
            for v in array {
                let matches = match (actual, v) {
                    (Value::Discrete(sid), pmml_ir::ir::SymbolIdOrContinuous::Symbol(s)) => {
                        sid == *s
                    }
                    (Value::Continuous(a), pmml_ir::ir::SymbolIdOrContinuous::Continuous(b)) => {
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
            pmml_ir::ir::CompoundOperator::And => {
                predicates.iter().all(|p| eval_predicate(&**p, values))
            }
            pmml_ir::ir::CompoundOperator::Or => {
                predicates.iter().any(|p| eval_predicate(&**p, values))
            }
            pmml_ir::ir::CompoundOperator::Xor => {
                let mut c = 0;
                for p in predicates.iter() {
                    if eval_predicate(&**p, values) {
                        c += 1;
                    }
                }
                c == 1
            }
            pmml_ir::ir::CompoundOperator::Surrogate => {
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
    use pmml_ir::ir::*;

    #[test]
    fn scorecard_simple() {
        let f_dept = FieldId(0);
        let f_age = FieldId(1);
        // Simple scorecard with 1 characteristic, 2 attributes
        let sc = ScorecardIr {
            function_name: "regression".into(),
            initial_score: 0.0,
            use_reason_codes: false,
            reason_code_algorithm: "pointsAbove".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f_dept, f_age],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            characteristics: vec![
                CharacteristicIr {
                    name: "deptScore".into(),
                    reason_code: None,
                    baseline_score: 10.0,
                    attributes: vec![
                        AttributeIr {
                            partial_score: 5.0,
                            predicate: PredicateIr::Simple {
                                field: f_dept,
                                operator: SimpleOperator::Equal,
                                value: SymbolIdOrContinuous::Symbol(SymbolId(1)),
                            },
                            reason_code: None,
                        },
                        AttributeIr {
                            partial_score: 2.0,
                            predicate: PredicateIr::True,
                            reason_code: None,
                        },
                    ],
                },
                CharacteristicIr {
                    name: "ageScore".into(),
                    reason_code: None,
                    baseline_score: 0.0,
                    attributes: vec![AttributeIr {
                        partial_score: 3.0,
                        predicate: PredicateIr::Simple {
                            field: f_age,
                            operator: SimpleOperator::GreaterThan,
                            value: SymbolIdOrContinuous::Continuous(30.0),
                        },
                        reason_code: None,
                    }],
                },
            ],
            output: vec![],
        };
        // dept = marketing (SymbolId 1) => 5, age 35 => 3, total 8
        let vals = vec![Value::Discrete(SymbolId(1)), Value::Continuous(35.0)];
        assert_eq!(evaluate_scorecard(&sc, &vals), Value::Continuous(8.0));
        // dept unknown (SymbolId 99) => matches second attr (True) => 2, age 20 => no match => baseline 0 => total 2
        let vals2 = vec![Value::Discrete(SymbolId(99)), Value::Continuous(20.0)];
        assert_eq!(evaluate_scorecard(&sc, &vals2), Value::Continuous(2.0));
    }
}
