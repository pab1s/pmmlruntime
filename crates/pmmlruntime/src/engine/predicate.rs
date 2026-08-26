//! Shared predicate evaluation — branch-friendly [`PredicateIr`] interpreter.
//!
//! Evaluates the four PMML predicate forms used by `TreeModel` nodes,
//! `RuleSet` rules, and `MiningModel` segment selection:
//! `True`, `Simple` (`field operator value`), `SimpleSet` (`isIn` / `isNotIn`),
//! and `Compound` (`and` / `or` / `xor` / `surrogate`). The implementation is
//! inlined and intentionally branch-predictor friendly to sustain the 402 ns
//! single-row tree benchmark (Iris).
//!
//! # What belongs here
//!
//! - [`eval_predicate`] — the single public pure function.
//! - Private `eval_simple` for the `Simple` operator dispatch.
//!
//! # Relationship to other modules
//!
//! `pmml-evaluator::models::tree` traverses `Vec<NodeIr>` and tests each child's
//! [`PredicateIr`] via [`eval_predicate`]; `models::mining` tests segment
//! predicates; `models::scorecard` and `models::rule_set` contain duplicated
//! copies for bootstrapping and are migrating to this shared helper.
//!
//! # Invariants
//!
//! - `values` is indexed by [`FieldId::as_usize`]; out-of-bounds access yields
//!   [`Value::Missing`] (never panics).
//! - `Missing` never satisfies equality / inequality / comparison predicates; only
//!   `isMissing` / `isNotMissing` observe it.
//! - Continuous equality uses an epsilon `1e-9`.

use crate::base::{FieldId, Value};
use crate::ir::{CompoundOperator, PredicateIr, SimpleOperator, SymbolIdOrContinuous};

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

/// Evaluate a [`PredicateIr`] against a dense `values` array.
///
/// Implements the PMML predicate semantics used by tree traversal, rule firing,
/// and segment selection. All indexing is bounds-checked.
///
/// # Parameters
///
/// - `pred`: Predicate to test. `True` always yields `true`; `Simple` / `SimpleSet` / `Compound`
///   are dispatched as described in the module docs.
/// - `values`: Dense `&[Value]` indexed by [`FieldId::as_usize`]. Out-of-bounds fields are treated as
///   [`Value::Missing`].
///
/// # Returns
///
/// `true` when the predicate holds for the given row, `false` otherwise.
/// `Missing` values cause equality/inequality/comparison predicates to return `false` (only
/// `isMissing`/`isNotMissing` return `true` for missing). `SimpleSet` with `is_in = false`
/// negates membership (`isNotIn`).
///
/// `Compound` operators:
/// - `And` → all true
/// - `Or` → any true
/// - `Xor` → exactly one true
/// - `Surrogate` → iterate in order, skip children whose field is missing, then evaluate the first
///   non-skipped predicate as `true`/`false` (missing children are ignored per PMML).
///
/// # Panics
///
/// Never panics. All `FieldId` indexing is bounds-checked.
///
/// # Performance
///
/// Branchless-friendly and `#[inline(always)]` for `Simple`; `Compound` recurses over a
/// `SmallVec<[Box<PredicateIr>; 4]>`, so typical arities 1–4 stay inline without allocation.
/// Complexity is `O(predicates)` for compound nodes.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, SymbolId, Value};
/// use pmmlruntime::ir::{PredicateIr, SimpleOperator, SymbolIdOrContinuous};
/// use pmmlruntime::engine::predicate::eval_predicate;
///
/// // Predicate: field 0 < 2.45
/// let pred = PredicateIr::Simple {
///     field: FieldId(0),
///     operator: SimpleOperator::LessThan,
///     value: SymbolIdOrContinuous::Continuous(2.45),
/// };
/// let values = vec![Value::Continuous(1.0)];
/// assert!(eval_predicate(&pred, &values));
/// let values2 = vec![Value::Continuous(3.0)];
/// assert!(!eval_predicate(&pred, &values2));
///
/// // isMissing predicate
/// let missing_pred = PredicateIr::Simple {
///     field: FieldId(0),
///     operator: SimpleOperator::IsMissing,
///     value: SymbolIdOrContinuous::Missing,
/// };
/// assert!(eval_predicate(&missing_pred, &[Value::Missing]));
/// assert!(!eval_predicate(&missing_pred, &[Value::Continuous(1.0)]));
/// ```
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
    use crate::base::{FieldId, SymbolId, Value};
    use crate::ir::PredicateIr;

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
