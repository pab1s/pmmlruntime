//! RegressionModel evaluation — intercept plus predictors with normalization.
//!
//! Implements `RegressionModel` scoring: `y = intercept + Σ coeff * field^exponent`
//! per [`RegressionTableIr`](crate::ir::RegressionTableIr), summed over numeric and
//! categorical predictors, then transformed by [`RegressionNormalizationMethod`].
//! Single-table regression returns `Continuous`; multi-table classification (multiple
//! `targetCategory` tables) returns the category whose normalized score is maximal.
//!
//! # What belongs here
//!
//! - [`evaluate_regression`] — public scoring entry point.
//! - `apply_normalization` — internal normalization helper (`Softmax`/`Logit`/`Probit`, `pub(crate)`).
//!
//! # Invariants
//!
//! - Missing numeric fields contribute `0` (skipped, matching JPMML).
//! - Categorical predictors contribute only when `Discrete(field) == value`.
//! - `Softmax` for multi-table is simplified (single-table `sigmoid` path).

use crate::base::Value;
use crate::ir::{RegressionIr, RegressionNormalizationMethod};

/// Evaluate a [`RegressionIr`] against a dense `values` array.
///
/// Computes per-table scores as `intercept + Σ numeric_predictor.coefficient * values[field].powi(exponent)`
/// plus `Σ categorical_predictor.coefficient` when the discrete value matches. For a single table
/// (regression) the sum is normalized by `normalization_method` and returned as `Continuous`.
///
/// For multiple tables (classification) each table's score is normalized in the same way and the
/// `target_category` of the table with the greatest normalized score is returned as `Discrete`.
/// When `regression_tables` is empty, `Missing` is returned.
///
/// # Parameters
///
/// - `reg`: Lowered regression model (`RegressionIr`) with `regression_tables`, `numeric_predictors`,
///   `categorical_predictors`, and `normalization_method`.
/// - `values`: Dense `&[Value]` indexed by [`FieldId`](crate::base::FieldId). Out-of-bounds or `Missing` numeric fields are skipped.
///
/// # Returns
///
/// `Continuous(normalized_score)` for single-table regression, `Discrete(target_category)` for
/// multi-table classification (max score), or `Missing` when there are no tables or no `target_category` wins.
///
/// # Panics
///
/// Never panics. All `FieldId` indexing is bounds-checked.
///
/// # Performance
///
/// `O(tables * (numeric_predictors + categorical_predictors))`. No allocation; `powi` is used for integer exponents.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, Value, SymbolId};
/// use pmmlruntime::ir::*;
/// use pmmlruntime::engine::models::evaluate_regression;
///
/// let f = FieldId(0);
/// let reg = RegressionIr {
///     function_name: "regression".into(),
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     regression_tables: vec![RegressionTableIr {
///         intercept: 0.0, target_category: None,
///         numeric_predictors: vec![NumericPredictorIr { field: f, coefficient: 2.0, exponent: 1 }],
///         categorical_predictors: vec![],
///     }],
///     normalization_method: RegressionNormalizationMethod::None, targets: vec![], output: vec![],
/// };
/// assert_eq!(evaluate_regression(&reg, &[Value::Continuous(2.5)]), Value::Continuous(5.0));
///
/// // Missing numeric contributes 0 → intercept only
/// assert_eq!(evaluate_regression(&reg, &[Value::Missing]), Value::Continuous(0.0));
/// ```
pub fn evaluate_regression(reg: &RegressionIr, values: &[Value]) -> Value {
    if reg.regression_tables.is_empty() {
        return Value::Missing;
    }
    // For single table (regression): compute intercept + sum(coeff * field^exp)
    if reg.regression_tables.len() == 1 {
        let tbl = &reg.regression_tables[0];
        let mut sum = tbl.intercept;
        for np in &tbl.numeric_predictors {
            let idx = np.field.as_usize();
            let actual = if idx < values.len() {
                values[idx]
            } else {
                Value::Missing
            };
            if let Value::Continuous(v) = actual {
                let powered = v.powi(np.exponent);
                sum += np.coefficient * powered;
            } else if actual.is_missing() {
                // missing -> skip (treated as 0?) In JPMML, missing numeric -> 0? For v1, skip
                continue;
            }
        }
        for cp in &tbl.categorical_predictors {
            let idx = cp.field.as_usize();
            let actual = if idx < values.len() {
                values[idx]
            } else {
                Value::Missing
            };
            let matches = match actual {
                Value::Discrete(sid) => sid == cp.value,
                _ => false,
            };
            if matches {
                sum += cp.coefficient;
            }
        }
        let normalized = apply_normalization(sum, reg.normalization_method);
        return Value::Continuous(normalized);
    }

    // Multiple tables (classification) — compute score per targetCategory, then normalize to probabilities
    // For v1, we just return the category with max score
    let mut best_score = f64::NEG_INFINITY;
    let mut best_category: Option<crate::base::SymbolId> = None;
    for tbl in &reg.regression_tables {
        let mut sum = tbl.intercept;
        for np in &tbl.numeric_predictors {
            let idx = np.field.as_usize();
            let actual = if idx < values.len() {
                values[idx]
            } else {
                Value::Missing
            };
            if let Value::Continuous(v) = actual {
                sum += np.coefficient * v.powi(np.exponent);
            }
        }
        for cp in &tbl.categorical_predictors {
            let idx = cp.field.as_usize();
            let actual = if idx < values.len() {
                values[idx]
            } else {
                Value::Missing
            };
            if let Value::Discrete(sid) = actual {
                if sid == cp.value {
                    sum += cp.coefficient;
                }
            }
        }
        let norm = apply_normalization(sum, reg.normalization_method);
        if norm > best_score {
            best_score = norm;
            best_category = tbl.target_category;
        }
    }
    if let Some(cat) = best_category {
        Value::Discrete(cat)
    } else {
        Value::Missing
    }
}

/// Apply a [`RegressionNormalizationMethod`] to a raw regression score.
///
/// Normalization is applied after the intercept plus predictor sum:
///
/// - `None` → identity
/// - `Logit` / `Softmax` (single-table) → `1 / (1 + exp(-score))` (sigmoid)
/// - `Exp` → `exp(score)`
/// - `Probit` → `0.5 * (1 + erf(score / √2))`
/// - `SimpleMax`, `ClogLog`, `Loglog`, `Cauchit` → currently identity (stub) except where mapped above.
///
/// This is `pub(crate)` so `simd` can reuse the same normalization for its lane results.
///
/// # Parameters
///
/// - `score`: Raw regression sum.
/// - `method`: Normalization method from `RegressionModel/@normalizationMethod`.
///
/// # Returns
///
/// Normalized `f64`. `NaN`/`±INFINITY` inputs follow IEEE 754 (`exp`, `erf`).
///
/// # Panics
///
/// Never panics.
///
/// # Performance
///
/// `O(1)`.
pub(crate) fn apply_normalization(score: f64, method: RegressionNormalizationMethod) -> f64 {
    match method {
        RegressionNormalizationMethod::None => score,
        RegressionNormalizationMethod::Softmax => 1.0 / (1.0 + (-score).exp()), // simplified sigmoid for single
        RegressionNormalizationMethod::Logit => 1.0 / (1.0 + (-score).exp()),
        RegressionNormalizationMethod::Exp => score.exp(),
        RegressionNormalizationMethod::SimpleMax => score, // stub
        RegressionNormalizationMethod::Probit => {
            0.5 * (1.0 + libm::erf(score / std::f64::consts::SQRT_2))
        }
        _ => score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, SymbolId};
    use crate::ir::*;

    #[test]
    fn single_table_regression() {
        let f_input = FieldId(0);
        let reg = RegressionIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f_input],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            regression_tables: vec![RegressionTableIr {
                intercept: 0.0,
                target_category: None,
                numeric_predictors: vec![NumericPredictorIr {
                    field: f_input,
                    coefficient: 2.0,
                    exponent: 1,
                }],
                categorical_predictors: vec![],
            }],
            normalization_method: RegressionNormalizationMethod::None,
            targets: vec![],
            output: vec![],
        };
        let vals = vec![Value::Continuous(2.5)];
        assert_eq!(evaluate_regression(&reg, &vals), Value::Continuous(5.0));
    }
}
