//! Targets post-processing — rescaling, clamping, and integer casting.
//!
//! Implements `TargetUtil` semantics from JPMML (`Target`/`Targets`): after a
//! model produces a raw score, the first [`TargetIr`] is applied to that value.
//! Categorical (`Discrete`) predictions pass through unchanged; continuous values
//! are optionally clamped by `min`/`max`, rescaled by `rescaleFactor`/`rescaleConstant`,
//! and then cast to integer.
//!
//! # What belongs here
//!
//! - [`apply_targets`] — authoritative entry point; also handles `defaultValue` when the prediction is `Missing`.
//! - [`apply_targets_with_prior`] — thin wrapper for classification contexts where prior probabilities
//!   would influence missing-value handling (currently delegates to [`apply_targets`]).
//!
//! # Relationship to other modules
//!
//! `pmml-session` calls [`apply_targets`] after `evaluate_*` for models that carry
//! `targets: Vec<TargetIr>` (e.g., `Tree`, `Regression`). `MiningModel` segment chaining
//! writes raw predictions back to `values` before this step, so targets apply only to the final score.
//!
//! # Invariants
//!
//! - When `targets` is empty the value is returned unchanged.
//! - Only the first [`TargetIr`] is applied (single-target case); multi-target is not yet supported.
//! - For `Missing` predictions, the first `TargetValue` with `default_value.is_some()` wins.

use crate::base::Value;
use crate::ir::{CastIntegerMethod, TargetIr};

/// Apply [`TargetIr`] post-processing to a predicted value.
///
/// Single-target semantics: when `targets` is empty the input is returned as-is.
/// Otherwise the first target is used and the following JPMML order is applied:
///
/// 1. **Missing** → search `targets[0].target_values` for the first `default_value.is_some()` and
///    return `Continuous(default)`; if none exists, return `Missing`.
/// 2. **Discrete** → returned unchanged (no rescale / clamp / cast).
/// 3. **Continuous** → `clamp(min, max)` → `value * rescaleFactor + rescaleConstant` → optional
///    integer cast (`Round` / `Ceiling` / `Floor`; legacy `cast_integer == true` maps to `Round`).
///
/// # Parameters
///
/// - `targets`: Slice of [`TargetIr`] from the model's `Targets`. Only `targets[0]` is consulted.
/// - `value`: Raw predicted [`Value`] from `evaluate_*` (typically `Continuous` for regression).
///
/// # Returns
///
/// Post-processed [`Value`]. `Missing` when the input is `Missing` and no `defaultValue` exists;
/// otherwise `Continuous` with rescaling applied, or the original `Discrete`.
///
/// # Panics
///
/// Never panics. All option handling is checked.
///
/// # Performance
///
/// `O(target_values)` to scan for a default when `value` is `Missing`; `O(1)` otherwise.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, Value};
/// use pmmlruntime::ir::{TargetIr, TargetValueIr, CastIntegerMethod};
/// use pmmlruntime::engine::targets::apply_targets;
///
/// let target = TargetIr {
///     field: Some(FieldId(0)),
///     field_name: "y".into(),
///     op_type: None,
///     rescale_constant: 1.0,
///     rescale_factor: 2.0,
///     cast_integer: true,
///     cast_method: Some(CastIntegerMethod::Round),
///     min: None,
///     max: None,
///     target_values: vec![],
/// };
/// // 2.3 * 2 + 1 = 5.6 → round 6
/// assert_eq!(apply_targets(&[target], Value::Continuous(2.3)), Value::Continuous(6.0));
/// ```
pub fn apply_targets(targets: &[TargetIr], value: Value) -> Value {
    if targets.is_empty() {
        return value;
    }

    // Missing case: check for defaultValue
    if value.is_missing() {
        for t in targets {
            for tv in &t.target_values {
                if let Some(default) = tv.default_value {
                    // TargetValue defaultValue is numeric (per XSD NUMBER)
                    // For continuous, return Continuous(default)
                    // For categorical, the default is still numeric? But we treat as Continuous for now.
                    // If target field is categorical, the default should be discrete, but XSD says defaultValue is NUMBER, so it's numeric.
                    // Return as Continuous; caller will handle type.
                    return Value::Continuous(default);
                }
            }
        }
        return Value::Missing;
    }

    // For non-missing, apply first target's transforms (or matching target if field matches)
    // In JPMML, each Target corresponds to a target field; for single target, apply that one.
    // For simplicity, use first target.
    let t = &targets[0];

    // Only apply rescale/min/max/cast for Continuous values.
    // For Discrete (classification), targets are used for prior probabilities etc., not rescale.
    match value {
        Value::Continuous(f) => {
            let mut cf = f;

            // Apply min/max clipping (TargetUtil.processValue restricts)
            if let Some(min) = t.min {
                if cf < min {
                    cf = min;
                }
            }
            if let Some(max) = t.max {
                if cf > max {
                    cf = max;
                }
            }

            // Rescale: v = v * rescaleFactor + rescaleConstant
            // Note: order is multiply then add per TargetUtil: value.multiply(rescaleFactor).add(rescaleConstant)
            cf = cf * t.rescale_factor + t.rescale_constant;

            // Cast
            if let Some(method) = t.cast_method {
                cf = match method {
                    CastIntegerMethod::Round => cf.round(),
                    CastIntegerMethod::Ceiling => cf.ceil(),
                    CastIntegerMethod::Floor => cf.floor(),
                };
            } else if t.cast_integer {
                // Old bool case: treat as round
                cf = cf.round();
            }

            Value::Continuous(cf)
        }
        Value::Discrete(_) => {
            // For discrete, no rescale/min/max/cast; return as is
            // JPMML would handle displayValue etc. via Output, not here
            value
        }
        Value::Missing => value, // already handled missing case above, but keep
    }
}

/// Apply [`TargetIr`] with a classification prior-probability hint.
///
/// For classification models where the prediction is [`Value::Missing`] and no
/// `defaultValue` exists, JPMML would fall back to prior probabilities from
/// `TargetValue/@priorProbability`. This function currently delegates to
/// [`apply_targets`] unchanged — prior handling is performed by the `Output`
/// layer rather than here — but the parameter is retained for API compatibility
/// with JPMML's `TargetUtil`.
///
/// # Parameters
///
/// - `targets`: Same as [`apply_targets`].
/// - `value`: Raw predicted value.
/// - `_is_classification`: When `true` the caller is a classification model; currently ignored.
///
/// # Returns
///
/// Same as [`apply_targets`]. No additional probability logic is applied yet.
///
/// # Panics
///
/// Never panics.
///
/// # Performance
///
/// Same as [`apply_targets`]: `O(target_values)` for missing, `O(1)` otherwise.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{Value, FieldId};
/// use pmmlruntime::ir::TargetIr;
/// use pmmlruntime::engine::targets::apply_targets_with_prior;
///
/// let targets = vec![TargetIr {
///     field: Some(FieldId(0)), field_name: "y".into(), op_type: None,
///     rescale_constant: 0.0, rescale_factor: 1.0, cast_integer: false, cast_method: None,
///     min: Some(0.0), max: Some(10.0), target_values: vec![],
/// }];
/// assert_eq!(apply_targets_with_prior(&targets, Value::Continuous(15.0), false), Value::Continuous(10.0));
/// assert_eq!(apply_targets_with_prior(&targets, Value::Missing, true), Value::Missing);
/// ```
pub fn apply_targets_with_prior(
    targets: &[TargetIr],
    value: Value,
    _is_classification: bool,
) -> Value {
    apply_targets(targets, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, Value};
    use crate::ir::{CastIntegerMethod, TargetIr, TargetValueIr};

    fn make_target(
        rescale_factor: f64,
        rescale_constant: f64,
        cast: Option<CastIntegerMethod>,
        min: Option<f64>,
        max: Option<f64>,
    ) -> TargetIr {
        TargetIr {
            field: Some(FieldId(0)),
            field_name: "y".to_string(),
            op_type: None,
            rescale_constant,
            rescale_factor,
            cast_integer: cast.is_some(),
            cast_method: cast,
            min,
            max,
            target_values: vec![],
        }
    }

    #[test]
    fn rescale_and_cast() {
        let t = make_target(2.0, 1.0, Some(CastIntegerMethod::Round), None, None);
        let v = Value::Continuous(2.3);
        // 2.3*2+1=5.6 round=6
        assert_eq!(apply_targets(&[t], v), Value::Continuous(6.0));
    }

    #[test]
    fn min_max_clipping() {
        let t = make_target(1.0, 0.0, None, Some(0.0), Some(10.0));
        assert_eq!(
            apply_targets(&[t.clone()], Value::Continuous(-5.0)),
            Value::Continuous(0.0)
        );
        assert_eq!(
            apply_targets(&[t], Value::Continuous(15.0)),
            Value::Continuous(10.0)
        );
    }

    #[test]
    fn missing_default() {
        let mut t = make_target(1.0, 0.0, None, None, None);
        t.target_values.push(TargetValueIr {
            value: None,
            value_str: None,
            display_value: None,
            prior_probability: None,
            default_value: Some(42.0),
        });
        assert_eq!(apply_targets(&[t], Value::Missing), Value::Continuous(42.0));
    }

    #[test]
    fn categorical_no_rescale() {
        use crate::base::SymbolId;
        let t = make_target(2.0, 1.0, Some(CastIntegerMethod::Round), None, None);
        let v = Value::Discrete(SymbolId(1));
        assert_eq!(apply_targets(&[t], v), v);
    }
}
