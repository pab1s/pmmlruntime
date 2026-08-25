//! Targets post-processing — rescale, cast, min/max, defaultValue per JPMML spec.
//! Mirrors `org.jpmml.evaluator.TargetUtil` (TargetUtil.java).

use pmml_core::Value;
use pmml_ir::ir::{CastIntegerMethod, TargetIr};

/// Apply Targets to a predicted value.
/// - If predicted is Missing, returns defaultValue if present (first TargetValue with Some default), else Missing.
/// - Else applies min/max clipping, rescaleFactor/rescaleConstant, and castInteger per first Target.
/// For MultiTarget, only first is applied (single target case). For categorical, no rescale.
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

/// Apply Targets and also handle priorProbability for classification missing case.
/// For classification, if predicted is Missing and no defaultValue, we could return prior probabilities distribution,
/// but this function only returns single Value, so we return Missing and let Output handle probabilities.
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
    use pmml_core::{FieldId, Value};
    use pmml_ir::ir::{CastIntegerMethod, TargetIr, TargetValueIr};

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
        use pmml_core::SymbolId;
        let t = make_target(2.0, 1.0, Some(CastIntegerMethod::Round), None, None);
        let v = Value::Discrete(SymbolId(1));
        assert_eq!(apply_targets(&[t], v), v);
    }
}
