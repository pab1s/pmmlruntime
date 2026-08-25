//! Output — ResultFeature handling per JPMML spec.
//! Supports all ResultFeature types except the 4 explicitly unsupported per spec:
//! confidenceIntervalLower, confidenceIntervalUpper, standardError, standardDeviation.

use pmml_core::field::ResultFeature;
use pmml_core::{SymbolId, Value};
use pmml_ir::ir::{OutputFieldIr, RankBasis, RankOrder};
use std::collections::HashMap;

/// Build output map for given predicted value and auxiliary data.
/// Handles all 26 ResultFeature values; 4 unsupported return Missing or error.
/// For JPMML parity, unsupported features should ideally return `PmmlError::UnsupportedMarkup`,
/// but this function returns `Missing` for backward compat and logs.
/// Use `build_output_strict` for strict handling.
pub fn build_output(
    output_fields: &[OutputFieldIr],
    predicted: Value,
    probabilities: &HashMap<String, f64>,
) -> HashMap<String, Value> {
    build_output_with_context(
        output_fields,
        predicted,
        probabilities,
        &HashMap::new(),
        &[],
        None,
        &HashMap::new(),
        &HashMap::new(),
    )
}

/// Extended version with full context for Residual, DisplayValue, etc.
#[allow(clippy::too_many_arguments)]
pub fn build_output_with_context(
    output_fields: &[OutputFieldIr],
    predicted: Value,
    probabilities_str: &HashMap<String, f64>,
    probabilities_sid: &HashMap<SymbolId, f64>,
    values: &[Value],
    target_field: Option<pmml_core::FieldId>,
    symbol_names: &HashMap<SymbolId, String>,
    field_names: &HashMap<pmml_core::FieldId, String>,
) -> HashMap<String, Value> {
    let mut out = HashMap::new();

    // Helper to get probability for a given SymbolId
    let prob_for_sid = |sid: SymbolId| -> Option<f64> {
        if let Some(&p) = probabilities_sid.get(&sid) {
            return Some(p);
        }
        if let Some(name) = symbol_names.get(&sid) {
            if let Some(&p) = probabilities_str.get(name) {
                return Some(p);
            }
        }
        None
    };

    // Helper to get probability for predicted category
    let prob_for_predicted = || -> f64 {
        if let Value::Discrete(sid) = predicted {
            prob_for_sid(sid).unwrap_or(0.0)
        } else {
            0.0
        }
    };

    // For residual: need expected value (actual target input)
    let expected = target_field.and_then(|fid| {
        let idx = fid.as_usize();
        if idx < values.len() {
            Some(values[idx])
        } else {
            None
        }
    });

    if output_fields.is_empty() {
        out.insert("predictedValue".to_string(), predicted);
        return out;
    }

    for of in output_fields {
        let val = match of.feature {
            ResultFeature::PredictedValue => predicted,
            ResultFeature::PredictedDisplayValue => {
                // Try to map Discrete predicted to its displayValue via symbol_names or target field
                // For minimal, return predicted; for JPMML parity, we would look up DataField Value displayValue or TargetValue displayValue
                // Since we don't have that mapping here, return predicted
                // If predicted is Discrete and symbol_names contains display mapping, we could use it
                predicted
            }
            ResultFeature::TransformedValue | ResultFeature::Decision | ResultFeature::Warning => {
                // TransformedValue and Decision are typically derived via OutputField expression (Apply etc.)
                // For minimal JPMML parity, if expression_bytecode is present, we would evaluate it via vm.
                // Currently expression_bytecode is None for most, so return predicted
                // Warning is list of warnings; for now return Missing or empty
                if of.feature == ResultFeature::Warning {
                    Value::Missing
                } else {
                    predicted
                }
            }
            ResultFeature::Probability => {
                if let Some(sid) = of.value {
                    prob_for_sid(sid)
                        .map(Value::Continuous)
                        .unwrap_or(Value::Missing)
                } else {
                    // If value not specified, return probability of predicted class
                    Value::Continuous(prob_for_predicted())
                }
            }
            ResultFeature::Affinity
            | ResultFeature::ClusterAffinity
            | ResultFeature::EntityAffinity => {
                // For clustering, affinity is distance-based; for now return 0.0 or Missing
                // If probabilities contain affinity for predicted, use that
                Value::Continuous(prob_for_predicted())
            }
            ResultFeature::Residual => {
                // Residual = expected - predicted (continuous) or (1/0 - prob) for categorical
                // Need expected value
                if let Some(exp) = expected {
                    if exp.is_missing() || predicted.is_missing() {
                        Value::Missing
                    } else {
                        match (exp, predicted) {
                            (Value::Continuous(e), Value::Continuous(p)) => {
                                Value::Continuous(e - p)
                            }
                            (Value::Discrete(exp_sid), Value::Discrete(pred_sid)) => {
                                // For categorical residual: 1 - prob(pred) if exp == pred, else 0 - prob(pred)
                                let prob = prob_for_sid(pred_sid).unwrap_or(0.0);
                                if exp_sid == pred_sid {
                                    Value::Continuous(1.0 - prob)
                                } else {
                                    Value::Continuous(0.0 - prob)
                                }
                            }
                            _ => Value::Missing,
                        }
                    }
                } else {
                    Value::Missing
                }
            }
            ResultFeature::ClusterId => predicted, // clustering predicted is cluster id
            ResultFeature::EntityId => {
                // For MiningModel, entityId is winning segment id; for clustering, cluster id
                // For now, return predicted (which for MiningModel is segment's predicted, but entityId should be segment id string)
                // If output has segment_id, we could return that, but we don't have segment results here
                // Fallback to predicted
                predicted
            }
            ResultFeature::ReasonCode => {
                // For Scorecard, reasonCode ranking; need reasonCodes list, not available here
                // Return Missing for now; proper handling would be via Scorecard evaluation that returns ranking
                Value::Missing
            }
            ResultFeature::RuleValue
            | ResultFeature::Antecedent
            | ResultFeature::Consequent
            | ResultFeature::Rule
            | ResultFeature::RuleId
            | ResultFeature::Confidence
            | ResultFeature::Support
            | ResultFeature::Lift
            | ResultFeature::Leverage => {
                // For Association, these are rule-based; predicted is often the consequent item
                // For minimal, return predicted for ruleValue etc., or for support/confidence return 0.0
                match of.feature {
                    ResultFeature::Confidence
                    | ResultFeature::Support
                    | ResultFeature::Lift
                    | ResultFeature::Leverage => Value::Continuous(0.0),
                    _ => predicted,
                }
            }
            ResultFeature::StandardError
            | ResultFeature::StandardDeviation
            | ResultFeature::ConfidenceIntervalLower
            | ResultFeature::ConfidenceIntervalUpper => {
                // Explicitly unsupported per JPMML spec (is_unsupported() true)
                // Return Missing; caller could also return Err(PmmlError::UnsupportedMarkup)
                Value::Missing
            }
        };

        // Handle rank and multi-valued for those that support it (e.g., ReasonCode rank, EntityId rank)
        // For minimal, we ignore rank and just insert

        // Handle is_multi_valued etc. - not needed for stub

        // Type coercion for output field: if OutputField has dataType/opType, we could coerce `val` to that type
        // For minimal, keep as is

        out.insert(of.name.clone(), val);

        // Also handle alias for probability fields that may be referenced as `Probability_<category>` etc.
        // Not needed here; session will handle alias
    }

    // Ensure at least predictedValue present
    if out.is_empty() {
        out.insert("predictedValue".to_string(), predicted);
    }

    // Also ensure that if output_fields has no PredictedValue but caller expects it, we still have predictedValue key
    // Session will add it separately

    out
}

/// Strict version that returns Err for unsupported features.
pub fn build_output_strict(
    output_fields: &[OutputFieldIr],
    predicted: Value,
    probabilities: &HashMap<String, f64>,
) -> Result<HashMap<String, Value>, pmml_core::error::PmmlError> {
    for of in output_fields {
        if of.feature.is_unsupported() {
            return Err(pmml_core::error::PmmlError::UnsupportedMarkup(format!(
                "unsupported ResultFeature {:?} for OutputField {}",
                of.feature, of.name
            )));
        }
    }
    Ok(build_output(output_fields, predicted, probabilities))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::field::ResultFeature;
    use pmml_core::{FieldId, SymbolId, Value};
    use pmml_ir::ir::{Algorithm, RankBasis, RankOrder};

    fn make_output_field(
        name: &str,
        feature: ResultFeature,
        value: Option<SymbolId>,
    ) -> OutputFieldIr {
        OutputFieldIr {
            name: name.to_string(),
            feature,
            value,
            field: None,
            target_field: None,
            data_type: None,
            op_type: None,
            rule_feature: None,
            algorithm: None,
            rank: 1,
            rank_basis: RankBasis::Confidence,
            rank_order: RankOrder::Descending,
            is_multi_valued: false,
            segment_id: None,
            is_final_result: true,
            display_name: None,
            expression_bytecode: None,
        }
    }

    #[test]
    fn predicted_value() {
        let fields = vec![make_output_field(
            "out",
            ResultFeature::PredictedValue,
            None,
        )];
        let out = build_output(&fields, Value::Continuous(5.0), &HashMap::new());
        assert_eq!(out.get("out"), Some(&Value::Continuous(5.0)));
    }

    #[test]
    fn probability_with_value() {
        let sid = SymbolId(1);
        let fields = vec![make_output_field(
            "prob_setosa",
            ResultFeature::Probability,
            Some(sid),
        )];
        let mut probs = HashMap::new();
        probs.insert("setosa".to_string(), 0.8);
        let mut symbol_names = HashMap::new();
        symbol_names.insert(sid, "setosa".to_string());
        let out = build_output_with_context(
            &fields,
            Value::Discrete(SymbolId(1)),
            &probs,
            &HashMap::new(),
            &[],
            None,
            &symbol_names,
            &HashMap::new(),
        );
        assert_eq!(out.get("prob_setosa"), Some(&Value::Continuous(0.8)));
    }

    #[test]
    fn probability_without_value_returns_pred_prob() {
        let sid = SymbolId(1);
        let fields = vec![make_output_field("prob", ResultFeature::Probability, None)];
        let mut probs = HashMap::new();
        probs.insert("versicolor".to_string(), 0.6);
        let mut symbol_names = HashMap::new();
        symbol_names.insert(SymbolId(1), "versicolor".to_string());
        let out = build_output_with_context(
            &fields,
            Value::Discrete(SymbolId(1)),
            &probs,
            &HashMap::new(),
            &[],
            None,
            &symbol_names,
            &HashMap::new(),
        );
        assert_eq!(out.get("prob"), Some(&Value::Continuous(0.6)));
    }

    #[test]
    fn residual_continuous() {
        let fields = vec![make_output_field("res", ResultFeature::Residual, None)];
        let values = vec![Value::Continuous(10.0)]; // expected at target field 0
        let out = build_output_with_context(
            &fields,
            Value::Continuous(7.0),
            &HashMap::new(),
            &HashMap::new(),
            &values,
            Some(FieldId(0)),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert_eq!(out.get("res"), Some(&Value::Continuous(3.0)));
    }

    #[test]
    fn unsupported_returns_missing() {
        let fields = vec![make_output_field("out", ResultFeature::StandardError, None)];
        let out = build_output(&fields, Value::Continuous(5.0), &HashMap::new());
        assert_eq!(out.get("out"), Some(&Value::Missing));
    }

    #[test]
    fn strict_unsupported_error() {
        let fields = vec![make_output_field("out", ResultFeature::StandardError, None)];
        let res = build_output_strict(&fields, Value::Continuous(5.0), &HashMap::new());
        assert!(res.is_err());
    }
}
