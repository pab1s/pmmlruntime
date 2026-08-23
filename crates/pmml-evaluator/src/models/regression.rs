use pmml_core::Value;
use pmml_ir::ir::{RegressionIr, RegressionNormalizationMethod, SymbolIdOrContinuous};

/// Evaluate regression model given values array.
/// Returns predicted Value (Continuous for regression, Discrete for classification).
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
    let mut best_category: Option<pmml_core::SymbolId> = None;
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

fn apply_normalization(score: f64, method: RegressionNormalizationMethod) -> f64 {
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
    use pmml_core::{FieldId, SymbolId};
    use pmml_ir::ir::*;

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
