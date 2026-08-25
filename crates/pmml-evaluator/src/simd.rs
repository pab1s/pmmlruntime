//! SIMD batch evaluation — AVX2 (x86_64) / NEON (aarch64) via `wide` crate.
//! P8: Data-parallel scoring for regression (and tree predicates) — blocked columnar.
//! Scalar fallback for wasm and when `simd` feature disabled.

use pmml_core::Value;
use pmml_ir::ir::RegressionIr;

#[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
use wide::f64x4;

/// Evaluate a batch of rows for a single-table regression using f64x4 SIMD.
/// `batch_values` is slice of row slices, each `&[Value]` length `needed`.
/// Only the first regression table is SIMD-accelerated; categorical predictors fall back to scalar per lane.
/// Returns Vec<Value> length batch.len().
#[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
pub fn evaluate_regression_batch_simd(
    reg: &RegressionIr,
    batch_values: &[&[Value]],
) -> Vec<Value> {
    if reg.regression_tables.len() != 1 {
        // Multi-table classification: fall back to scalar per row (softmax etc. not vectorized yet)
        return batch_values
            .iter()
            .map(|vals| crate::models::regression::evaluate_regression(reg, vals))
            .collect();
    }
    let tbl = &reg.regression_tables[0];
    // Fast path: only numeric predictors, no categorical — pure SIMD
    let has_categorical = !tbl.categorical_predictors.is_empty();
    let mut out = Vec::with_capacity(batch_values.len());
    let mut i = 0;
    let n = batch_values.len();
    // Process 4-wide
    while i + 4 <= n {
        let mut sums = f64x4::splat(tbl.intercept);
        for np in &tbl.numeric_predictors {
            let fid = np.field.as_usize();
            // Load 4 rows' field values; missing -> 0.0 (contributes 0)
            let v0 = match batch_values[i][fid] {
                Value::Continuous(f) => f,
                _ => 0.0,
            };
            let v1 = match batch_values[i + 1][fid] {
                Value::Continuous(f) => f,
                _ => 0.0,
            };
            let v2 = match batch_values[i + 2][fid] {
                Value::Continuous(f) => f,
                _ => 0.0,
            };
            let v3 = match batch_values[i + 3][fid] {
                Value::Continuous(f) => f,
                _ => 0.0,
            };
            let mut vals = f64x4::new([v0, v1, v2, v3]);
            // Handle exponent
            if np.exponent != 1 {
                if np.exponent == 2 {
                    vals = vals * vals;
                } else if np.exponent == 0 {
                    vals = f64x4::splat(1.0);
                } else {
                    // General powi per lane (scalar fallback inside SIMD)
                    let arr = vals.to_array();
                    let powered = [
                        arr[0].powi(np.exponent),
                        arr[1].powi(np.exponent),
                        arr[2].powi(np.exponent),
                        arr[3].powi(np.exponent),
                    ];
                    vals = f64x4::new(powered);
                }
            }
            let coeff = f64x4::splat(np.coefficient);
            sums += coeff * vals;
        }
        // For categorical, scalar fixup per lane (rare for simd batch)
        if has_categorical {
            // Fall back to scalar per row for categorical part, add to sums
            let sums_arr = sums.to_array();
            for lane in 0..4 {
                let mut lane_sum = sums_arr[lane];
                let vals = batch_values[i + lane];
                for cp in &tbl.categorical_predictors {
                    let idx = cp.field.as_usize();
                    if let Value::Discrete(sid) = vals[idx] {
                        if sid == cp.value {
                            lane_sum += cp.coefficient;
                        }
                    }
                }
                let final_val =
                    crate::models::regression::apply_normalization(lane_sum, reg.normalization_method);
                out.push(Value::Continuous(final_val));
            }
        } else {
            let sums_arr = sums.to_array();
            for lane in 0..4 {
                let lane_sum = sums_arr[lane];
                let final_val =
                    crate::models::regression::apply_normalization(lane_sum, reg.normalization_method);
                out.push(Value::Continuous(final_val));
            }
        }
        i += 4;
    }
    // Remainder scalar
    while i < n {
        out.push(crate::models::regression::evaluate_regression(
            reg,
            batch_values[i],
        ));
        i += 1;
    }
    out
}

#[cfg(any(not(feature = "simd"), target_arch = "wasm32"))]
pub fn evaluate_regression_batch_simd(
    reg: &RegressionIr,
    batch_values: &[&[Value]],
) -> Vec<Value> {
    // Fallback scalar
    batch_values
        .iter()
        .map(|vals| crate::models::regression::evaluate_regression(reg, vals))
        .collect()
}

/// Scalar helper exposed for testing — same as `evaluate_regression` but via batch API.
pub fn evaluate_regression_batch_scalar(
    reg: &RegressionIr,
    batch_values: &[&[Value]],
) -> Vec<Value> {
    batch_values
        .iter()
        .map(|vals| crate::models::regression::evaluate_regression(reg, vals))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::{FieldId, Value};
    use pmml_ir::ir::*;

    fn make_reg() -> RegressionIr {
        let f0 = FieldId(0);
        RegressionIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f0],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            regression_tables: vec![RegressionTableIr {
                intercept: 1.0,
                target_category: None,
                numeric_predictors: vec![NumericPredictorIr {
                    field: f0,
                    coefficient: 2.0,
                    exponent: 1,
                }],
                categorical_predictors: vec![],
            }],
            normalization_method: RegressionNormalizationMethod::None,
            targets: vec![],
            output: vec![],
        }
    }

    #[test]
    fn simd_regression_batch() {
        let reg = make_reg();
        let rows: Vec<Vec<Value>> = (0..8)
            .map(|i| vec![Value::Continuous(i as f64)])
            .collect();
        let refs: Vec<&[Value]> = rows.iter().map(|r| r.as_slice()).collect();
        let simd_out = evaluate_regression_batch_simd(&reg, &refs);
        let scalar_out = evaluate_regression_batch_scalar(&reg, &refs);
        assert_eq!(simd_out.len(), 8);
        for (a, b) in simd_out.iter().zip(scalar_out.iter()) {
            assert_eq!(a, b);
        }
        // Check values: 1 + 2*x
        assert_eq!(simd_out[0], Value::Continuous(1.0));
        assert_eq!(simd_out[1], Value::Continuous(3.0));
        assert_eq!(simd_out[7], Value::Continuous(15.0));
    }

    #[test]
    fn simd_vs_scalar_with_missing() {
        let reg = make_reg();
        let rows = vec![
            vec![Value::Continuous(2.0)],
            vec![Value::Missing],
            vec![Value::Continuous(3.0)],
            vec![Value::Continuous(4.0)],
        ];
        let refs: Vec<&[Value]> = rows.iter().map(|r| r.as_slice()).collect();
        let simd_out = evaluate_regression_batch_simd(&reg, &refs);
        let scalar_out = evaluate_regression_batch_scalar(&reg, &refs);
        assert_eq!(simd_out, scalar_out);
    }
}
