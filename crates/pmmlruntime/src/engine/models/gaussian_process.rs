//! GaussianProcessModel evaluation — kernel-weighted regression / classification.
//!
//! Implements `GaussianProcessModel` scoring per `pmml.xsd:4405-4489`.
//! The model holds a kernel (`RadialBasisKernel`, `ARDSquaredExponentialKernel`,
//! `AbsoluteExponentialKernel`, `GeneralizedExponentialKernel`) and `TrainingInstances`
//! (instance fields + inline table). Scoring computes the kernel `k(x, x_i)` between
//! the query vector `x` (active fields) and each training vector `x_i`, then
//! predicts:
//!
//! - regression (`functionName="regression"`): weighted average `Σ k_i * y_i / Σ k_i`
//! - classification: weighted vote `argmax_c Σ_{i:y_i=c} k_i`
//!
//! `Missing` propagates if any active input is missing or if no training data.
//! `noiseVariance` is currently unused for mean prediction (variance prediction is not yet exposed).
//!
//! # What belongs here
//!
//! - [`evaluate_gaussian_process`] — public entry point `(&GaussianProcessIr, &[Value]) -> Value`.

use crate::base::Value;
use crate::ir::{GaussianKernelIr, GaussianProcessIr};
use std::collections::HashMap;

/// Evaluate a [`GaussianProcessIr`] against a dense `values` array.
pub fn evaluate_gaussian_process(model: &GaussianProcessIr, values: &[Value]) -> Value {
    if model.training_vectors.is_empty() || model.training_targets.is_empty() {
        return Value::Missing;
    }
    // Determine vector fields order (same as lower)
    let vector_fields: Vec<crate::base::FieldId> = if !model.mining_schema.active_fields.is_empty()
    {
        model.mining_schema.active_fields.clone()
    } else {
        model
            .instance_fields
            .iter()
            .copied()
            .filter(|fid| Some(*fid) != model.mining_schema.target_field)
            .collect()
    };
    if vector_fields.is_empty() {
        return Value::Missing;
    }
    // Build query vector
    let mut query: Vec<f64> = Vec::with_capacity(vector_fields.len());
    for &fid in &vector_fields {
        let idx = fid.as_usize();
        let v = if idx < values.len() {
            values[idx]
        } else {
            Value::Missing
        };
        match v {
            Value::Continuous(f) => {
                if !f.is_finite() {
                    return Value::Missing;
                }
                query.push(f)
            }
            Value::Missing => return Value::Missing,
            Value::Discrete(_) => return Value::Missing,
        }
    }
    // Compute kernel weights
    let mut weights: Vec<f64> = Vec::with_capacity(model.training_vectors.len());
    for train_vec in &model.training_vectors {
        if train_vec.len() != query.len() {
            // dimension mismatch -> weight 0
            weights.push(0.0);
            continue;
        }
        let k = kernel_value(&model.kernel, &query, train_vec);
        // kernel should be in (0,1]; clamp non-finite to 0
        let w = if k.is_finite() && k >= 0.0 { k } else { 0.0 };
        weights.push(w);
    }
    let sum_w: f64 = weights.iter().sum();
    if sum_w == 0.0 || !sum_w.is_finite() {
        return Value::Missing;
    }
    // Check target type to decide regression vs classification
    let is_classification = model.function_name == "classification"
        || model
            .training_targets
            .iter()
            .any(|v| matches!(v, Value::Discrete(_)));
    if is_classification {
        // weighted vote
        let mut votes: HashMap<crate::base::SymbolId, f64> = HashMap::new();
        for (w, target) in weights.iter().zip(model.training_targets.iter()) {
            match target {
                Value::Discrete(sid) => {
                    *votes.entry(*sid).or_insert(0.0) += *w;
                }
                Value::Continuous(f) => {
                    // For classification but target is continuous (unlikely) treat as discrete via symbol?
                    // Try to treat continuous target as categorical string key via its bits? Instead fallback to average path.
                    // To still produce discrete, map continuous to Symbol via interning? But we lack interner.
                    // For now treat continuous as not vote.
                    let _ = f;
                }
                Value::Missing => {}
            }
        }
        if votes.is_empty() {
            return Value::Missing;
        }
        // pick max weight
        let mut best_sid = None;
        let mut best_weight = f64::NEG_INFINITY;
        for (sid, w) in votes {
            if w > best_weight {
                best_weight = w;
                best_sid = Some(sid);
            }
        }
        if let Some(sid) = best_sid {
            return Value::Discrete(sid);
        }
        return Value::Missing;
    } else {
        // regression weighted average
        let mut weighted_sum = 0.0;
        let mut weight_used = 0.0;
        for (w, target) in weights.iter().zip(model.training_targets.iter()) {
            match target {
                Value::Continuous(f) => {
                    if f.is_finite() {
                        weighted_sum += *w * *f;
                        weight_used += *w;
                    }
                }
                Value::Discrete(sid) => {
                    // discrete stored but function is regression: try to parse symbol as f64? Can't without symbol map.
                    // Seek fallback: treat discrete as 0.0? Better skip.
                    // Attempt to recover via symbol: we don't have map, so skip.
                    let _ = sid;
                }
                Value::Missing => {}
            }
        }
        if weight_used == 0.0 {
            return Value::Missing;
        }
        // If all targets were discrete but function is regression, weighted_sum stays 0 -> Missing already handled
        // Normalize by sum_w but we used weight_used which equals sum_w minus missing targets; use sum_w for proper average if some targets missing?
        // Use weight_used for numerator's weights, denominator sum_w would be slightly off if some targets missing. Use weight_used.
        let pred = weighted_sum / weight_used;
        if !pred.is_finite() {
            return Value::Missing;
        }
        Value::Continuous(pred)
    }
}

fn kernel_value(kernel: &GaussianKernelIr, x: &[f64], y: &[f64]) -> f64 {
    match kernel {
        GaussianKernelIr::RadialBasis { gamma, lambda, .. } => {
            let dist2: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            let scaled = *gamma * *lambda * dist2;
            (-scaled).exp()
        }
        GaussianKernelIr::ARDSquaredExponential { gamma, lambdas, .. } => {
            // Flatten lambdas to vec
            let flat: Vec<f64> = lambdas.iter().flat_map(|v| v.iter().copied()).collect();
            if flat.is_empty() || flat.len() != x.len() {
                let dist2: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                (-*gamma * dist2).exp()
            } else {
                let mut sum = 0.0;
                for ((a, b), &lambda) in x.iter().zip(y.iter()).zip(flat.iter()) {
                    if lambda == 0.0 {
                        // avoid div0
                        sum += (a - b).powi(2) * 1e6;
                    } else {
                        sum += (a - b).powi(2) / (2.0 * lambda * lambda);
                    }
                }
                (-*gamma * sum).exp()
            }
        }
        GaussianKernelIr::AbsoluteExponential { gamma, lambdas, .. } => {
            let flat: Vec<f64> = lambdas.iter().flat_map(|v| v.iter().copied()).collect();
            if flat.is_empty() || flat.len() != x.len() {
                let dist1: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
                (-*gamma * dist1).exp()
            } else {
                let mut sum = 0.0;
                for ((a, b), &lambda) in x.iter().zip(y.iter()).zip(flat.iter()) {
                    if lambda == 0.0 {
                        sum += (a - b).abs() * 1e6;
                    } else {
                        sum += (a - b).abs() / lambda.abs();
                    }
                }
                (-*gamma * sum).exp()
            }
        }
        GaussianKernelIr::GeneralizedExponential {
            gamma,
            lambdas,
            degree,
            ..
        } => {
            let flat: Vec<f64> = lambdas.iter().flat_map(|v| v.iter().copied()).collect();
            let distance = if flat.is_empty() || flat.len() != x.len() {
                let dist2: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                dist2.sqrt()
            } else {
                let mut sum = 0.0;
                for ((a, b), &lambda) in x.iter().zip(y.iter()).zip(flat.iter()) {
                    let d = if lambda == 0.0 {
                        (a - b) * 1e6
                    } else {
                        (a - b) / lambda
                    };
                    sum += d * d;
                }
                sum.sqrt()
            };
            let deg = *degree;
            if deg == 0.0 {
                0.0
            } else {
                (-*gamma * distance.powf(deg)).exp()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, Value};
    use crate::ir::*;

    fn make_gp_model(kernel: GaussianKernelIr) -> GaussianProcessIr {
        let f1 = FieldId(0);
        let f2 = FieldId(1);
        let target = FieldId(2);
        GaussianProcessIr {
            function_name: "regression".into(),
            model_name: Some("gp".into()),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f1, f2],
                target_field: Some(target),
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            kernel,
            instance_fields: vec![f1, f2, target],
            training_instances: vec![],
            training_vectors: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            training_targets: vec![Value::Continuous(0.0), Value::Continuous(10.0)],
            is_transformed: false,
        }
    }

    #[test]
    fn gp_rbf_weighted_average() {
        let model = make_gp_model(GaussianKernelIr::RadialBasis {
            gamma: 1.0,
            noise_variance: 1.0,
            lambda: 1.0,
            description: None,
        });
        // query at (0,0) should be close to 0
        let q = vec![
            Value::Continuous(0.0),
            Value::Continuous(0.0),
            Value::Missing,
        ];
        // need values length 3, target missing placeholder
        let mut values = vec![Value::Missing; 3];
        values[0] = Value::Continuous(0.0);
        values[1] = Value::Continuous(0.0);
        let pred = evaluate_gaussian_process(&model, &values);
        if let Value::Continuous(v) = pred {
            // weighted average should be near 0 because k for (0,0) =1, for (1,1) = exp(-2)=0.135
            // pred = (1*0 +0.135*10)/(1.135)= 1.19
            assert!(v < 2.0, "got {}", v);
            assert!(v > 0.5, "got {}", v);
        } else {
            panic!("expected continuous");
        }
        // query at (1,1) near 10
        values[0] = Value::Continuous(1.0);
        values[1] = Value::Continuous(1.0);
        let pred2 = evaluate_gaussian_process(&model, &values);
        if let Value::Continuous(v) = pred2 {
            assert!(v > 8.0, "got {}", v);
        } else {
            panic!()
        }
        let _ = q;
    }

    #[test]
    fn gp_missing_input() {
        let model = make_gp_model(GaussianKernelIr::RadialBasis {
            gamma: 0.5,
            noise_variance: 1.0,
            lambda: 1.0,
            description: None,
        });
        let values = vec![Value::Missing, Value::Continuous(0.0), Value::Missing];
        assert_eq!(evaluate_gaussian_process(&model, &values), Value::Missing);
    }

    #[test]
    fn gp_ard_kernel() {
        let model = make_gp_model(GaussianKernelIr::ARDSquaredExponential {
            gamma: 1.0,
            noise_variance: 1.0,
            lambdas: vec![vec![1.0, 2.0]],
            description: None,
        });
        let values = vec![
            Value::Continuous(0.5),
            Value::Continuous(0.5),
            Value::Missing,
        ];
        let pred = evaluate_gaussian_process(&model, &values);
        assert!(matches!(pred, Value::Continuous(_)));
    }

    #[test]
    fn gp_classification_vote() {
        let f1 = FieldId(0);
        let target = FieldId(1);
        let sid_a = crate::base::SymbolId(10);
        let sid_b = crate::base::SymbolId(11);
        let model = GaussianProcessIr {
            function_name: "classification".into(),
            model_name: None,
            mining_schema: MiningSchemaIr {
                active_fields: vec![f1],
                target_field: Some(target),
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            kernel: GaussianKernelIr::RadialBasis {
                gamma: 1.0,
                noise_variance: 1.0,
                lambda: 1.0,
                description: None,
            },
            instance_fields: vec![f1, target],
            training_instances: vec![],
            training_vectors: vec![vec![0.0], vec![10.0], vec![0.1]],
            training_targets: vec![
                Value::Discrete(sid_a),
                Value::Discrete(sid_b),
                Value::Discrete(sid_a),
            ],
            is_transformed: false,
        };
        let values = vec![Value::Continuous(0.05), Value::Missing];
        let pred = evaluate_gaussian_process(&model, &values);
        assert_eq!(pred, Value::Discrete(sid_a));
    }
}
