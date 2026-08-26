//! AnomalyDetectionModel evaluation — wrapper over an embedded model.
//!
//! Implements `AnomalyDetectionModel` scoring per `pmml.xsd:1718-1737` and
//! `https://dmg.org/pmml/v4-4-1/AnomalyDetectionModel.html`.
//! The outer model delegates to its embedded `MODEL-ELEMENT`, then maps the raw
//! score to an anomaly score per `algorithmType`:
//!
//! - `iforest` → `2^(-avg_path / c(n))` where `c(n)=2*H(n-1)-2*(n-1)/n`, `H(k)=ln(k)+γ`.
//! - `clusterMeanDist` → `distance_to_cluster / mean_cluster_distances[cluster_idx]`.
//! - `ocsvm` / `other` → raw embedded score (no transform).
//!
//! # What belongs here
//!
//! - [`evaluate_anomaly_detection`] — public entry point, pure `(&AnomalyDetectionIr, &[Value]) -> Value`.
//!
//! # Performance
//!
//! `O(1)` plus embedded model cost. No allocation beyond embedded evaluation.

use crate::base::Value;
use crate::ir::{AnomalyDetectionIr, ClusteringIr, ModelIr};

/// Evaluate an [`AnomalyDetectionIr`] against a dense `values` array.
///
/// Delegates to the embedded model, then applies algorithm-specific normalization.
///
/// # Parameters
///
/// - `model`: Lowered anomaly detection model with `algorithm_type`, `sample_data_size`, `model`, and optional `mean_cluster_distances`.
/// - `values`: Dense `&[Value]` indexed by `FieldId`. Out-of-bounds is `Missing`.
///
/// # Returns
///
/// `Continuous` anomaly score, or `Missing` when embedded evaluation is missing or
/// `clusterMeanDist` cannot find a cluster.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, Value};
/// use pmmlruntime::ir::*;
/// use pmmlruntime::engine::models::evaluate_anomaly_detection;
///
/// let f = FieldId(0);
/// let tree = TreeIr {
///     function_name: "regression".into(),
///     missing_value_strategy: MissingValueStrategy::NullPrediction,
///     no_true_child_strategy: NoTrueChildStrategy::ReturnNullPrediction,
///     nodes: vec![NodeIr { id: None, score: Some(SymbolIdOrContinuous::Continuous(3.5)), predicate: PredicateIr::True, children: vec![], default_child: None, score_distributions: vec![] }],
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     targets: vec![], output: vec![],
/// };
/// let adm = AnomalyDetectionIr {
///     function_name: "regression".into(),
///     algorithm_type: "iforest".into(),
///     sample_data_size: Some(5.0),
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     output: vec![], targets: vec![],
///     model: Box::new(ModelIr::Tree(tree)),
///     mean_cluster_distances: None,
/// };
/// let score = evaluate_anomaly_detection(&adm, &[Value::Continuous(1.0)]);
/// // 2^(-3.5 / c(5)) ≈ 0.3525
/// match score { Value::Continuous(v) => assert!((v - 0.3525).abs() < 1e-4), _ => panic!() }
/// ```
pub fn evaluate_anomaly_detection(model: &AnomalyDetectionIr, values: &[Value]) -> Value {
    // First evaluate embedded model to get raw score
    let raw = evaluate_embedded(&model.model, values);
    // Handle Missing propagation
    if raw.is_missing() {
        return Value::Missing;
    }
    match model.algorithm_type.as_str() {
        "iforest" => {
            // Expect raw continuous avg_path_length
            let avg_path = match raw {
                Value::Continuous(f) => f,
                Value::Discrete(_) => return raw, // shouldn't happen, return as is
                Value::Missing => return Value::Missing,
            };
            let n = model.sample_data_size.unwrap_or(256.0);
            let cn = c_n(n);
            if cn == 0.0 || !cn.is_finite() {
                return Value::Continuous(avg_path);
            }
            let score = 2.0_f64.powf(-avg_path / cn);
            Value::Continuous(score)
        }
        "clusterMeanDist" => {
            // Embedded should be ClusteringModel; we need distance ratio
            // If embedded is clustering, compute distance and divide by mean
            if let ModelIr::Clustering(clustering) = model.model.as_ref() {
                if let Some(mean_dists) = &model.mean_cluster_distances {
                    // compute clustering assignment and distance
                    let (cluster_idx, dist) = clustering_distance_and_index(clustering, values);
                    if let (Some(idx), Some(d)) = (cluster_idx, dist) {
                        if idx < mean_dists.len() {
                            let mean = mean_dists[idx];
                            if mean == 0.0 {
                                // spec says special treatment for mean 0: return 0 if dist==0 else infinity?
                                // Use 0 if both 0 else large
                                if d == 0.0 {
                                    return Value::Continuous(0.0);
                                } else {
                                    return Value::Continuous(f64::INFINITY);
                                }
                            }
                            return Value::Continuous(d / mean);
                        }
                    }
                    return Value::Missing;
                } else {
                    // no mean distances → return raw distance? But spec requires mean; fallback to raw
                    return raw;
                }
            } else {
                // Not clustering → fallback to raw
                return raw;
            }
        }
        "ocsvm" | "other" | _ => raw,
    }
}

fn evaluate_embedded(model: &ModelIr, values: &[Value]) -> Value {
    match model {
        ModelIr::Tree(t) => crate::engine::models::evaluate_tree(t, values),
        ModelIr::Regression(r) => crate::engine::models::evaluate_regression(r, values),
        ModelIr::Mining(m) => {
            // For mining we need field_names maps; but anomaly embedded mining is typically evaluated without those
            // For simplicity, if mining, we need to handle via evaluate_mining with empty maps fallback
            // We'll use empty maps and expect it to still work for basic tree segments (since mining delegates).
            // To avoid allocation, create empty maps on stack; need mutable copy for evaluate_mining
            let empty_field_names: std::collections::HashMap<crate::base::FieldId, String> =
                std::collections::HashMap::new();
            let empty_symbol_names: std::collections::HashMap<crate::base::SymbolId, String> =
                std::collections::HashMap::new();
            let empty_name_to_id: std::collections::HashMap<String, crate::base::FieldId> =
                std::collections::HashMap::new();
            let mut values_mut = values.to_vec();
            crate::engine::models::evaluate_mining(
                m,
                &mut values_mut,
                &empty_field_names,
                &empty_symbol_names,
                &empty_name_to_id,
            )
        }
        ModelIr::Scorecard(s) => crate::engine::models::evaluate_scorecard(s, values),
        ModelIr::Clustering(c) => crate::engine::models::evaluate_clustering(c, values),
        ModelIr::NaiveBayes(nb) => crate::engine::models::evaluate_naive_bayes(nb, values),
        ModelIr::NearestNeighbor(nn) => {
            crate::engine::models::evaluate_nearest_neighbor(nn, values, None, None)
        }
        ModelIr::SupportVectorMachine(svm) => {
            crate::engine::models::evaluate_support_vector_machine(svm, values)
        }
        ModelIr::NeuralNetwork(nn) => crate::engine::models::evaluate_neural_network(nn, values),
        ModelIr::GeneralRegression(gr) => {
            let empty_field_names = std::collections::HashMap::new();
            let empty_symbol_names = std::collections::HashMap::new();
            let empty_name_to_id = std::collections::HashMap::new();
            crate::engine::models::evaluate_general_regression(
                gr,
                values,
                &empty_field_names,
                &empty_symbol_names,
                &empty_name_to_id,
            )
        }
        ModelIr::Association(a) => crate::engine::models::evaluate_association(a, values),
        ModelIr::RuleSet(r) => crate::engine::models::evaluate_rule_set(r, values),
        ModelIr::AnomalyDetection(inner) => evaluate_anomaly_detection(inner, values),
        ModelIr::Baseline(b) => crate::engine::models::evaluate_baseline(b, values),
    }
}

fn c_n(n: f64) -> f64 {
    if n <= 1.0 {
        return 1.0;
    }
    // H(n-1) approx ln(n-1)+γ
    let hm1 = (n - 1.0).ln() + 0.5772156649015328606_f64;
    2.0 * hm1 - 2.0 * (n - 1.0) / n
}

fn clustering_distance_and_index(
    clustering: &ClusteringIr,
    values: &[Value],
) -> (Option<usize>, Option<f64>) {
    if clustering.clusters.is_empty() || clustering.clustering_fields.is_empty() {
        return (None, None);
    }
    let mut input_vec = Vec::new();
    for &fid in &clustering.clustering_fields {
        let idx = fid.as_usize();
        let v = if idx < values.len() {
            values[idx]
        } else {
            Value::Missing
        };
        match v {
            Value::Continuous(f) => input_vec.push(f),
            _ => return (None, None),
        }
    }
    let mut best_idx: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    for (i, cluster) in clustering.clusters.iter().enumerate() {
        let d = distance(&input_vec, &cluster.array, &clustering.comparison_measure);
        if d < best_dist {
            best_dist = d;
            best_idx = Some(i);
        }
    }
    if best_dist.is_infinite() {
        (None, None)
    } else {
        (best_idx, Some(best_dist))
    }
}

fn distance(a: &[f64], b: &[f64], measure: &str) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    match measure {
        "squaredEuclidean" => a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum(),
        "euclidean" => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt(),
        "manhattan" => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        "chebyshev" => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max),
        _ => a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, SymbolId, Value};
    use crate::ir::*;

    #[test]
    fn iforest_normalization() {
        let f = FieldId(0);
        let tree = TreeIr {
            function_name: "regression".into(),
            missing_value_strategy: MissingValueStrategy::NullPrediction,
            no_true_child_strategy: NoTrueChildStrategy::ReturnNullPrediction,
            nodes: vec![NodeIr {
                id: None,
                score: Some(SymbolIdOrContinuous::Continuous(3.5)),
                predicate: PredicateIr::True,
                children: vec![],
                default_child: None,
                score_distributions: vec![],
            }],
            mining_schema: MiningSchemaIr {
                active_fields: vec![f],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            targets: vec![],
            output: vec![],
        };
        let adm = AnomalyDetectionIr {
            function_name: "regression".into(),
            algorithm_type: "iforest".into(),
            sample_data_size: Some(5.0),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            model: Box::new(ModelIr::Tree(tree)),
            mean_cluster_distances: None,
        };
        let score = evaluate_anomaly_detection(&adm, &[Value::Continuous(1.0)]);
        if let Value::Continuous(v) = score {
            // expected ~0.352557
            assert!((v - 0.3525).abs() < 1e-3, "got {v}");
        } else {
            panic!("expected continuous");
        }
    }

    #[test]
    fn cluster_mean_dist_ratio() {
        let f = FieldId(0);
        let s1 = SymbolId(0);
        let s2 = SymbolId(1);
        let clustering = ClusteringIr {
            function_name: "clustering".into(),
            model_class: "centerBased".into(),
            number_of_clusters: 2,
            mining_schema: MiningSchemaIr {
                active_fields: vec![f],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            comparison_measure: "euclidean".into(),
            clustering_fields: vec![f],
            clusters: vec![
                ClusterIr {
                    name: s1,
                    name_str: "c1".into(),
                    array: vec![0.0],
                },
                ClusterIr {
                    name: s2,
                    name_str: "c2".into(),
                    array: vec![10.0],
                },
            ],
            output: vec![],
        };
        let adm = AnomalyDetectionIr {
            function_name: "clustering".into(),
            algorithm_type: "clusterMeanDist".into(),
            sample_data_size: None,
            mining_schema: MiningSchemaIr {
                active_fields: vec![f],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            model: Box::new(ModelIr::Clustering(clustering)),
            mean_cluster_distances: Some(vec![1.0, 2.0]),
        };
        // input 1.0 -> closest to c1 distance 1.0 ratio 1.0
        let score = evaluate_anomaly_detection(&adm, &[Value::Continuous(1.0)]);
        assert_eq!(score, Value::Continuous(1.0));
        // input 12.0 -> closest to c2 distance 2.0 ratio 1.0
        let score2 = evaluate_anomaly_detection(&adm, &[Value::Continuous(12.0)]);
        assert_eq!(score2, Value::Continuous(1.0));
        // input 14.0 -> distance 4.0 ratio 2.0
        let score3 = evaluate_anomaly_detection(&adm, &[Value::Continuous(14.0)]);
        assert_eq!(score3, Value::Continuous(2.0));
    }
}
