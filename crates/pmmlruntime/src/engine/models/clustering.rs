//! ClusteringModel evaluation — nearest centroid via comparison measure.
//!
//! Implements center-based clustering where each `Cluster` has a centroid vector
//! aligned with `clustering_fields`. The evaluator builds the input vector from
//! `values[ clustering_fields[i] ]` (all must be `Continuous`; `Missing` or `Discrete`
//! yields `Missing`) and computes distance to every cluster using the `ComparisonMeasure`
//! (`squaredEuclidean`, `euclidean`, `manhattan`, `chebyshev`; unknown defaults to
//! squared Euclidean). The nearest cluster's `name: SymbolId` is returned as `Discrete`.
//!
//! # What belongs here
//!
//! - [`evaluate_clustering`] — the single public entry point.
//!
//! # Performance
//!
//! `O(clusters * dims)` where `dims = clustering_fields.len()`. No allocation.

use crate::base::Value;
use crate::ir::ClusteringIr;

/// Evaluate a [`ClusteringIr`] against a dense `values` array.
///
/// Builds the input coordinate vector from `clustering.clustering_fields` and returns
/// the id of the nearest cluster per `comparison_measure`. All clustering inputs must
/// be `Continuous`; any `Missing` or `Discrete` input short-circuits to `Missing`.
///
/// Distance semantics:
///
/// - `squaredEuclidean` → `Σ (x - y)²`
/// - `euclidean` → `√Σ (x - y)²`
/// - `manhattan` → `Σ |x - y|`
/// - `chebyshev` → `max |x - y|`
/// - unknown → `Σ (x - y)²` (fallback)
///
/// Mismatched dimensionalities (`input_vec.len() != cluster.array.len()`) yield `INFINITY` for that cluster
/// so it can never win.
///
/// # Parameters
///
/// - `clustering`: Lowered clustering model (`ClusteringIr`) with `clustering_fields` and `clusters`.
/// - `values`: Dense `&[Value]` indexed by [`FieldId`](crate::base::FieldId). Out-of-bounds fields are `Missing`.
///
/// # Returns
///
/// `Discrete(cluster.name)` for the nearest centroid, or `Missing` when `clusters` or `clustering_fields` is empty,
/// any input coordinate is `Missing`/`Discrete`, or no finite distance was found.
///
/// # Panics
///
/// Never panics. All `FieldId` indexing is bounds-checked.
///
/// # Performance
///
/// `O(clusters * dims)` with no allocation and `f64` arithmetic. Vector construction from `values` is `O(dims)`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, SymbolId, Value};
/// use pmmlruntime::ir::*;
/// use pmmlruntime::engine::models::evaluate_clustering;
///
/// let f = FieldId(0);
/// let s_neg = SymbolId(0);
/// let s_pos = SymbolId(2);
/// let clustering = ClusteringIr {
///     function_name: "clustering".into(),
///     model_class: "centerBased".into(),
///     number_of_clusters: 2,
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     comparison_measure: "squaredEuclidean".into(),
///     clustering_fields: vec![f],
///     clusters: vec![
///         ClusterIr { name: s_neg, name_str: "negative".into(), array: vec![-3.0] },
///         ClusterIr { name: s_pos, name_str: "positive".into(), array: vec![3.0] },
///     ],
///     output: vec![],
/// };
/// assert_eq!(evaluate_clustering(&clustering, &[Value::Continuous(2.8)]), Value::Discrete(s_pos));
/// assert_eq!(evaluate_clustering(&clustering, &[Value::Missing]), Value::Missing);
/// ```
pub fn evaluate_clustering(clustering: &ClusteringIr, values: &[Value]) -> Value {
    if clustering.clusters.is_empty() || clustering.clustering_fields.is_empty() {
        return Value::Missing;
    }

    let mut input_vec: Vec<f64> = Vec::new();
    for &fid in &clustering.clustering_fields {
        let idx = fid.as_usize();
        let v = if idx < values.len() {
            values[idx]
        } else {
            Value::Missing
        };
        match v {
            Value::Continuous(f) => input_vec.push(f),
            Value::Missing => return Value::Missing,
            Value::Discrete(_) => return Value::Missing,
        }
    }

    let mut best_idx: Option<usize> = None;
    let mut best_dist = f64::INFINITY;

    for (i, cluster) in clustering.clusters.iter().enumerate() {
        let dist = distance(&input_vec, &cluster.array, &clustering.comparison_measure);
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(i);
        }
    }

    if let Some(idx) = best_idx {
        return Value::Discrete(clustering.clusters[idx].name);
    }

    Value::Missing
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
    fn clustering_1d() {
        let f = FieldId(0);
        let s_neg = SymbolId(0);
        let s_neu = SymbolId(1);
        let s_pos = SymbolId(2);
        let clustering = ClusteringIr {
            function_name: "clustering".into(),
            model_class: "centerBased".into(),
            number_of_clusters: 3,
            mining_schema: MiningSchemaIr {
                active_fields: vec![f],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            comparison_measure: "squaredEuclidean".into(),
            clustering_fields: vec![f],
            clusters: vec![
                ClusterIr {
                    name: s_neg,
                    name_str: "negative".into(),
                    array: vec![-3.0],
                },
                ClusterIr {
                    name: s_neu,
                    name_str: "neutral".into(),
                    array: vec![0.0],
                },
                ClusterIr {
                    name: s_pos,
                    name_str: "positive".into(),
                    array: vec![3.0],
                },
            ],
            output: vec![],
        };
        let vals = vec![Value::Continuous(2.8)];
        let pred = evaluate_clustering(&clustering, &vals);
        assert_eq!(pred, Value::Discrete(s_pos));
        let vals2 = vec![Value::Continuous(-2.9)];
        let pred2 = evaluate_clustering(&clustering, &vals2);
        assert_eq!(pred2, Value::Discrete(s_neg));
    }
}
