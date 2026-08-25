//! MiningModel (segmentation) evaluation — combining segment predictions.
//!
//! Implements `MiningModel` with `Segmentation`. Each `Segment` has a
//! [`PredicateIr`](crate::ir::PredicateIr) tested via [`crate::engine::predicate::eval_predicate`];
//! when true the segment's boxed `ModelIr` is evaluated (`Tree`, `Regression`,
//! `Scorecard`, `Clustering`, etc.). `missingPredictionTreatment` (`ReturnMissing`,
//! `SkipSegment`, `Continue`) controls how `Missing` segment outputs affect the ensemble.
//! The ensemble is then combined per `multipleModelMethod` (`average`, `weightedAverage`,
//! `sum`, `weightedSum`, `majorityVote`, `weightedMajorityVote`, `selectFirst`,
//! `modelChain`, etc.). `modelChain` writes segment outputs back into `values`
//! so later segments can consume them as additional fields.
//!
//! # What belongs here
//!
//! - [`evaluate_mining`] — the single public entry point.
//!
//! # Performance
//!
//! `O(segments * segment_model_cost)`. No per-row allocation beyond a small `Vec<(Value, weight)>`.

use crate::base::Value;
use crate::ir::{
    MiningIr, MissingPredictionTreatment, ModelIr, MultipleModelMethod, SymbolIdOrContinuous,
};
use std::collections::HashMap;

use crate::engine::predicate::eval_predicate;

/// Evaluate a segmented [`MiningIr`] against a mutable `values` array.
///
/// Tests each segment's predicate via [`eval_predicate`](crate::engine::predicate::eval_predicate). For each
/// matching segment its model is evaluated (currently `Tree`, `Regression`, `Scorecard`, `Clustering`;
/// other types yield `Missing`). `ModelChain` writes the segment's prediction into
/// `values` for the next segment (target field and any field aliased by `Output`).
///
/// Missing handling per `missingPredictionTreatment`:
/// - `ReturnMissing` → `Missing` immediately.
/// - `SkipSegment` → ignore this segment.
/// - `Continue` → push `Missing` into the predictions list and continue.
///
/// Combination per `multipleModelMethod`:
/// - `Average` → mean of continuous predictions.
/// - `WeightedAverage` → weighted mean.
/// - `Sum` / `WeightedSum` → sum / weighted sum.
/// - `MajorityVote` / `WeightedMajorityVote` → most frequent `Discrete` value.
/// - `SelectFirst` / `SelectAll` → first prediction.
/// - `ModelChain` → last prediction.
/// - Others → last prediction.
///
/// # Parameters
///
/// - `mining`: The lowered `MiningModel`.
/// - `values`: Dense mutable `&mut [Value]` indexed by [`FieldId`](crate::base::FieldId). For `ModelChain`,
///   intermediate outputs are written back here at `targetField` and any `OutputField` alias.
/// - `field_names`: `FieldId → name` snapshot from `Ir` (for `ModelChain` tree probability aliasing).
/// - `symbol_names`: `SymbolId → string` for probability lookup in `Tree` segment `ScoreDistribution`.
/// - `name_to_id`: `name → FieldId` reverse map for writing `ModelChain` outputs.
///
/// # Returns
///
/// Combined predicted [`Value`] (typically `Continuous` for regression ensembles, `Discrete` for majority vote,
/// `Missing` when no segment matches or `ReturnMissing` triggers).
///
/// # Panics
///
/// Never panics. All `FieldId` indexing is bounds-checked.
///
/// # Performance
///
/// `O(segments)` predicate tests plus the cost of each segment's `evaluate_*`. Allocates only the `predictions` vec.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, SymbolId, Value};
/// use pmmlruntime::ir::*;
/// use pmmlruntime::engine::models::evaluate_mining;
/// use std::collections::HashMap;
///
/// let f = FieldId(0);
/// let tree = TreeIr {
///     function_name: "regression".into(),
///     missing_value_strategy: MissingValueStrategy::None,
///     no_true_child_strategy: NoTrueChildStrategy::ReturnNullPrediction,
///     nodes: vec![NodeIr { id: None, score: Some(SymbolIdOrContinuous::Continuous(1.0)), predicate: PredicateIr::True, children: vec![], default_child: None, score_distributions: vec![] }],
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     targets: vec![], output: vec![],
/// };
/// let mining = MiningIr {
///     function_name: "regression".into(),
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     segmentation: SegmentationIr {
///         multiple_model_method: MultipleModelMethod::Average,
///         missing_prediction_treatment: MissingPredictionTreatment::ReturnMissing,
///         segments: vec![SegmentIr { id: Some("s1".into()), predicate: PredicateIr::True, weight: 1.0, model: Box::new(ModelIr::Tree(tree)) }],
///     },
///     targets: vec![], output: vec![],
/// };
/// let mut values = vec![Value::Continuous(2.0)];
/// let pred = evaluate_mining(&mining, &mut values, &HashMap::new(), &HashMap::new(), &HashMap::new());
/// assert_eq!(pred, Value::Continuous(1.0));
/// ```
pub fn evaluate_mining(
    mining: &MiningIr,
    values: &mut [crate::base::Value],
    field_names: &HashMap<crate::base::FieldId, String>,
    symbol_names: &HashMap<crate::base::SymbolId, String>,
    name_to_id: &HashMap<String, crate::base::FieldId>,
) -> Value {
    let mut predictions: Vec<(Value, f64)> = Vec::new(); // (pred, weight)
    let mut last_pred: Option<Value> = None;

    for seg in &mining.segmentation.segments {
        // Evaluate predicate
        let pred_true = eval_predicate(&seg.predicate, values);
        if !pred_true {
            continue;
        }

        // Evaluate segment model
        let seg_pred = match &*seg.model {
            ModelIr::Tree(tree) => {
                let pred = crate::engine::models::tree::evaluate_tree(tree, values);
                if mining.segmentation.multiple_model_method == MultipleModelMethod::ModelChain {
                    write_tree_outputs(tree, pred, values, field_names, symbol_names, name_to_id);
                }
                pred
            }
            ModelIr::Regression(reg) => {
                crate::engine::models::regression::evaluate_regression(reg, values)
            }
            ModelIr::Scorecard(sc) => {
                crate::engine::models::scorecard::evaluate_scorecard(sc, values)
            }
            ModelIr::Clustering(cl) => {
                crate::engine::models::clustering::evaluate_clustering(cl, values)
            }
            ModelIr::Mining(_) => Value::Missing,
            _ => Value::Missing, // other models stub
        };

        if seg_pred.is_missing() {
            match mining.segmentation.missing_prediction_treatment {
                MissingPredictionTreatment::ReturnMissing => return Value::Missing,
                MissingPredictionTreatment::SkipSegment => continue,
                MissingPredictionTreatment::Continue => {
                    // treat as missing but continue
                    predictions.push((seg_pred, seg.weight));
                    last_pred = Some(seg_pred);
                    continue;
                }
            }
        }

        predictions.push((seg_pred, seg.weight));
        last_pred = Some(seg_pred);

        // For modelChain, write this segment's prediction as intermediate field for next segment
        if mining.segmentation.multiple_model_method == MultipleModelMethod::ModelChain {
            // The output of this segment may be used as input to next segment.
            // We already wrote Tree's probability outputs above. For Regression, its predictedValue is also an output.
            // For Regression segment, its output is typically named like "Pollen Index" or target field.
            // We need to ensure that regression's predictedValue is written to appropriate field for next segment if needed.
            // For v1, we write the prediction to the target field of this regression's mining_schema (if any)
            if let ModelIr::Regression(reg) = &*seg.model {
                if let Some(target_fid) = reg.mining_schema.target_field {
                    if (target_fid.as_usize()) < values.len() {
                        values[target_fid.as_usize()] = seg_pred;
                    }
                }
                // Also write output fields if any (e.g., "Pollen Index")
                for out in &reg.output {
                    if let Some(fid) = name_to_id.get(&out.name) {
                        if fid.as_usize() < values.len() {
                            values[fid.as_usize()] = seg_pred;
                        }
                    }
                }
            }
            // For Tree, also handle its target field
            if let ModelIr::Tree(tree) = &*seg.model {
                if let Some(target_fid) = tree.mining_schema.target_field {
                    if target_fid.as_usize() < values.len() {
                        values[target_fid.as_usize()] = seg_pred;
                    }
                }
            }
        }
    }

    // Combine per multipleModelMethod
    match mining.segmentation.multiple_model_method {
        MultipleModelMethod::ModelChain => last_pred.unwrap_or(Value::Missing),
        MultipleModelMethod::Average => {
            if predictions.is_empty() {
                return Value::Missing;
            }
            let sum: f64 = predictions
                .iter()
                .filter_map(|(v, _)| match v {
                    Value::Continuous(f) => Some(*f),
                    _ => None,
                })
                .sum();
            let count = predictions
                .iter()
                .filter(|(v, _)| matches!(v, Value::Continuous(_)))
                .count() as f64;
            if count == 0.0 {
                Value::Missing
            } else {
                Value::Continuous(sum / count)
            }
        }
        MultipleModelMethod::WeightedAverage => {
            let mut sum = 0.0;
            let mut wsum = 0.0;
            for (v, w) in &predictions {
                if let Value::Continuous(f) = v {
                    sum += f * w;
                    wsum += w;
                }
            }
            if wsum == 0.0 {
                Value::Missing
            } else {
                Value::Continuous(sum / wsum)
            }
        }
        MultipleModelMethod::Sum => {
            let sum: f64 = predictions
                .iter()
                .filter_map(|(v, _)| match v {
                    Value::Continuous(f) => Some(*f),
                    _ => None,
                })
                .sum();
            Value::Continuous(sum)
        }
        MultipleModelMethod::WeightedSum => {
            let sum: f64 = predictions
                .iter()
                .filter_map(|(v, w)| match v {
                    Value::Continuous(f) => Some(*f * *w),
                    _ => None,
                })
                .sum();
            Value::Continuous(sum)
        }
        MultipleModelMethod::MajorityVote | MultipleModelMethod::WeightedMajorityVote => {
            // For classification: vote for most frequent discrete value
            let mut counts: HashMap<crate::base::SymbolId, f64> = HashMap::new();
            for (v, w) in &predictions {
                if let Value::Discrete(sid) = v {
                    *counts.entry(*sid).or_default() += if matches!(
                        mining.segmentation.multiple_model_method,
                        MultipleModelMethod::WeightedMajorityVote
                    ) {
                        *w
                    } else {
                        1.0
                    };
                }
            }
            if counts.is_empty() {
                return Value::Missing;
            }
            let (best_sid, _) = counts
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            Value::Discrete(best_sid)
        }
        MultipleModelMethod::SelectFirst => predictions
            .into_iter()
            .next()
            .map(|(v, _)| v)
            .unwrap_or(Value::Missing),
        MultipleModelMethod::SelectAll => {
            // Return first for v1
            predictions
                .into_iter()
                .next()
                .map(|(v, _)| v)
                .unwrap_or(Value::Missing)
        }
        _ => last_pred.unwrap_or(Value::Missing),
    }
}

fn write_tree_outputs(
    tree: &crate::ir::TreeIr,
    predicted: Value,
    values: &mut [Value],
    _field_names: &HashMap<crate::base::FieldId, String>,
    symbol_names: &HashMap<crate::base::SymbolId, String>,
    name_to_id: &HashMap<String, crate::base::FieldId>,
) {
    // Find leaf node for this prediction to compute probabilities
    // We need to find the leaf node that gave this predicted value.
    // For now, we can compute probabilities from the leaf's ScoreDistributions.
    // We need to know which leaf was taken. Our evaluate_tree doesn't return leaf index.
    // For v1, we can approximate: leaf is the one where score == predicted.
    // Iterate nodes that are leaves (no children) and score matches.
    for node in &tree.nodes {
        if !node.children.is_empty() {
            continue;
        }
        let node_score = match node.score {
            Some(SymbolIdOrContinuous::Symbol(s)) => Value::Discrete(s),
            Some(SymbolIdOrContinuous::Continuous(f)) => Value::Continuous(f),
            _ => continue,
        };
        if node_score != predicted {
            continue;
        }
        // Found leaf
        let total: f64 = node
            .score_distributions
            .iter()
            .map(|sd| sd.record_count)
            .sum();
        if total == 0.0 {
            continue;
        }
        for sd in &node.score_distributions {
            let prob = sd.record_count / total;
            // Find output field with feature probability and value == sd.value
            for out in &tree.output {
                if out.feature == crate::base::field::ResultFeature::Probability {
                    if let Some(cat_sid) = out.value {
                        if cat_sid == sd.value {
                            // output field name is like "Probability_setosa" or "Probability_versicolor"
                            if let Some(fid) = name_to_id.get(&out.name) {
                                if fid.as_usize() < values.len() {
                                    values[fid.as_usize()] = Value::Continuous(prob);
                                }
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    // Also set predictedValue outputs? Not needed.
    let _ = symbol_names;
}

// eval_predicate now shared via crate::engine::predicate::eval_predicate (P5 deduplication)
