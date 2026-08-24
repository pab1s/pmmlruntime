use pmml_core::Value;
use pmml_ir::ir::{
    CompoundOperator, MiningIr, MissingPredictionTreatment, ModelIr, MultipleModelMethod,
    PredicateIr, SimpleOperator, SymbolIdOrContinuous,
};
use std::collections::HashMap;

/// Evaluate mining model (segmentation) given values array.
/// `values` is mutable shared array; segment outputs are written back to it.
/// Returns final predicted Value.
pub fn evaluate_mining(
    mining: &MiningIr,
    values: &mut [pmml_core::Value],
    field_names: &HashMap<pmml_core::FieldId, String>,
    symbol_names: &HashMap<pmml_core::SymbolId, String>,
    name_to_id: &HashMap<String, pmml_core::FieldId>,
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
                let pred = crate::models::tree::evaluate_tree(tree, values);
                if mining.segmentation.multiple_model_method == MultipleModelMethod::ModelChain {
                    write_tree_outputs(tree, pred, values, field_names, symbol_names, name_to_id);
                }
                pred
            }
            ModelIr::Regression(reg) => crate::models::regression::evaluate_regression(reg, values),
            ModelIr::Scorecard(sc) => crate::models::scorecard::evaluate_scorecard(sc, values),
            ModelIr::Clustering(cl) => crate::models::clustering::evaluate_clustering(cl, values),
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
            let mut counts: HashMap<pmml_core::SymbolId, f64> = HashMap::new();
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
    tree: &pmml_ir::ir::TreeIr,
    predicted: Value,
    values: &mut [Value],
    _field_names: &HashMap<pmml_core::FieldId, String>,
    symbol_names: &HashMap<pmml_core::SymbolId, String>,
    name_to_id: &HashMap<String, pmml_core::FieldId>,
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
                if out.feature == pmml_core::field::ResultFeature::Probability {
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

fn eval_predicate(pred: &PredicateIr, values: &[Value]) -> bool {
    match pred {
        PredicateIr::True => true,
        PredicateIr::Simple {
            field,
            operator,
            value,
        } => {
            let idx = field.as_usize();
            let actual = if idx < values.len() {
                values[idx]
            } else {
                Value::Missing
            };
            match operator {
                SimpleOperator::IsMissing => actual.is_missing(),
                SimpleOperator::IsNotMissing => !actual.is_missing(),
                _ => {
                    if actual.is_missing() {
                        return false;
                    }
                    match (actual, value) {
                        (Value::Continuous(a), SymbolIdOrContinuous::Continuous(b)) => {
                            match operator {
                                SimpleOperator::Equal => (a - b).abs() < 1e-9,
                                SimpleOperator::NotEqual => (a - b).abs() >= 1e-9,
                                SimpleOperator::LessThan => a < *b,
                                SimpleOperator::LessOrEqual => a <= *b,
                                SimpleOperator::GreaterThan => a > *b,
                                SimpleOperator::GreaterOrEqual => a >= *b,
                                _ => false,
                            }
                        }
                        (Value::Discrete(sid), SymbolIdOrContinuous::Symbol(s)) => match operator {
                            SimpleOperator::Equal => sid == *s,
                            SimpleOperator::NotEqual => sid != *s,
                            _ => false,
                        },
                        _ => false,
                    }
                }
            }
        }
        PredicateIr::SimpleSet {
            field,
            is_in,
            array,
        } => {
            let idx = field.as_usize();
            let actual = if idx < values.len() {
                values[idx]
            } else {
                Value::Missing
            };
            if actual.is_missing() {
                return false;
            }
            let mut found = false;
            for v in array {
                let matches = match (actual, v) {
                    (Value::Discrete(sid), SymbolIdOrContinuous::Symbol(s)) => sid == *s,
                    (Value::Continuous(a), SymbolIdOrContinuous::Continuous(b)) => {
                        (a - b).abs() < 1e-9
                    }
                    _ => false,
                };
                if matches {
                    found = true;
                    break;
                }
            }
            if *is_in {
                found
            } else {
                !found
            }
        }
        PredicateIr::Compound {
            operator,
            predicates,
        } => match operator {
            CompoundOperator::And => predicates.iter().all(|p| eval_predicate(&**p, values)),
            CompoundOperator::Or => predicates.iter().any(|p| eval_predicate(&**p, values)),
            CompoundOperator::Xor => {
                let mut c = 0;
                for p in predicates.iter() {
                    if eval_predicate(&**p, values) {
                        c += 1;
                    }
                }
                c == 1
            }
            CompoundOperator::Surrogate => {
                for p in predicates.iter() {
                    let field_missing = match &**p {
                        PredicateIr::Simple { field, .. } => {
                            let idx = field.as_usize();
                            idx < values.len() && values[idx].is_missing()
                        }
                        PredicateIr::SimpleSet { field, .. } => {
                            let idx = field.as_usize();
                            idx < values.len() && values[idx].is_missing()
                        }
                        _ => false,
                    };
                    if field_missing {
                        continue;
                    }
                    if eval_predicate(&**p, values) {
                        return true;
                    }
                    return false;
                }
                false
            }
        },
    }
}
