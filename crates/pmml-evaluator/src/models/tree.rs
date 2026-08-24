//! TreeModel evaluation — flat Node traversal with ONNX-style branchless predicates.

use pmml_core::{FieldId, Value};
use pmml_ir::ir::{
    CompoundOperator, NoTrueChildStrategy, PredicateIr,
    SimpleOperator, TreeIr,
};

fn eval_simple(
    field: FieldId,
    operator: SimpleOperator,
    value: &pmml_ir::ir::SymbolIdOrContinuous,
    values: &[Value],
) -> bool {
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
                return false; // missing fails for other operators unless isMissing
            }
            match (actual, value) {
                (Value::Continuous(a), pmml_ir::ir::SymbolIdOrContinuous::Continuous(b)) => {
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
                (Value::Discrete(sid), pmml_ir::ir::SymbolIdOrContinuous::Symbol(s)) => {
                    match operator {
                        SimpleOperator::Equal => sid == *s,
                        SimpleOperator::NotEqual => sid != *s,
                        _ => false, // for discrete, only equal/notEqual make sense; treat others as false
                    }
                }
                (Value::Continuous(a), pmml_ir::ir::SymbolIdOrContinuous::Symbol(_)) => {
                    // type mismatch: try interpret symbol as f64? For now false
                    let _ = a;
                    false
                }
                (Value::Discrete(_), pmml_ir::ir::SymbolIdOrContinuous::Continuous(_)) => false,
                _ => false,
            }
        }
    }
}

fn eval_predicate(pred: &PredicateIr, values: &[Value]) -> bool {
    match pred {
        PredicateIr::True => true,
        PredicateIr::Simple {
            field,
            operator,
            value,
        } => eval_simple(*field, *operator, value, values),
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
                    (Value::Discrete(sid), pmml_ir::ir::SymbolIdOrContinuous::Symbol(s)) => {
                        sid == *s
                    }
                    (Value::Continuous(a), pmml_ir::ir::SymbolIdOrContinuous::Continuous(b)) => {
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
                let mut true_count = 0;
                for p in predicates.iter() {
                    if eval_predicate(&**p, values) {
                        true_count += 1;
                    }
                }
                true_count == 1
            }
            CompoundOperator::Surrogate => {
                // Surrogate: evaluate predicates in order, first whose field is not missing
                // For v1 we approximate: return first predicate where field not missing and predicate true.
                // If none evaluatable, false.
                for p in predicates.iter() {
                    // Check if predicate's field is missing — need to extract field
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
                    // if first non-missing predicate is false, surrogate false (don't try next?)
                    // Actually JPMML tries next surrogate if primary fails due to missing? We'll just false for now.
                    return false;
                }
                false
            }
        },
    }
}

fn score_to_value(score: &Option<pmml_ir::ir::SymbolIdOrContinuous>) -> Option<Value> {
    match score {
        Some(pmml_ir::ir::SymbolIdOrContinuous::Continuous(f)) => Some(Value::Continuous(*f)),
        Some(pmml_ir::ir::SymbolIdOrContinuous::Symbol(s)) => Some(Value::Discrete(*s)),
        Some(pmml_ir::ir::SymbolIdOrContinuous::Missing) => Some(Value::Missing),
        None => None,
    }
}

/// Evaluate TreeIr given flat `values` array.
/// Returns predicted Value (Discrete for classification, Continuous for regression).
pub fn evaluate_tree(tree: &TreeIr, values: &[Value]) -> Value {
    if tree.nodes.is_empty() {
        return Value::Missing;
    }
    evaluate_node(0, tree, values, None)
}

fn evaluate_node(idx: usize, tree: &TreeIr, values: &[Value], last_score: Option<Value>) -> Value {
    let node = &tree.nodes[idx];
    let current_score = score_to_value(&node.score).or(last_score);

    // Find true child (first where predicate true)
    let mut true_child: Option<usize> = None;
    for &child_idx in &node.children {
        let child = &tree.nodes[child_idx];
        if eval_predicate(&child.predicate, values) {
            true_child = Some(child_idx);
            break;
        }
    }

    if let Some(child_idx) = true_child {
        // recurse, current becomes last_score for deeper
        evaluate_node(child_idx, tree, values, current_score)
    } else {
        // No true child
        if !node.children.is_empty() {
            // has children but none matched
            match tree.no_true_child_strategy {
                NoTrueChildStrategy::ReturnLastPrediction => {
                    current_score.unwrap_or(Value::Missing)
                }
                NoTrueChildStrategy::ReturnNullPrediction => Value::Missing,
            }
        } else {
            // leaf
            current_score.unwrap_or(Value::Missing)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::{FieldId, SymbolId, Value};
    use pmml_ir::ir::*;

    fn make_tree_simple() -> TreeIr {
        // Tree: root True score "A", child1 Simple Petal.Length <2.45 -> "setosa", child2 >=2.45 -> "versicolor"
        let f0 = FieldId(0);
        TreeIr {
            function_name: "classification".into(),
            missing_value_strategy: MissingValueStrategy::NullPrediction,
            no_true_child_strategy: NoTrueChildStrategy::ReturnLastPrediction,
            nodes: vec![
                NodeIr {
                    id: Some("1".into()),
                    score: Some(SymbolIdOrContinuous::Symbol(SymbolId(0))), // A placeholder
                    predicate: PredicateIr::True,
                    children: vec![1, 2],
                    score_distributions: vec![],
                },
                NodeIr {
                    id: Some("2".into()),
                    score: Some(SymbolIdOrContinuous::Symbol(SymbolId(1))),
                    predicate: PredicateIr::Simple {
                        field: f0,
                        operator: SimpleOperator::LessThan,
                        value: SymbolIdOrContinuous::Continuous(2.45),
                    },
                    children: vec![],
                    score_distributions: vec![],
                },
                NodeIr {
                    id: Some("3".into()),
                    score: Some(SymbolIdOrContinuous::Symbol(SymbolId(2))),
                    predicate: PredicateIr::Simple {
                        field: f0,
                        operator: SimpleOperator::GreaterOrEqual,
                        value: SymbolIdOrContinuous::Continuous(2.45),
                    },
                    children: vec![],
                    score_distributions: vec![],
                },
            ],
            mining_schema: MiningSchemaIr {
                active_fields: vec![f0],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            targets: vec![],
            output: vec![],
        }
    }

    #[test]
    fn tree_predicates() {
        let tree = make_tree_simple();
        let mut vals = vec![Value::Missing; 1];
        vals[0] = Value::Continuous(1.0);
        assert_eq!(evaluate_tree(&tree, &vals), Value::Discrete(SymbolId(1)));
        vals[0] = Value::Continuous(3.0);
        assert_eq!(evaluate_tree(&tree, &vals), Value::Discrete(SymbolId(2)));
    }

    #[test]
    fn missing_returns_last() {
        let tree = make_tree_simple();
        let vals = vec![Value::Missing; 1];
        // missing -> no child true -> returnLastPrediction -> root score
        assert_eq!(evaluate_tree(&tree, &vals), Value::Discrete(SymbolId(0)));
    }
}
