//! TreeModel evaluation — flat Node traversal with branchless predicates.

use pmml_core::Value;
use pmml_ir::ir::{MissingValueStrategy, NoTrueChildStrategy, TreeIr};

use crate::predicate::eval_predicate;

fn score_to_value(score: &Option<pmml_ir::ir::SymbolIdOrContinuous>) -> Option<Value> {
    match score {
        Some(pmml_ir::ir::SymbolIdOrContinuous::Continuous(f)) => Some(Value::Continuous(*f)),
        Some(pmml_ir::ir::SymbolIdOrContinuous::Symbol(s)) => Some(Value::Discrete(*s)),
        Some(pmml_ir::ir::SymbolIdOrContinuous::Missing) => Some(Value::Missing),
        None => None,
    }
}

/// Evaluate TreeIr given flat `values` array.
/// Iterative, no recursion, branch-friendly.
/// Handles missingValueStrategy (LastPrediction, NullPrediction, DefaultChild) and noTrueChildStrategy.
/// DefaultChild uses NodeIr.default_child index when no child predicate is true.
pub fn evaluate_tree(tree: &TreeIr, values: &[Value]) -> Value {
    if tree.nodes.is_empty() {
        return Value::Missing;
    }
    let mut idx = 0usize;
    let mut last: Option<Value> = None;
    loop {
        let node = &tree.nodes[idx];
        let cur = score_to_value(&node.score).or(last);
        // Find true child (first where predicate true) — linear scan, early exit
        let mut next: Option<usize> = None;
        for &child_idx in &node.children {
            let child = &tree.nodes[child_idx];
            if eval_predicate(&child.predicate, values) {
                next = Some(child_idx);
                break;
            }
        }
        if let Some(n) = next {
            last = cur;
            idx = n;
            continue;
        } else {
            if !node.children.is_empty() {
                // No true child — handle missingValueStrategy for DefaultChild
                if tree.missing_value_strategy == MissingValueStrategy::DefaultChild {
                    if let Some(dc_idx) = node.default_child {
                        if node.children.contains(&dc_idx) {
                            last = cur;
                            idx = dc_idx;
                            continue;
                        }
                    }
                }
                if tree.missing_value_strategy == MissingValueStrategy::LastPrediction {
                    return cur.unwrap_or(Value::Missing);
                }
                if tree.missing_value_strategy == MissingValueStrategy::NullPrediction
                    || tree.missing_value_strategy == MissingValueStrategy::None
                {
                    match tree.no_true_child_strategy {
                        NoTrueChildStrategy::ReturnLastPrediction => {
                            return cur.unwrap_or(Value::Missing)
                        }
                        NoTrueChildStrategy::ReturnNullPrediction => return Value::Missing,
                    }
                }
                match tree.no_true_child_strategy {
                    NoTrueChildStrategy::ReturnLastPrediction => {
                        return cur.unwrap_or(Value::Missing)
                    }
                    NoTrueChildStrategy::ReturnNullPrediction => return Value::Missing,
                }
            } else {
                return cur.unwrap_or(Value::Missing);
            }
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
                    default_child: None,
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
                    default_child: None,
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
                    default_child: None,
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
