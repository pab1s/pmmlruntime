//! SequenceModel evaluation — ordered itemsets with sequence rules.
//!
//! Implements `SequenceModel` where each transaction is a discrete item (single active field).
//! The evaluator resolves the input `Discrete` symbol to its `Item/@id`, then treats it as a singleton transaction
//! and scans `SequenceRule`s for one whose antecedent `Sequence` contains that item (via its `Itemset` or `SetPredicate`).
//! The first matching rule's consequent `Sequence` first item is returned.
//!
//! `gap` and `Delimiter` handling is simplified to first-set match (gap=true/unknown is considered match).

use crate::base::Value;
use crate::ir::SequenceModelIr;
use std::collections::HashMap;

/// Evaluate a [`SequenceModelIr`] against a dense `values` array.
pub fn evaluate_sequence(model: &SequenceModelIr, values: &[Value]) -> Value {
    if model.mining_schema.active_fields.is_empty() {
        return Value::Missing;
    }
    let fid = model.mining_schema.active_fields[0].as_usize();
    let actual = if fid < values.len() {
        values[fid]
    } else {
        Value::Missing
    };
    if actual.is_missing() {
        return Value::Missing;
    }
    let input_sid = match actual {
        Value::Discrete(sid) => sid,
        Value::Continuous(_) => return Value::Missing,
        Value::Missing => return Value::Missing,
    };

    // Build lookup maps
    let item_value_map: HashMap<&String, crate::base::SymbolId> =
        model.items.iter().map(|it| (&it.id, it.value)).collect();
    let itemset_map: HashMap<&String, &[String]> = model
        .itemsets
        .iter()
        .map(|is| (&is.id, is.item_ids.as_slice()))
        .collect();
    let set_predicate_map: HashMap<&String, &crate::ir::SetPredicateIr> =
        model.set_predicates.iter().map(|sp| (&sp.id, sp)).collect();
    // Sequence map: id -> ordered set ids
    let mut sequence_map: HashMap<&String, Vec<String>> = HashMap::new();
    for seq in &model.sequences {
        let mut ordered = seq.sets.clone();
        for (_, _, set_id) in &seq.follow_sets {
            ordered.push(set_id.clone());
        }
        sequence_map.insert(&seq.id, ordered);
    }

    // Helper to check if a set id (itemset or setpredicate) matches input
    let set_matches_input = |set_id: &String, values: &[Value]| -> bool {
        if let Some(item_ids) = itemset_map.get(set_id) {
            // Find input item id matching input_sid
            let mut input_item_id: Option<&String> = None;
            for it in &model.items {
                if it.value == input_sid {
                    input_item_id = Some(&it.id);
                    break;
                }
            }
            if let Some(iid) = input_item_id {
                return item_ids.contains(iid);
            }
            return false;
        }
        if let Some(pred) = set_predicate_map.get(set_id) {
            let p_idx = pred.field.as_usize();
            let pred_val = if p_idx < values.len() {
                values[p_idx]
            } else {
                Value::Missing
            };
            if pred_val.is_missing() {
                return false;
            }
            if let Value::Discrete(sid) = pred_val {
                return pred.values.contains(&sid);
            }
            return false;
        }
        false
    };

    // Helper to check if a sequence matches input (any set in sequence matches)
    let sequence_matches = |seq_id: &String, values: &[Value]| -> bool {
        if let Some(ordered) = sequence_map.get(seq_id) {
            for set_id in ordered {
                if set_matches_input(set_id, values) {
                    return true;
                }
            }
        }
        false
    };

    // Find first rule where antecedent matches
    for rule in &model.sequence_rules {
        if sequence_matches(&rule.antecedent, values) {
            // Consequent sequence's first set's first item
            if let Some(ordered) = sequence_map.get(&rule.consequent) {
                if let Some(first_set_id) = ordered.first() {
                    // Try itemset
                    if let Some(item_ids) = itemset_map.get(first_set_id) {
                        if let Some(first_item_id) = item_ids.first() {
                            if let Some(&val) = item_value_map.get(first_item_id) {
                                return Value::Discrete(val);
                            }
                        }
                    } else if let Some(pred) = set_predicate_map.get(first_set_id) {
                        // For set predicate consequent, return its first predicate value
                        if let Some(&first_val) = pred.values.first() {
                            return Value::Discrete(first_val);
                        }
                    }
                }
            }
        }
    }

    // Fallback: if no rule matches, try to return first sequence's first item as direct prediction (like association)
    // Check if any sequence's first set contains input and return its next set's first item?
    // For now return Missing if no rule.
    Value::Missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, SymbolId, Value};
    use crate::ir::*;

    fn make_model() -> SequenceModelIr {
        let f = FieldId(0);
        let sid_a = SymbolId(1);
        let sid_b = SymbolId(2);
        let sid_c = SymbolId(3);
        SequenceModelIr {
            function_name: "sequences".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            constraints: None,
            items: vec![
                ItemIr {
                    id: "i0".into(),
                    value: sid_a,
                },
                ItemIr {
                    id: "i1".into(),
                    value: sid_b,
                },
                ItemIr {
                    id: "i2".into(),
                    value: sid_c,
                },
            ],
            itemsets: vec![
                ItemsetIr {
                    id: "is0".into(),
                    item_ids: vec!["i0".into()],
                },
                ItemsetIr {
                    id: "is1".into(),
                    item_ids: vec!["i1".into()],
                },
                ItemsetIr {
                    id: "is2".into(),
                    item_ids: vec!["i2".into()],
                },
            ],
            set_predicates: vec![],
            sequences: vec![
                SequenceIr {
                    id: "s0".into(),
                    number_of_sets: Some(1),
                    occurrence: Some(5),
                    support: Some(0.2),
                    sets: vec!["is0".into()],
                    follow_sets: vec![],
                    time: None,
                },
                SequenceIr {
                    id: "s1".into(),
                    number_of_sets: Some(1),
                    occurrence: Some(5),
                    support: Some(0.2),
                    sets: vec!["is1".into()],
                    follow_sets: vec![],
                    time: None,
                },
                SequenceIr {
                    id: "s2".into(),
                    number_of_sets: Some(1),
                    occurrence: Some(5),
                    support: Some(0.2),
                    sets: vec!["is2".into()],
                    follow_sets: vec![],
                    time: None,
                },
            ],
            sequence_rules: vec![SequenceRuleIr {
                id: "r0".into(),
                number_of_sets: 2,
                occurrence: 5,
                support: 0.2,
                confidence: 0.9,
                lift: None,
                antecedent: "s0".into(),
                consequent: "s1".into(),
                delimiter: DelimiterIr {
                    delimiter: "acrossTimeWindows".into(),
                    gap: "unknown".into(),
                },
                time_between: None,
                time_total: None,
            }],
        }
    }

    #[test]
    fn sequence_rule_fires() {
        let model = make_model();
        let vals = vec![Value::Discrete(SymbolId(1))]; // a
        assert_eq!(
            evaluate_sequence(&model, &vals),
            Value::Discrete(SymbolId(2))
        );
        let vals2 = vec![Value::Discrete(SymbolId(3))];
        assert_eq!(evaluate_sequence(&model, &vals2), Value::Missing);
    }
}
