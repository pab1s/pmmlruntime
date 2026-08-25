//! AssociationModel evaluation — antecedent → consequent rule firing.
//!
//! Implements `AssociationModel` where each transaction is a single active-field
//! discrete item (e.g., `"milk"`). The evaluator:
//!
//! 1. Resolves the input `Discrete` symbol to its `Item/@id` via `items`.
//! 2. Treats the single item as an antecedent `Itemset` and scans `AssociationRule`s
//!    for one whose `antecedent` `Itemset` contains that `Item` id.
//! 3. Returns the first item of the `consequent` `Itemset`'s value.
//!
//! The PMML `InputTable` / `JoinTable` host factor for transactional inputs is not yet
//! implemented; the v1 path handles single-item transactions as used by fixtures.
//!
//! # What belongs here
//!
//! - [`evaluate_association`] — the single public entry point.
//!
//! # Performance
//!
//! `O(items + itemsets + rules * antecedent_size)`; linear scan, no heap beyond two small `HashMap`s.
//! Typically `rules < 100`.

use crate::base::Value;
use crate::ir::AssociationIr;

/// Evaluate an [`AssociationIr`] against a dense `values` array.
///
/// Single-item transactional path: `mining_schema.active_fields[0]` holds the input item
/// as `Discrete`. The function reverse-looks up its `Item.id`, then finds the first rule
/// whose antecedent itemset contains that id and returns the consequent's first item value.
/// When the input is `Missing`/`Continuous`, or no item/rule matches, `Missing` is returned.
///
/// # Parameters
///
/// - `assoc`: Lowered association model (`AssociationIr`) with `items`, `itemsets`, `rules`, `mining_schema`.
/// - `values`: Dense `&[Value]` indexed by [`FieldId`](crate::base::FieldId). Only `active_fields[0]` is read; out-of-bounds → `Missing`.
///
/// # Returns
///
/// `Discrete(consequent_value)` when a rule fires, otherwise `Missing`.
///
/// # Panics
///
/// Never panics. All `FieldId` indexing is bounds-checked.
///
/// # Performance
///
/// `O(items + itemsets + rules * antecedent_size)` scans; small constants for fixture sizes.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, SymbolId, Value};
/// use pmmlruntime::ir::*;
/// use pmmlruntime::engine::models::evaluate_association;
///
/// let fid = FieldId(0);
/// let sid_milk = SymbolId(1);
/// let sid_bread = SymbolId(2);
/// let model = AssociationIr {
///     function_name: "associationRules".into(),
///     mining_schema: MiningSchemaIr { active_fields: vec![fid], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     output: vec![],
///     items: vec![ItemIr { id: "1".into(), value: sid_milk }, ItemIr { id: "2".into(), value: sid_bread }],
///     itemsets: vec![ItemsetIr { id: "a".into(), item_ids: vec!["1".into()] }, ItemsetIr { id: "c".into(), item_ids: vec!["2".into()] }],
///     rules: vec![AssociationRuleIr { antecedent: "a".into(), consequent: "c".into(), support: 0.1, confidence: 0.9, lift: 1.2 }],
/// };
/// let pred = evaluate_association(&model, &[Value::Discrete(sid_milk)]);
/// assert_eq!(pred, Value::Discrete(sid_bread));
/// assert_eq!(evaluate_association(&model, &[Value::Discrete(SymbolId(99))]), Value::Missing);
/// ```
pub fn evaluate_association(assoc: &AssociationIr, values: &[Value]) -> Value {
    // Association evaluation: input transaction (active field) contains items; find matching rules.
    // Full InputTable handling would require Host table join, but v1 handles simple discrete item input.
    // We also support transactional field where value is Discrete representing a single item (e.g., "milk")
    // and check which rules have antecedent containing that item.
    if assoc.mining_schema.active_fields.is_empty() {
        return Value::Missing;
    }
    let fid = assoc.mining_schema.active_fields[0].as_usize();
    let actual = if fid < values.len() {
        values[fid]
    } else {
        Value::Missing
    };
    if actual.is_missing() {
        return Value::Missing;
    }

    // Extract input SymbolId (discrete item). For continuous fallback, try to use as is.
    let input_sid = match actual {
        Value::Discrete(sid) => sid,
        Value::Continuous(_) => {
            // For transactional numeric inputs, treat as missing for categorical association
            return Value::Missing;
        }
        Value::Missing => return Value::Missing,
    };

    // Build map item_id -> value SymbolId for quick lookup
    use std::collections::HashMap;
    let item_value_map: HashMap<&String, crate::base::SymbolId> =
        assoc.items.iter().map(|it| (&it.id, it.value)).collect();
    // Find which item id corresponds to input_sid (reverse lookup)
    let mut input_item_id: Option<&String> = None;
    for it in &assoc.items {
        if it.value == input_sid {
            input_item_id = Some(&it.id);
            break;
        }
    }
    let input_item_id = match input_item_id {
        Some(id) => id,
        None => {
            // Input value not found among items — no matching rule
            return Value::Missing;
        }
    };

    // Build map itemset_id -> item_ids set
    let itemset_map: HashMap<&String, &[String]> = assoc
        .itemsets
        .iter()
        .map(|is| (&is.id, is.item_ids.as_slice()))
        .collect();

    // Find rules where antecedent itemset contains input item
    for rule in &assoc.rules {
        if let Some(ante_items) = itemset_map.get(&rule.antecedent) {
            if ante_items.contains(input_item_id) {
                // Rule antecedent matches input; return consequent's first item's value
                if let Some(cons_items) = itemset_map.get(&rule.consequent) {
                    if let Some(first_con_id) = cons_items.first() {
                        if let Some(&val) = item_value_map.get(first_con_id) {
                            return Value::Discrete(val);
                        }
                    }
                }
            }
        }
    }

    // Also consider rules where antecedent is single item equal to input itemset (fallback original logic)
    // If no rule matched via containment, fallback to first rule's consequent if antecedent equals input item id directly
    for rule in &assoc.rules {
        if &rule.antecedent == input_item_id {
            if let Some(cons_items) = itemset_map.get(&rule.consequent) {
                if let Some(first_con_id) = cons_items.first() {
                    if let Some(&val) = item_value_map.get(first_con_id) {
                        return Value::Discrete(val);
                    }
                }
            }
        }
    }

    // No matching rule
    Value::Missing
}
