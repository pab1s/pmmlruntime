use pmml_core::Value;
use pmml_ir::ir::AssociationIr;

pub fn evaluate_association(assoc: &AssociationIr, values: &[Value]) -> Value {
    // Association evaluation: input transaction (active field) contains items; find matching rules.
    // Full InputTable handling would require Host table join, but v1 handles simple discrete item input.
    // We also support transactional field where value is Discrete representing a single item (e.g., "milk")
    // and check which rules have antecedent containing that item.
    if assoc.mining_schema.active_fields.is_empty() {
        return Value::Missing;
    }
    let fid = assoc.mining_schema.active_fields[0].as_usize();
    let actual = if fid < values.len() { values[fid] } else { Value::Missing };
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
    let item_value_map: HashMap<&String, pmml_core::SymbolId> = assoc
        .items
        .iter()
        .map(|it| (&it.id, it.value))
        .collect();
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
