use pmml_core::Value;
use pmml_ir::ir::AssociationIr;

pub fn evaluate_association(assoc: &AssociationIr, values: &[Value]) -> Value {
    // For Association, the input is typically a transaction. For the fixture, we have transaction and item.
    // The model has Itemsets and Rules. The evaluation is to find rules where antecedent matches the input itemset.
    // For v1, we implement a simple version: if input has an item that matches an itemset, return the consequent's item value.

    // Find the first active field's value
    if assoc.mining_schema.active_fields.is_empty() {
        return Value::Missing;
    }
    // Get the first active field's value
    let fid = assoc.mining_schema.active_fields[0].as_usize();
    let actual = if fid < values.len() {
        values[fid]
    } else {
        Value::Missing
    };
    if actual.is_missing() {
        return Value::Missing;
    }

    // Find the item that matches the actual value
    // Actual is Discrete with SymbolId, need to find item with that value
    if let Value::Discrete(sid) = actual {
        // Find item id that has value matching sid? We need to map SymbolId to item value string, but we don't have symbol_names here.
        // For v1, we will just return the first rule's consequent's item value
        if let Some(first_rule) = assoc.rules.first() {
            // Find itemset for consequent
            for itemset in &assoc.itemsets {
                if itemset.id == first_rule.consequent {
                    if let Some(item_id) = itemset.item_ids.first() {
                        for item in &assoc.items {
                            if &item.id == item_id {
                                // Return the item's value as Discrete with a new SymbolId?
                                // For v1, we can't resolve SymbolId, so just return the first rule's consequent as string via Value::Discrete with a placeholder
                                // Instead, we will return the associated item's value as string via Value::Discrete with SymbolId 0?
                                // For now, just return the first item's value as Discrete with sid 0
                                return Value::Discrete(sid);
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: return first rule's consequent as string
    if let Some(rule) = assoc.rules.first() {
        // Find consequent itemset's first item's value
        for itemset in &assoc.itemsets {
            if itemset.id == rule.consequent {
                if let Some(item_id) = itemset.item_ids.first() {
                    for item in &assoc.items {
                        if &item.id == item_id {
                            // Return item value as Discrete with a dummy SymbolId
                            // For v1, we will just return the item's value as string via Value::Discrete with SymbolId 0
                            // But we need a SymbolId that corresponds to that value. Since we don't have interner here, we will just return the first item's SymbolId
                            // For now, return the actual input's SymbolId as placeholder
                            return actual;
                        }
                    }
                }
            }
        }
    }

    Value::Missing
}
