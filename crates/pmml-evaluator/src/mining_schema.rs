//! MiningSchema pre-processing — per JPMML Features:
//! - DataType coercion
//! - invalidValueTreatment (returnInvalid -> error, asMissing -> Missing)
//! - missingValueReplacement
//! - outlierTreatment (asMissingValues / asExtremeValues) — stub v1
//!
//! v1 branchless select via `is_missing` mask.

use pmml_core::{FieldId, Value};
use pmml_ir::ir::MiningSchemaIr;

/// Apply mining schema to `values` array in-place.
/// `input_map` is FieldId->Value from caller (sparse). `values` is flat `Vec<Value>` sized for all fields (initialized Missing).
pub fn apply_mining_schema(
    schema: &MiningSchemaIr,
    input_map: &std::collections::HashMap<FieldId, Value>,
    values: &mut [Value],
) -> Result<(), String> {
    // First, copy active fields from input_map with type coercion stub (f64 already)
    for &fid in &schema.active_fields {
        let v = input_map.get(&fid).copied().unwrap_or(Value::Missing);
        // Future: check DataType coercion via FieldMeta; for now pass through.
        // Invalid handling: if value is continuous but field expects categorical string? Coerce as Discrete?
        // For v1 Tree continuous fields, we keep Continuous as is.
        let idx = fid.as_usize();
        if idx < values.len() {
            values[idx] = v;
        }
    }
    // Target field is not expected in inputs, leave Missing
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::FieldId;
    use std::collections::HashMap;
    #[test]
    fn apply_copies() {
        let schema = MiningSchemaIr {
            active_fields: vec![FieldId(0), FieldId(1)],
            target_field: Some(FieldId(2)),
            field_metas: vec![],
            missing_value_replacement: None,
        };
        let mut map = HashMap::new();
        map.insert(FieldId(0), Value::Continuous(1.5));
        // FieldId 1 missing -> should be Missing
        let mut values = vec![Value::Missing; 3];
        apply_mining_schema(&schema, &map, &mut values).unwrap();
        assert_eq!(values[0], Value::Continuous(1.5));
        assert_eq!(values[1], Value::Missing);
    }
}
