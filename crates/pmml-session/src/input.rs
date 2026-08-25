//! Value conversion helpers — replaces `Session::string_to_value`.
//! Mirrors ONNX `OrtValue` string vs tensor handling: PMML `Value` needs
//! `FieldId` + `DataType`/`OpType` to decide `Continuous` vs `Discrete`.
//! This module is `pub(crate)`; `Session` exposes `field_id`/`symbol_id` for callers.

use pmml_core::{FieldId, SymbolId, Value};

/// Convert a raw string (e.g. from CSV) to `Value` using `Session`'s caches.
/// Returns `Missing` for empty/"Missing" or unknown categorical.
pub fn string_to_value(
    _field_name: &str,
    s: &str,
    field_id: Option<FieldId>,
    data_type: Option<pmml_core::field::DataType>,
    op_type: Option<pmml_core::field::OpType>,
    symbol_str_to_id: &std::collections::HashMap<String, SymbolId>,
) -> Value {
    if s.is_empty() || s.eq_ignore_ascii_case("missing") {
        return Value::Missing;
    }
    // Try numeric first, but if field is categorical/string, prefer SymbolId
    if let Ok(f) = s.parse::<f64>() {
        if let Some(fid) = field_id {
            // Check if field expects string/categorical → intern as Discrete
            let _ = fid; // keep param for future DataDictionary lookup
            if let (Some(dt), Some(op)) = (data_type, op_type) {
                if dt == pmml_core::field::DataType::String
                    || op == pmml_core::field::OpType::Categorical
                {
                    if let Some(sid) = symbol_str_to_id.get(s) {
                        return Value::Discrete(*sid);
                    }
                }
            } else if let Some(sid) = symbol_str_to_id.get(s) {
                // Fallback: if string looks numeric but we have a symbol for it, use Discrete
                // This handles categorical fields that look numeric (e.g. "0")
                // Prefer Discrete if symbol exists and op_type unknown
                // Heuristic: if s is integer-like and symbol exists, use Discrete
                if s.chars()
                    .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
                {
                    // still check
                }
                return Value::Discrete(*sid);
            }
        }
        return Value::Continuous(f);
    }
    if let Some(sid) = symbol_str_to_id.get(s) {
        Value::Discrete(*sid)
    } else {
        // Unknown categorical — treat as Missing so predicate fails rather than wrong match
        Value::Missing
    }
}

#[allow(dead_code)]
pub fn value_from_string_simple(
    s: &str,
    symbol_str_to_id: &std::collections::HashMap<String, SymbolId>,
) -> Value {
    string_to_value("", s, None, None, None, symbol_str_to_id)
}
