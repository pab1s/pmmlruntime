//! Value conversion helpers — replaces `Session::string_to_value`.
//!
//! Mirrors ONNX `OrtValue` string vs tensor handling: PMML `Value` needs
//! `FieldId` + `DataType`/`OpType` to decide `Continuous` vs `Discrete`.
//! This module is `pub(crate)`; `Session` exposes `field_id`/`symbol_id` for callers.
//!
//! The helper is used by `Session::string_to_value` (CSV / CLI) and by
//! `Session::run_from_strings`. It handles empty / `"Missing"` → `Missing`,
//! numeric `parse::<f64>` → `Continuous` vs categorical intern → `Discrete`,
//! and unknown categorical → `Missing` (so predicates fail safely rather than wrong-match).

use pmml_core::{FieldId, SymbolId, Value};

/// Convert a raw string (e.g. from CSV) to `Value` using `Session`'s caches.
///
/// Returns `Missing` for empty/`"Missing"` or unknown categorical. Tries numeric `parse`
/// first, but if the field's `DataType::String` / `OpType::Categorical` and a `SymbolId`
/// exists for `s`, it prefers `Discrete` (so numeric-looking categoricals like `"0"` don't mis-fire).
///
/// # Parameters
///
/// - `_field_name`: unused in v1 (kept for future diagnostics; field name that owns this value).
/// - `s`: raw string from CSV / user input.
/// - `field_id`: `Some(FieldId)` if field is known from `DataDictionary`, else `None`.
/// - `data_type`: `Some(DataType)` if `field_id` was found in `Ir.data_dictionary`, else `None`.
/// - `op_type`: `Some(OpType)` if `field_id` was found, else `None`.
/// - `symbol_str_to_id`: `String → SymbolId` map from `Session` (cold interning).
///
/// # Returns
///
/// `Value::Continuous(f)` if `s` parses as `f64` and is not categorical-with-symbol,
/// `Discrete(SymbolId)` if `s` is known categorical, else `Missing`.
///
/// # Examples
///
/// ```
/// use pmml_session::input::string_to_value;
/// use pmml_core::{FieldId, field::{DataType, OpType}, Value, SymbolId};
/// use std::collections::HashMap;
/// let mut sym = HashMap::new();
/// sym.insert("setosa".to_string(), SymbolId(1));
/// let v = string_to_value("Species", "setosa", Some(FieldId(0)), Some(DataType::String), Some(OpType::Categorical), &sym);
/// assert!(matches!(v, Value::Discrete(_)));
/// let v2 = string_to_value("x", "3.14", None, None, None, &sym);
/// assert_eq!(v2, Value::Continuous(3.14));
/// assert_eq!(string_to_value("x","", None, None, None, &sym), Value::Missing);
/// ```
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

/// Simple helper that converts `s` via `string_to_value` without field metadata.
///
/// Uses `""` for field name and `None` for `FieldId`/`DataType`/`OpType`, so categorical
/// detection relies only on `symbol_str_to_id`. Handy for tests / `InlineTable` bridging.
///
/// # Parameters
///
/// - `s`: raw string.
/// - `symbol_str_to_id`: map for `Discrete` interning.
///
/// # Returns
///
/// `Value` as per [`string_to_value`].
#[allow(dead_code)]
pub fn value_from_string_simple(
    s: &str,
    symbol_str_to_id: &std::collections::HashMap<String, SymbolId>,
) -> Value {
    string_to_value("", s, None, None, None, symbol_str_to_id)
}
