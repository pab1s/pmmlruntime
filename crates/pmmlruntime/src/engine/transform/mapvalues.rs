//! MapValues table lookup — discrete input → discrete output.
//!
//! Implements PMML `MapValues` (`InlineTable` / `TextIndex`) for single- and
//! multi-input tables. The evaluator variant here handles the single-column case
//! used by `DerivedField` expressions; the multi-column variant is evaluated
//! via [`crate::ir::Op::MapValuesMulti`] in the VM.
//!
//! # What belongs here
//!
//! - [`eval_mapvalues`] — linear search over `&[(SymbolId, SymbolId)]`.
//!
//! # Performance
//!
//! `O(table.len())` linear scan. Tables are small (typical <32 entries).
//! For larger tables a `HashMap` in `lower` would be generated, but the VM
//! retains the `Vec` form to avoid hashing on the hot path for small tables.

use crate::base::{SymbolId, Value};

/// Look up a discrete input in a `MapValues` table.
///
/// Mirrors PMML `MapValues` for a single `FieldColumnPair` plus
/// `InlineTable` / `TextIndex` derived fields. The table is an ordered
/// `Vec<(input SymbolId, output SymbolId)>` produced during lowering; search
/// is linear and returns the first match.
///
/// # Parameters
///
/// - `input`: Value to map. Only [`Value::Discrete`] is looked up; `Missing`
///   or `Continuous` immediately returns `default` (or `Missing` when `None`).
/// - `table`: Sorted map from input symbol to output symbol.
/// - `default`: Value when no key matches or when `input` is `Missing`/`Continuous`. `None` yields `Missing`.
///
/// # Returns
///
/// `Discrete(output)` when found, otherwise `Discrete(default)` when `default.is_some()`,
/// otherwise [`Value::Missing`].
///
/// # Panics
///
/// Never panics.
///
/// # Performance
///
/// `O(table.len())` with early exit on match. No allocation.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{SymbolId, Value};
/// use pmmlruntime::engine::transform::mapvalues::eval_mapvalues;
///
/// let table = vec![(SymbolId(1), SymbolId(10)), (SymbolId(2), SymbolId(20))];
/// assert_eq!(eval_mapvalues(Value::Discrete(SymbolId(1)), &table, None), Value::Discrete(SymbolId(10)));
/// assert_eq!(eval_mapvalues(Value::Discrete(SymbolId(3)), &table, Some(SymbolId(99))), Value::Discrete(SymbolId(99)));
/// assert_eq!(eval_mapvalues(Value::Discrete(SymbolId(3)), &table, None), Value::Missing);
/// assert_eq!(eval_mapvalues(Value::Missing, &table, Some(SymbolId(5))), Value::Discrete(SymbolId(5)));
/// ```
pub fn eval_mapvalues(
    input: Value,
    table: &[(SymbolId, SymbolId)],
    default: Option<SymbolId>,
) -> Value {
    let sid = match input {
        Value::Discrete(s) => s,
        Value::Missing => return default.map(Value::Discrete).unwrap_or(Value::Missing),
        Value::Continuous(_) => {
            // For continuous input, try to match via string? For now return default/missing
            return default.map(Value::Discrete).unwrap_or(Value::Missing);
        }
    };
    for (k, v) in table {
        if *k == sid {
            return Value::Discrete(*v);
        }
    }
    default.map(Value::Discrete).unwrap_or(Value::Missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapvalues_found() {
        let table = vec![(SymbolId(1), SymbolId(10)), (SymbolId(2), SymbolId(20))];
        assert_eq!(
            eval_mapvalues(Value::Discrete(SymbolId(1)), &table, None),
            Value::Discrete(SymbolId(10))
        );
        assert_eq!(
            eval_mapvalues(Value::Discrete(SymbolId(3)), &table, Some(SymbolId(99))),
            Value::Discrete(SymbolId(99))
        );
        assert_eq!(
            eval_mapvalues(Value::Discrete(SymbolId(3)), &table, None),
            Value::Missing
        );
        assert_eq!(
            eval_mapvalues(Value::Missing, &table, Some(SymbolId(5))),
            Value::Discrete(SymbolId(5))
        );
    }
}
