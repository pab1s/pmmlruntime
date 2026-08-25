//! MapValues — inline table lookup (full per plan D2).

use pmml_core::{SymbolId, Value};

/// MapValues lookup: `input` Discrete/Categorical is looked up in `table` (input->output).
/// If found, returns corresponding output SymbolId as Discrete; else returns `default` if Some, else input or Missing.
/// Mirrors JPMML `MapValues` handling for InlineTable/TextIndex derived fields.
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
        assert_eq!(eval_mapvalues(Value::Discrete(SymbolId(1)), &table, None), Value::Discrete(SymbolId(10)));
        assert_eq!(eval_mapvalues(Value::Discrete(SymbolId(3)), &table, Some(SymbolId(99))), Value::Discrete(SymbolId(99)));
        assert_eq!(eval_mapvalues(Value::Discrete(SymbolId(3)), &table, None), Value::Missing);
        assert_eq!(eval_mapvalues(Value::Missing, &table, Some(SymbolId(5))), Value::Discrete(SymbolId(5)));
    }
}
