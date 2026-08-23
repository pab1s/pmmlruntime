//! Value types — hot path, zero-alloc, `Copy` friendly.

/// Interned discrete value identifier. `u32::MAX` reserved for missing sentinel internally, not used in Value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

/// Field identifier after interning `FIELD-NAME`. Index into `values[field_id]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldId(pub u32);

impl FieldId {
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// PMML field value in the evaluator (hot path).
/// `Missing` is an explicit variant — not `Option<Value>` — to avoid double wrapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    /// Continuous numeric (f64 canonical — PMML `double`/`float`/`integer` all coerce to f64 here).
    Continuous(f64),
    /// Discrete categorical/ordinal interned.
    Discrete(SymbolId),
    /// Missing / invalid after `MiningSchema` handling.
    Missing,
}

impl Value {
    pub fn is_missing(self) -> bool {
        matches!(self, Value::Missing)
    }

    pub fn as_f64(self) -> Option<f64> {
        match self {
            Value::Continuous(v) => Some(v),
            _ => None,
        }
    }

    pub fn approx_eq(self, other: Self, eps: f64) -> bool {
        match (self, other) {
            (Value::Missing, Value::Missing) => true,
            (Value::Continuous(a), Value::Continuous(b)) => (a - b).abs() <= eps || (a.is_nan() && b.is_nan()),
            (Value::Discrete(a), Value::Discrete(b)) => a == b,
            _ => false,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Continuous(v) => write!(f, "{v}"),
            Value::Discrete(SymbolId(id)) => write!(f, "Discrete({id})"),
            Value::Missing => write!(f, "Missing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_is_missing() {
        assert!(Value::Missing.is_missing());
        assert!(!Value::Continuous(3.14).is_missing());
    }

    #[test]
    fn approx_eq_eps() {
        assert!(Value::Continuous(1.0).approx_eq(Value::Continuous(1.0 + 1e-10), 1e-9));
        assert!(!Value::Continuous(1.0).approx_eq(Value::Continuous(1.1), 1e-9));
        assert!(Value::Missing.approx_eq(Value::Missing, 1e-9));
        assert!(!Value::Missing.approx_eq(Value::Continuous(0.0), 1e-9));
    }
}
