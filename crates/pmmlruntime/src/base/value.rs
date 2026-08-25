//! Value types — hot path, zero-alloc, `Copy`-friendly.
//!
//! `pmml-session` materializes `&mut [Value]` indexed by [`FieldId`] for every row.
//! For `<=64` fields the buffer is stack-allocated (L1-hot); larger models use a
//! `thread_local!` heap buffer reused across rows. See `pmml-session::session::with_value_buffer`.
//!
//! [`FieldId`] is a `u32` newtype so `values[field.as_usize()]` is a single bounds check.
//! [`SymbolId`] is the interned discrete value; `pmml-session` holds a dense
//! `Vec<String>` for `SymbolId → String` to avoid `HashMap` in the hot path.

/// Interned discrete value identifier.
///
/// Produced by `pmml-ir::Interner::intern_symbol` (cold) and carried as [`Value::Discrete`].
/// `u32::MAX` is reserved for a missing sentinel internally and never appears in [`Value`].
/// Dense `Vec<String>` in `pmml-session` maps `SymbolId.0 as usize` → display string.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::SymbolId;
/// let a = SymbolId(0);
/// let b = SymbolId(1);
/// assert_ne!(a, b);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

/// Field identifier after interning `FIELD-NAME`.
///
/// Stable `u32` assigned by `pmml-ir::Interner::intern_field` (cold). Hot path
/// indexes `values[field_id.as_usize()]` — a single array lookup, no `HashMap`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::FieldId;
/// let id = FieldId(2);
/// assert_eq!(id.as_usize(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldId(pub u32);

impl FieldId {
    /// Convert to `usize` for array indexing. `const` so it can be used in const contexts.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::FieldId;
    /// let slot = FieldId(3).as_usize();
    /// assert_eq!(slot, 3);
    /// ```
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// PMML field value in the evaluator (hot path).
///
/// `Missing` is an explicit variant, not `Option<Value>`, to avoid double wrapping
/// and to keep `Value` `Copy` with a predictable discriminant. `pmml-evaluator`
/// treats `Discrete` as an interned [`SymbolId`] and `Continuous` as canonical `f64`
/// (PMML `double`/`float`/`integer` all coerce to `f64` here).
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{Value, SymbolId};
/// let a = Value::Continuous(1.0);
/// let b = Value::Discrete(SymbolId(2));
/// assert!(!a.is_missing());
/// assert_eq!(a.as_f64(), Some(1.0));
/// assert_eq!(b.as_f64(), None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    /// Continuous numeric (`f64` canonical — PMML `double`/`float`/`integer` all coerce here).
    Continuous(f64),
    /// Discrete categorical/ordinal interned as [`SymbolId`].
    Discrete(SymbolId),
    /// Missing / invalid after `MiningSchema` handling (outlier, invalid, or absent).
    Missing,
}

impl Value {
    /// Returns `true` iff this is [`Value::Missing`].
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::Value;
    /// assert!(Value::Missing.is_missing());
    /// assert!(!Value::Continuous(1.0).is_missing());
    /// ```
    #[must_use]
    pub fn is_missing(self) -> bool {
        matches!(self, Value::Missing)
    }

    /// Returns the inner `f64` if [`Value::Continuous`], else `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::Value;
    /// assert_eq!(Value::Continuous(2.5).as_f64(), Some(2.5));
    /// assert_eq!(Value::Missing.as_f64(), None);
    /// ```
    #[must_use]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Value::Continuous(v) => Some(v),
            _ => None,
        }
    }

    /// Approximate equality with tolerance `eps`, plus `NaN == NaN` and `Missing == Missing`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::Value;
    /// assert!(Value::Continuous(1.0).approx_eq(Value::Continuous(1.0 + 1e-10), 1e-9));
    /// assert!(!Value::Continuous(1.0).approx_eq(Value::Continuous(1.1), 1e-9));
    /// ```
    #[must_use]
    pub fn approx_eq(self, other: Self, eps: f64) -> bool {
        match (self, other) {
            (Value::Missing, Value::Missing) => true,
            (Value::Continuous(a), Value::Continuous(b)) => {
                (a - b).abs() <= eps || (a.is_nan() && b.is_nan())
            }
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
#[allow(clippy::pedantic)]
mod tests {
    use super::*;

    #[test]
    fn missing_is_missing() {
        assert!(Value::Missing.is_missing());
        assert!(!Value::Continuous(2.71).is_missing());
    }

    #[test]
    fn approx_eq_eps() {
        assert!(Value::Continuous(1.0).approx_eq(Value::Continuous(1.0 + 1e-10), 1e-9));
        assert!(!Value::Continuous(1.0).approx_eq(Value::Continuous(1.1), 1e-9));
        assert!(Value::Missing.approx_eq(Value::Missing, 1e-9));
        assert!(!Value::Missing.approx_eq(Value::Continuous(0.0), 1e-9));
    }
}
