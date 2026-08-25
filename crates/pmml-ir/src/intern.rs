//! String interning — stable `FieldId` / `SymbolId` assignment for the cold path.
//!
//! Lowering interns every `DataField/@name`, `MiningField/@name`, and discrete
//! value once via [`Interner`]. The interner wraps a `lasso::Rodeo` for
//! debugging reverse lookup but authoritative ids are the monotonic `u32`
//! counters (`FieldId.0`, `SymbolId.0`). Hot-path scoring never touches the
//! interner; it uses the `Ir.field_names` / `Ir.symbol_names` snapshots.

use lasso::{Rodeo, Spur};
use pmml_core::{FieldId, SymbolId};
use std::collections::HashMap;

/// Central string interner for the cold lowering path.
///
/// Not `Send` on purpose in typical use — held mutably only inside
/// [`crate::lower::lower`] on a single thread. Wraps a `lasso::Rodeo` for
/// `get_or_intern` debugging plus two `HashMap`s that provide stable
/// sequential [`FieldId`] / [`SymbolId`] (`u32`) allocation.
///
/// # Performance
///
/// `Rodeo::get_or_intern` is called only for the first occurrence of each
/// distinct string; subsequent calls hit the `HashMap` fast path.
///
/// # Thread safety
///
/// `Interner` itself is `!Sync` in practice because `Rodeo` and the `HashMap`s
/// are mutated without locking. It is constructed per `lower` invocation and
/// never shared across threads. `Ir` snapshots (`field_names`, `symbol_names`)
/// are `Send + Sync`.
///
/// # Examples
///
/// ```
/// use pmml_ir::Interner;
/// let mut inter = Interner::new();
/// let a = inter.intern_field("age");
/// let b = inter.intern_field("age");
/// let c = inter.intern_field("income");
/// assert_eq!(a, b);
/// assert_ne!(a, c);
/// ```
#[derive(Default)]
pub struct Interner {
    rodeo: Rodeo,
    field_map: HashMap<String, FieldId>,
    symbol_map: HashMap<String, SymbolId>,
    next_field: u32,
    next_symbol: u32,
}

impl Interner {
    /// Creates an empty interner with sequential ids starting at 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmml_ir::Interner;
    /// let inter = Interner::new();
    /// assert_eq!(inter.num_fields(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns `name` as a field and returns its stable [`FieldId`].
    ///
    /// When `name` was already interned, returns the existing id. Otherwise
    /// allocates the next `FieldId(next_field)` and stores `name` in both the
    /// `field_map` and the underlying [`Rodeo`] for reverse lookup.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmml_ir::Interner;
    /// let mut inter = Interner::new();
    /// let id1 = inter.intern_field("x");
    /// let id2 = inter.intern_field("x");
    /// assert_eq!(id1, id2);
    /// assert_eq!(inter.num_fields(), 1);
    /// ```
    pub fn intern_field(&mut self, name: &str) -> FieldId {
        if let Some(&id) = self.field_map.get(name) {
            return id;
        }
        let id = FieldId(self.next_field);
        self.next_field += 1;
        self.field_map.insert(name.to_string(), id);
        // also store in rodeo for debug / reverse lookup
        self.rodeo.get_or_intern(name);
        id
    }

    /// Interns a discrete value string and returns its stable [`SymbolId`].
    ///
    /// Stable (returns the same id for repeated values) and monotonic.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmml_ir::Interner;
    /// let mut inter = Interner::new();
    /// let s1 = inter.intern_symbol("setosa");
    /// let s2 = inter.intern_symbol("setosa");
    /// let s3 = inter.intern_symbol("virginica");
    /// assert_eq!(s1, s2);
    /// assert_ne!(s1, s3);
    /// ```
    pub fn intern_symbol(&mut self, val: &str) -> SymbolId {
        if let Some(&id) = self.symbol_map.get(val) {
            return id;
        }
        let id = SymbolId(self.next_symbol);
        self.next_symbol += 1;
        self.symbol_map.insert(val.to_string(), id);
        self.rodeo.get_or_intern(val);
        id
    }

    /// Looks up the [`FieldId`] for a field `name`, if previously interned.
    ///
    /// Returns `None` when `name` has never been interned.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmml_ir::Interner;
    /// let mut inter = Interner::new();
    /// inter.intern_field("age");
    /// assert!(inter.field_id("age").is_some());
    /// assert!(inter.field_id("unknown").is_none());
    /// ```
    #[must_use]
    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        self.field_map.get(name).copied()
    }

    /// Looks up the [`SymbolId`] for a discrete value `name`, if previously interned.
    ///
    /// Returns `None` when the value has never been interned.
    #[must_use]
    pub fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbol_map.get(name).copied()
    }

    /// Resolves a [`SymbolId`] back to its interned string slice.
    ///
    /// Linear scan over `symbol_map`; `None` when `id` was not interned (for
    /// example, a synthetically constructed `SymbolId`). No allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmml_ir::Interner;
    /// let mut inter = Interner::new();
    /// let id = inter.intern_symbol("hello");
    /// assert_eq!(inter.resolve_symbol(id), Some("hello"));
    /// ```
    #[must_use]
    pub fn resolve_symbol(&self, id: SymbolId) -> Option<&str> {
        // lasso stores via Spur; we can use rodeo.resolve
        // But we used separate map, need to invert. For now, brute search.
        for (k, &v) in &self.symbol_map {
            if v == id {
                return Some(k.as_str());
            }
        }
        None
    }

    /// Number of distinct field names interned.
    ///
    /// Also equals the next [`FieldId`] offset after lowering.
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.field_map.len()
    }

    /// Borrowed view of the `field_name → FieldId` map.
    ///
    /// Used by lowering to build [`crate::ir::Ir::field_names`] and for the
    /// hot-path `pmml-session` field layout.
    #[must_use]
    pub fn field_map(&self) -> &HashMap<String, FieldId> {
        &self.field_map
    }

    /// Borrowed view of the `value_string → SymbolId` map.
    #[must_use]
    pub fn symbol_map(&self) -> &HashMap<String, SymbolId> {
        &self.symbol_map
    }
}

/// Unused `Spur → SymbolId` helper, retained for `lasso` interop tests.
///
/// Reserved; not part of the public cold-path API.
#[allow(unused)]
#[doc(hidden)]
pub fn spur_to_symbol(_spur: Spur) -> SymbolId {
    SymbolId(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_field_stable() {
        let mut inter = Interner::new();
        let a = inter.intern_field("age");
        let b = inter.intern_field("age");
        let c = inter.intern_field("income");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn intern_symbol() {
        let mut inter = Interner::new();
        let s1 = inter.intern_symbol("setosa");
        let s2 = inter.intern_symbol("virginica");
        assert_ne!(s1, s2);
        assert_eq!(s1, inter.intern_symbol("setosa"));
    }
}
