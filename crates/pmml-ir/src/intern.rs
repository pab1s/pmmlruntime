//! String interning — lasso Rodeo for FieldName and Discrete values.

use lasso::{Rodeo, Spur};
use pmml_core::{FieldId, SymbolId};
use std::collections::HashMap;

/// Central interner. Cold path only (lower).
#[derive(Default)]
pub struct Interner {
    rodeo: Rodeo,
    field_map: HashMap<String, FieldId>,
    symbol_map: HashMap<String, SymbolId>,
    next_field: u32,
    next_symbol: u32,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern field name -> FieldId (stable u32).
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

    /// Intern discrete value string -> SymbolId.
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

    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        self.field_map.get(name).copied()
    }

    pub fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbol_map.get(name).copied()
    }

    /// Resolve SymbolId back to string (for output).
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

    /// Number of fields interned.
    pub fn num_fields(&self) -> usize {
        self.field_map.len()
    }

    /// Get FieldId mapping (field_name -> id)
    pub fn field_map(&self) -> &HashMap<String, FieldId> {
        &self.field_map
    }

    pub fn symbol_map(&self) -> &HashMap<String, SymbolId> {
        &self.symbol_map
    }
}

// Unused Spur helper
#[allow(unused)]
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
