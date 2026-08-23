//! MapValues — inline table lookup (stub v1).

use pmml_core::{SymbolId, Value};

pub fn eval_mapvalues(_input: Value, _table: &[(SymbolId, SymbolId)], _default: Option<SymbolId>) -> Value {
    // v1: not used for Tree fixtures; return input
    _input
}
