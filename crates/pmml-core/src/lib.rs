//! pmml-core — zero-cost types, arena, errors. No XML, no IR.

pub mod arena;
pub mod error;
pub mod field;
pub mod value;

pub use arena::with_arena;
pub use error::{PmmlError, Result};
pub use field::{DataType, MiningFunction, OpType, ResultFeature};
pub use value::{FieldId, SymbolId, Value};

pub fn placeholder() {}
