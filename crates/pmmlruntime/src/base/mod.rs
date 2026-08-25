//! Core — zero-cost foundation: arena, field types, values, errors.
//!
//! See original `pmml-core` crate for details.
//! This module is `pub` as `crate::base` (renamed from `core` to avoid `::core` shadowing).

pub mod arena;
pub mod error;
pub mod field;
pub mod value;

pub use arena::with_arena;
pub use error::{PmmlError, Result};
pub use field::{DataType, MiningFunction, OpType, ResultFeature};
pub use value::{FieldId, SymbolId, Value};
