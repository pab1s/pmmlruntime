//! Core — zero-cost foundation for the hot path.
//!
//! Provides arena [`BumpArena`](crate::base::arena::BumpArena), field identity [`FieldId`]/[`SymbolId`], typed
//! [`Value`] (`Missing`/`Continuous`/`Discrete`), and [`PmmlError`]/[`Result`].
//! This module is `pub` as `crate::base` (renamed from `core` to avoid `::core`
//! shadowing). It has no XML/IR dependencies and is used by `engine` and
//! `session` on every `Session::run` call.
//!
//! # Main types
//!
//! - [`Value`] — PMML value on the hot path (see [`crate::base::value`])
//! - [`FieldId`]/[`SymbolId`] — dense interned identities (see [`crate::base::field`])
//! - [`DataType`]/[`OpType`]/[`MiningFunction`] — PMML dictionaries
//! - [`BumpArena`](crate::base::arena::BumpArena) — bump allocator for per-row scoring (see [`crate::base::arena`])
//! - [`PmmlError`] — fallible API error (see [`crate::base::error`])
//!
//! # Examples
//!
//! ```rust
//! use pmmlruntime::base::{FieldId, SymbolId, Value, DataType};
//! let id = FieldId(0);
//! let v = Value::Continuous(1.5);
//! let sid = SymbolId(42);
//! assert!(matches!(v, Value::Continuous(_)));
//! ```

pub mod arena;
pub mod error;
pub mod field;
pub mod value;

pub use arena::with_arena;
pub use error::{PmmlError, Result};
pub use field::{DataType, MiningFunction, OpType, ResultFeature};
pub use value::{FieldId, SymbolId, Value};
