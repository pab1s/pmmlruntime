//! `pmml-core` — zero-cost foundation: arena, field types, values, errors.
//!
//! This crate has **no XML and no IR** dependencies. It is the hot-path foundation
//! imported by every other crate. Keep it minimal, `Copy`-friendly, and allocation-free
//! where possible.
//!
//! # What belongs here
//!
//! - [`arena`] — bump allocation (`BumpArena`, `with_arena`) for per-`run()` scoring, mirrors
//!   ONNX Runtime `BFCArena`. Hot path uses a stack-allocated `Value` buffer for `<=64` fields.
//! - [`field`] — PMML `DATATYPE`/`OPTYPE`/`MINING-FUNCTION`/`RESULT-FEATURE` enums derived from
//!   `pmml.xsd:4490`. `FromStr` is case-sensitive per the spec.
//! - [`value`] — core scoring types: [`FieldId`] (`u32` index into `values[field_id]`), [`SymbolId`]
//!   (interned discrete `u32`), and [`Value`] (`Continuous(f64)` / `Discrete(SymbolId)` / `Missing`).
//!   `Missing` is a variant, not `Option<Value>`, to avoid double wrapping on the hot path.
//! - [`error`] — unified [`PmmlError`] / [`Result`] (`thiserror`, no backtrace on hot path).
//!
//! # Why it exists
//!
//! Separates cold-path interning/parsing from hot-path scoring. `pmml-xml` parses,
//! `pmml-ir` lowers to `Ir`, `pmml-session` materializes `Value[FieldId]` arrays,
//! and `pmml-evaluator` operates purely on `&[Value]`.
//!
//! # What to import
//!
//! Most callers need [`Value`], [`FieldId`], [`PmmlError`], and field enums:
//!
//! ```
//! use pmml_core::{FieldId, PmmlError, Value, DataType};
//! let id = FieldId(0);
//! let v = Value::Continuous(3.14);
//! assert!(!v.is_missing());
//! ```
//!
//! # Invariants
//!
//! - `FieldId(0)` is a valid id; there is no sentinel. `Missing` is carried in [`Value`].
//! - `SymbolId(u32)` is stable per `Interner` (`pmml-ir`) but dense for `symbol_names_vec` in `pmml-session`.
//! - `DataType::DateDaysSince0` and `DateTimeSecondsSince0` are unsupported per JPMML and rejected early.
//!
//! [`arena`]: crate::arena
//! [`field`]: crate::field
//! [`value`]: crate::value
//! [`error`]: crate::error

#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

pub mod arena;
pub mod error;
pub mod field;
pub mod value;

pub use arena::with_arena;
pub use error::{PmmlError, Result};
pub use field::{DataType, MiningFunction, OpType, ResultFeature};
pub use value::{FieldId, SymbolId, Value};
