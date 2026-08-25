//! Transformation evaluation — `DerivedField` and PMML expression handling.
//!
//! This module groups the four transformation helpers used by `pmml-session` before
//! model scoring:
//!
//! - [`builtin`] — 100+ `Apply` function name resolution and numeric/string evaluation.
//! - [`discretize`] — interval binning (`Discretize`).
//! - [`mapvalues`] — table lookup (`MapValues`).
//! - [`vm`] — the bytecode interpreter for `DerivedField` DAGs (the hot path; re-exported as [`eval_derived_fields`]).
//!
//! All helpers are pure except `vm`, which maintains a thread-local lag buffer.
//! See each submodule for feature and performance notes.

pub mod builtin;
pub mod discretize;
pub mod mapvalues;
pub mod vm;

pub use vm::eval_derived_fields;
