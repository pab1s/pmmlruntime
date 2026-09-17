//! Engine — pure PMML evaluation on `&[Value]`.
//!
//! Evaluates the optimized [`crate::ir::Ir`] without I/O or allocation on the
//! hot path. Every evaluator is a pure function over `&[Value]` indexed by
//! [`FieldId`](crate::base::FieldId), `Send + Sync`, and benchable (~402 ns single-row for trees).
//!
//! # Main types
//!
//! - [`mining_schema::apply_mining_schema`] — rewrites `Missing`/outlier per `MiningField`
//! - [`transform::eval_derived_fields`] — `DerivedField` DAG bytecode (`Vec<Op>`)
//! - [`predicate`] — `CompoundPredicate`/`SimplePredicate` evaluation
//! - [`models`] — 19 model scorers (`Tree`/`Regression`/`Mining`/…)
//! - [`output`] — `OutputField` post-processing
//! - [`targets`] — `Target` rescaling/casting
//! - [`simd`] — `wide` `f64x4` batch when `simd` feature active
//!
//! # Examples
//!
//! ```rust
//! use pmmlruntime::base::{FieldId, Value};
//! use pmmlruntime::engine::predicate::eval_predicate;
//! use pmmlruntime::ir::{PredicateIr, SimpleOperator, SymbolIdOrContinuous};
//! let pred = PredicateIr::True;
//! assert!(eval_predicate(&pred, &[Value::Missing]));
//! ```

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_arguments,
    clippy::pedantic,
    clippy::nursery,
    clippy::style,
    clippy::perf,
    clippy::complexity,
    clippy::suspicious,
    rustdoc::redundant_explicit_links
)]

pub mod mining_schema;
pub mod models;
pub mod output;
pub mod predicate;
pub mod simd;
pub mod targets;
pub mod transform;

pub use mining_schema::apply_mining_schema;
pub use transform::eval_derived_fields;
