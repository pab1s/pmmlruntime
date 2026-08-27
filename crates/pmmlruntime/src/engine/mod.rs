//! Engine — pure PMML evaluation on `&[Value]`.

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
