//! `pmml-ir` — optimized intermediate representation (hot path reads this).
//!
//! Cold path: [`pmml_xml::unmarshal()`] → [`pmml_xml::RawPmml`] → [`crate::lower::lower()`] → [`Ir`].
//! Hot path: `pmml-session` holds `Arc<Ir>` and evaluates via `pmml-evaluator` on `&[Value]`.
//!
//! # What belongs here
//!
//! - [`ir`] — [`Ir`] + [`ir::FieldMeta`], [`ir::MiningSchemaIr`], [`ir::DerivedFieldIr`] (bytecode `Vec<Op>`),
//!   [`ir::BuiltinId`] (100+), [`ir::ModelIr`] (12 variants), [`ir::TreeIr`] etc. All `Clone` and `Send+Sync` via `Arc`.
//! - [`intern`] — [`Interner`] (cold-only `lasso::Rodeo` for `FieldName`/`SymbolId` interning, ~15 call sites via `get_or_intern_field` in [`mod@crate::lower`])
//! - [`mod@lower`] — [`lower()`] `RawPmml -> Result<Ir>`: `DataDictionary` + `MiningSchema` + `TransformationDictionary` + per-model lowering
//! - [`verify`] — [`verify_raw()`] / [`verify_ir()`] mirrors JPMML `UnsupportedMarkupInspector`: rejects `AnomalyDetectionModel` etc. with [`pmml_core::PmmlError::UnsupportedMarkup`]
//!
//! # Why it exists
//!
//! Separates XML parsing (`pmml-xml` 5758 LOC, `quick-xml` hardened) from evaluation. Lowering
//! assigns stable [`pmml_core::FieldId`]/[`pmml_core::SymbolId`], flattens `TreeModel` nodes to
//! `Vec<NodeIr>`, topologically sorts `DerivedFieldIr` DAG, and caches `field_names` / `symbol_names` maps.
//!
//! # What to import
//!
//! ```
//! use pmml_ir::{lower, verify_raw, verify_ir};
//! // let raw = pmml_xml::unmarshal(bytes)?;
//! // verify_raw(&raw)?;
//! // let ir = lower(raw)?;
//! // verify_ir(&ir)?;
//! ```
//!
//! # Invariants
//!
//! - Every `FieldId` in `Ir.mining_schema` / `DerivedFieldIr` is in `Ir.field_names`.
//! - `Ir.max_field_id` is at least 16 and `values: &mut [Value]` length is `max_field_id` vs `num_fields()+4`.
//! - `verify_raw` handles `Extension` gracefully (stored not error) and captures `unsupported_model` for D1.

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::too_many_lines,
    clippy::pedantic
)]

pub mod intern;
pub mod ir;
pub mod lower;
pub mod verify;

pub use intern::Interner;
pub use ir::{Ir, MissingValueStrategy, ModelIr, NoTrueChildStrategy, TreeIr};
pub use lower::lower;
pub use verify::{verify_ir, verify_raw};
