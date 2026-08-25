//! `pmml-evaluator` — pure PMML evaluation on `&[Value]` (no `Session` state).
//!
//! This crate is the `pmml-ir::Ir` interpreter. All `evaluate_*` functions are pure
//! `(&IrStruct, &[Value]) -> Value` that read `values[field.as_usize()]` and never allocate.
//! It ports `ExpressionUtil`/`TypeUtil`/`OutputUtil`/`Functions` and the 13 `visitors/` batteries
//! as explicit `lower` passes in `pmml-ir`, plus the 12 model evaluators.
//!
//! # What belongs here
//!
//! - [`mining_schema`] — [`mining_schema::apply_mining_schema`] handles
//!   `invalidValueTreatment`/`outlierTreatment`/`missingValueReplacement` per [`FieldMeta`](pmml_ir::ir::FieldMeta).
//! - [`models`] — [`models::evaluate_tree`] (275 LOC), [`models::evaluate_regression`], [`models::evaluate_mining`] (369 LOC),
//!   [`models::evaluate_scorecard`], [`models::evaluate_clustering`], [`models::evaluate_naive_bayes`],
//!   [`models::evaluate_nearest_neighbor`] (356 LOC), [`models::evaluate_support_vector_machine`],
//!   [`models::evaluate_neural_network`], [`models::evaluate_general_regression`] etc. Each is a free fn.
//! - [`transform`] — [`transform::vm::eval_derived_fields`] bytecode interpreter for [`Op`](pmml_ir::ir::Op),
//!   [`transform::builtin`] 100+ PMML functions, [`transform::discretize`], [`transform::mapvalues`].
//! - [`predicate`] — [`predicate::eval_predicate`] for `Tree`/`RuleSet`/`Segment`.
//! - [`output`] — [`output::build_output`] / [`output::build_output_with_context`] for 26 [`ResultFeature`](pmml_core::field::ResultFeature) (4 unsupported → `Missing`).
//! - [`targets`] — [`targets::apply_targets`] post-processing (`rescaleConstant/Factor`, `castInteger`).
//! - [`simd`] — optional `wide` `f64x4` batch for single-table regression (`feature = "simd"`).
//!
//! # Why pure functions not trait objects
//!
//! JPMML used `ModelEvaluator` subclass + `Visitor` mutation. Here `Ir` is `Arc` immutable and
//! evaluators are `match ModelIr { Tree(t) => evaluate_tree(t, values), ... }` — branchless,
//! `Send+Sync`, and benchable at 402 ns single.
//!
//! # What to import
//!
//! ```
//! use pmml_evaluator::models::evaluate_tree;
//! use pmml_core::{FieldId, SymbolId, Value};
//! use pmml_ir::ir::*;
//! // Build a tiny tree and score it:
//! let f = FieldId(0);
//! let t = TreeIr {
//!     function_name: "classification".into(),
//!     missing_value_strategy: MissingValueStrategy::None,
//!     no_true_child_strategy: NoTrueChildStrategy::ReturnNullPrediction,
//!     nodes: vec![NodeIr { id: None, score: Some(SymbolIdOrContinuous::Symbol(SymbolId(1))), predicate: PredicateIr::True, children: vec![], default_child: None, score_distributions: vec![] }],
//!     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
//!     targets: vec![], output: vec![],
//! };
//! let pred = evaluate_tree(&t, &[Value::Continuous(1.0)]);
//! assert_eq!(pred, Value::Discrete(SymbolId(1)));
//! ```
//!
//! # Architecture
//!
//! Cold path: `pmml_xml::unmarshal` → `pmml_ir::lower` (assigns [`FieldId`](pmml_core::FieldId)/[`SymbolId`](pmml_core::SymbolId),
//! flattens `TreeModel` nodes, sorts `DerivedField` DAG) → `Arc<Ir>`.
//! Hot path (`pmml-session`): allocate `Vec<Value>` sized to `Ir::num_fields()`, call
//! [`apply_mining_schema`](mining_schema::apply_mining_schema) → [`transform::eval_derived_fields`] → `evaluate_*`
//! → [`targets::apply_targets`] → [`output::build_output`].
//!
//! # Feature flags
//!
//! - `simd` (via `wide`): enables [`simd::evaluate_regression_batch_simd`] AVX2/NEON fast path.
//!   No `wasm32` SIMD; fallback is scalar.
//!
//! # Panics and thread safety
//!
//! No evaluator panics on well-formed `Ir`; all `FieldId` indexing is bounds-checked and `Missing`
//! is propagated. Evaluators are `Send + Sync` and share `Ir` immutably. `transform::vm` uses
//! a `thread_local!` lag buffer (`!Sync` but `Send`).

pub mod mining_schema;
pub mod models;
pub mod output;
pub mod predicate;
pub mod simd;
pub mod targets;
pub mod transform;

pub use mining_schema::apply_mining_schema;
pub use transform::eval_derived_fields;
