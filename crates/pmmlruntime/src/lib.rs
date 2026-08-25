//! `pmmlruntime` — single-crate facade over the workspace (re-exports `pmml-core`, `pmml-session`, `pmml-ir`, `pmml-xml`, `pmml-evaluator`).
//!
//! The workspace is virtual (`[workspace]` without `[package]` at the root), so `cargo doc --workspace`
//! generates 9 separate crates (`pmml_core`, `pmml_session`, …) instead of a single `pmmlruntime`.
//! This crate exists so `cargo doc -p pmmlruntime` and `docs.rs` show **one** entry point for the full public API.
//!
//! # Why 9 crates
//!
//! - `pmml-core` (hot types `Value`/`FieldId`/`DataType`), `pmml-xml` (`quick-xml` cold 5758 LOC), `pmml-ir` (optimized `Ir` `Arc`), `pmml-evaluator` (12 pure models), `pmml-session` (`PmmlEnv`+`Session`+`Batch` ONNX), `pmml-ffi` (C ABI), `pmml-python` (pyo3), `pmml-cli` (bin), `pmml-bench` (criterion).
//! - Splitting cold from hot allows `cargo doc -p pmml-core` without XML/IR and per-crate `cargo check` (prevents Bun-style 16k cycles).
//!
//! # What it re-exports
//!
//! - `core` → [`pmml_core`] (arena, `FieldId`/`SymbolId`/`Value`, `DataType`/`OpType`/`ResultFeature`, `PmmlError`)
//! - `xml` → [`pmml_xml`] (`unmarshal`, `RawPmml`, `PmmlReader`)
//! - `ir` → [`pmml_ir`] (`Ir`, `Interner`, `lower`, `verify`)
//! - `evaluator` → [`pmml_evaluator`] (`evaluate_tree`, `apply_mining_schema`, `simd`)
//! - `session` → [`pmml_session`] (`PmmlEnv`, `Session`, `SessionOptions`, `Batch`, `ExecutionProviderKind` — **primary API**)
//!
//! # How to view the unified docs
//!
//! ```sh
//! cargo doc -p pmmlruntime --no-deps --open   # opens target/doc/pmmlruntime/index.html
//! cargo doc -p pmmlruntime --open              # with deps (quick-xml, arrow, rayon)
//! ```
//! Also: `cargo doc --workspace --no-deps` generates all 9, and `target/doc/index.html` lists them.
//! `docs/ARCHITECTURE.md` explains the `bytes→RawPmml→Ir→Session::run(Value[FieldId])` flow.
//!
//! # Minimal example (via the facade)
//!
//! ```rust
//! use std::collections::HashMap;
//! use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
//! use pmmlruntime::core::Value;
//!
//! let env = PmmlEnv::new();
//! let xml = br#"
//! <PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary>
//! <TreeModel functionName="classification">
//!   <MiningSchema><MiningField name="x"/></MiningSchema>
//!   <Node score="a"><True/></Node>
//! </TreeModel></PMML>"#;
//! let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
//! let mut input = HashMap::new();
//! input.insert("x".to_string(), Value::Continuous(1.4));
//! let out = sess.run(input).unwrap();
//! assert!(out.contains_key("predictedValue"));
//! ```
//!
//! # Feature flags
//!
//! - `simd` — enables `wide` `f64x4` in `pmml-session`/`pmml-evaluator` (`cargo doc -p pmmlruntime --features simd`).
//! - `python` — reserved for `pyo3` (does not link `libpython` without the feature).
//!
//! [`pmml_core`]: pmml_core
//! [`pmml_xml`]: pmml_xml
//! [`pmml_ir`]: pmml_ir
//! [`pmml_evaluator`]: pmml_evaluator
//! [`pmml_session`]: pmml_session

pub use pmml_core as core;
pub use pmml_evaluator as evaluator;
pub use pmml_ir as ir;
pub use pmml_session as session;
pub use pmml_xml as xml;

/// Facade version (workspace `0.1.0`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder so `cargo doc` does not flag the crate as empty when built with `--no-deps` and no features.
pub fn placeholder() {}
