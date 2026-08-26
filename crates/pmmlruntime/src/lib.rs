//! `pmmlruntime` — PMML 4.4 runtime.
//!
//! Modules `base`, `xml`, `ir`, `engine`, `session`, `ffi`, `python` — see `docs/ARCHITECTURE.md`.
//!
//! # Modules
//!
//! - [`base`] — zero-cost types `Value`/`FieldId`/`DataType`/`PmmlError`, arena `BumpArena` (hot-path, no XML/IR — see docs/ARCHITECTURE.md).
//! - [`xml`] — hardened `quick-xml` 0.37 → `RawPmml` (cold path, `MAX_DEPTH 512`, `100 MB` cap, DTD/XXE blocked — see `crate::xml::reader`).
//! - [`ir`] — optimized `Ir` for posterior plan optimization (`Arc` immutable, `Vec<NodeIr>` flat, `DerivedFieldIr` DAG with `Vec<Op>` bytecode, cold `Rodeo` interning).
//! - [`engine`] — pure evaluation on `&[Value]` (19 models: `Tree`/`Regression`/`Mining`/`Scorecard`/`Clustering`/`NaiveBayes`/… + `vm` bytecode, `simd` `wide` `f64x4` when `simd` feature is active).
//! - [`session`] — session API (`PmmlEnv` + `Session` + `Batch` + `ExecutionProvider` `CpuSerial`/`CpuBatched` via `rayon` — see docs/ARCHITECTURE.md §2).
//! - [`ffi`] — C ABI with opaque `PmmlEnv`/`PmmlSession` handles and `Safety` contracts (see `crate::ffi`).
//! - [`python`] — `pyo3 0.22` extension-module placeholder (feature `python`, see `crate::python`).
//!
//! # Primary API — `session`
//!
//! ```rust
//! use std::collections::HashMap;
//! use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
//! use pmmlruntime::base::Value;
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
//! - `simd` — enables `wide` `f64x4` batch for `engine` + `session` (`cargo doc --features simd`).
//! - `python` — enables `pyo3` extension-module (otherwise no `libpython` link).
//!
//! # Architecture
//!
//! See `docs/ARCHITECTURE.md` for the `bytes→RawPmml→Ir→Session::run(Value[FieldId])` flow, ownership `Arc<Ir>`,
//! concurrency `rayon`, and `BumpArena` vs `LoadingCache` tradeoffs.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::if_same_then_else,
    clippy::manual_map,
    clippy::large_enum_variant
)]
#![allow(dead_code, unused_mut, unused_variables)]

pub mod base;
pub mod engine;
pub mod ffi;
pub mod ir;
pub mod python;
pub mod session;
pub mod xml;

// Re-exports for ergonomic `use pmmlruntime::{Session, Value, PmmlEnv}`.
pub use base::{FieldId, PmmlError, Result, SymbolId, Value};
pub use session::{PmmlEnv, Session, SessionOptions};

/// Crate version (workspace `0.1.0`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
