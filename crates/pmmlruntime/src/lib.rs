//! `pmmlruntime` — PMML 4.4 runtime in a single crate (inspired by JPMML + ONNX Runtime).
//!
//! Previously a 9-crate workspace (`pmml-core`, `pmml-xml`, `pmml-ir`, `pmml-evaluator`, `pmml-session`, …).
//! Now a **single crate** with modules `base`, `xml`, `ir`, `engine`, `session`, `ffi`, `python` —
//! one `cargo add pmmlruntime` and one `cargo doc -p pmmlruntime` page.
//!
//! # Modules
//!
//! - [`base`] — zero-cost types `Value`/`FieldId`/`DataType`/`PmmlError`, arena `BumpArena` (hot path foundation, no XML/IR).
//! - [`xml`] — hardened `quick-xml` 0.37 → `RawPmml` (cold, `MAX_DEPTH 512`, `100 MB`, XXE blocked).
//! - [`ir`] — optimized `Ir` (`Arc` immutable, `Vec<NodeIr>` flat, `DerivedFieldIr` DAG `Vec<Op>`), `Interner` (cold `Rodeo`).
//! - [`engine`] — pure evaluation on `&[Value]` (12 models: `Tree`/`Regression`/`Mining`/`Scorecard`/…+ `vm` bytecode, `simd` `wide` `f64x4`).
//! - [`session`] — ONNX-style `Session` API (`PmmlEnv` + `Session` + `Batch` + `ExecutionProvider` `CpuSerial`/`CpuBatched` `rayon`).
//! - [`ffi`] — C ABI `PmmlEnv`/`PmmlSession` (`onnxruntime_c_api.h` parity, `Safety` contracts).
//! - [`python`] — `pyo3 0.22` placeholder (`python` feature, future `PySession`).
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

#![allow(clippy::pedantic, clippy::nursery)]

pub mod base;
pub mod engine;
pub mod ffi;
pub mod ir;
pub mod python;
pub mod session;
pub mod xml;

// Re-exports for ergonomic `use pmmlruntime::{Session, Value, PmmlEnv}` and backwards compat with `pmml_*` paths.
pub use base::{FieldId, PmmlError, Result, SymbolId, Value};
pub use session::{PmmlEnv, Session, SessionOptions};

/// Crate version (workspace `0.1.0`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder so `cargo doc` does not flag the crate as empty.
pub fn placeholder() {}

// Keep `pmml_*` aliases for code that still uses `crate::base::` etc. during migration — will be removed in 0.2.
#[doc(hidden)]
pub mod pmml_core {
    pub use crate::base::*;
}
#[doc(hidden)]
pub mod pmml_xml {
    pub use crate::xml::*;
}
#[doc(hidden)]
pub mod pmml_ir {
    pub use crate::ir::*;
}
#[doc(hidden)]
pub mod pmml_evaluator {
    pub use crate::engine::*;
}
#[doc(hidden)]
pub mod pmml_session {
    pub use crate::session::*;
}
