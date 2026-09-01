//! Session — ORT-style session API for PMML scoring.
//!
//! Owns an `Arc<Ir>` plan and exposes `PmmlEnv` + [`Session`] + [`SessionOptions`]
//! + [`GraphOptimizationLevel`] + `Batch` for `HashMap`/`Vec<HashMap>`/`Arrow`.
//! All scoring is via [`Session::run`] over `&dyn Batch`, thread-safe (`Send+Sync`)
//! with `rayon` auto-parallel for batches. See `docs/ARCHITECTURE.md` §2.
//!
//! # Main types
//!
//! - [`PmmlEnv`] — environment for `Session` creation (see [`crate::session::env`])
//! - [`Session`] — compiled model session (see [`crate::session::session`])
//! - [`SessionOptions`]/[`GraphOptimizationLevel`] — optimization + threading
//! - [`batch::Batch`] — input row trait, `Arrow` zero-copy in [`arrow`]
//!
//! # Examples
//!
//! ```rust
//! use std::collections::HashMap;
//! use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
//! use pmmlruntime::session::batch::Batch;
//! use pmmlruntime::base::Value;
//! let env = PmmlEnv::new();
//! let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
//! let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
//! let mut m = HashMap::new();
//! m.insert("x".into(), Value::Continuous(1.0));
//! let out = sess.run(&m as &dyn Batch).unwrap().into_single().unwrap();
//! assert!(out.contains_key("predictedValue"));
//! ```

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::pedantic,
    clippy::module_inception
)]

pub mod arrow;
pub mod batch;
pub mod env;
pub mod input;
pub mod options;
pub mod providers;
pub mod session;

pub use env::PmmlEnv;
pub use options::{GraphOptimizationLevel, SessionOptions};
pub use session::Session;

pub(crate) use session::with_value_buffer;
