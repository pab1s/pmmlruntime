//! `pmml-session` — ONNX Runtime-style `Session` API: `PmmlEnv` + `Session` + `Batch` + `ExecutionProvider`.
//!
//! `Session` is the primary user-facing type, analogous to `OrtSession`. It holds
//! `Arc<Ir>` (immutable model) + `SessionOptions` + `Box<dyn ExecutionProvider>` and is
//! `Send+Sync`. Scoring is `Session::run(HashMap<String,Value>)` single (402 ns) or
//! `Session::run_batch` / `run_batch_arrow` via the `Batch` trait (61 ns/row Arrow 100k).
//!
//! # What belongs here
//!
//! - `session` — [`Session`] lifecycle `from_bytes`/`from_file`/`from_ir`, `run`, `run_batch`, `run_batch_arrow`,
//!   `run_record_batch`, `with_value_buffer` (stack `64` + `thread_local!` heap), `field_id`/`symbol_id`/`string_to_value` helpers.
//! - `env` — [`PmmlEnv`] global environment (`Arc` inner, like `OrtEnv`), cheap to `clone`.
//! - `options` — [`SessionOptions`] builder (`GraphOptimizationLevel`, `ExecutionProviderKind`, `intra_op_threads`).
//! - `batch` — [`batch::Batch`] trait (`RowMajor` `Vec<HashMap>` vs `Columnar` `RecordBatch`), [`batch::BatchCtx`] (no per-row alloc), [`batch::BatchResult`].
//! - `arrow` — `RecordBatch` ↔ `Vec<HashMap<String, Value>>` ↔ CSV bridging (`arrow` 53 `csv`).
//! - `providers` — `ExecutionProvider` trait `eval_row`/`eval_batch`, `CpuSerial` vs `CpuBatched` (`rayon` `par_chunks(256)`, fallback `64` serial).
//! - `input` — `string_to_value(field_id: FieldId, s: &str) -> Value` using `DataType`/`OpType` + `symbol_str_to_id`.
//!
//! # Mental model (mirrors ONNX Runtime)
//!
//! ```text
//! bytes: &[u8] → pmml_xml::unmarshal → RawPmml → lower → Ir → Session::from_ir(env.clone(), ir, opts)
//! row: HashMap<String,Value> → Session::run → Value[FieldId] via stack 64 + thread_local → ExecutionProvider::evaluate → HashMap<String,Value>
//! batch: Vec<HashMap> or RecordBatch → BatchCtx (name_to_id + col_map) → provider.eval_batch → BatchResult
//! ```
//!
//! # Why `Batch` trait not only `RecordBatch`
//!
//! Arrow wins at 100k (61 ns/row) but loses for single row (conversion >1µs) and needs schema agreement.
//! `Collection`/`List` (Association) and Python `dict` map naturally to `HashMap`. Provider picks.
//!
//! # What to import
//!
//! ```
//! use pmml_session::{PmmlEnv, Session, SessionOptions, ExecutionProviderKind};
//! use std::collections::HashMap;
//! use pmml_core::Value;
//! let env = PmmlEnv::new();
//! // let sess = Session::from_bytes(&env, bytes, SessionOptions::default())?;
//! // let out = sess.run(HashMap::new())?;
//! # Ok::<(), pmml_core::PmmlError>(())
//! ```

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::pedantic
)]

pub mod arrow;
pub mod batch;
pub mod env;
pub mod input;
pub mod options;
pub mod providers;
pub mod session;

pub use env::PmmlEnv;
pub use options::{ExecutionProviderKind, GraphOptimizationLevel, SessionOptions};
pub use session::Session;
