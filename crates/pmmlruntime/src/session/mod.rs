//! Session — session runtime-style `Session` API.

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
