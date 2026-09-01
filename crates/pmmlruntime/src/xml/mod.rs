//! XML — hardened `quick-xml` 0.37 → `RawPmml` unmarshaling.
//!
//! Cold path only; `Session` never calls this on `run`. Enforces `MAX_DEPTH 512`,
//! `100 MB` cap, and blocks `DTD`/`XXE` per [`reader::PmmlReader`]. `RawPmml` is
//! then lowered to [`crate::ir::Ir`] via [`crate::ir::lower::lower`].
//!
//! # Main types
//!
//! - [`PmmlReader`] — hardened reader (see [`crate::xml::reader`])
//! - [`RawPmml`] — direct PMML infoset (see [`mod@crate::xml::unmarshal`])
//! - [`new_reader`] — creates a hardened `quick-xml` reader
//!
//! # Examples
//!
//! ```rust
//! use pmmlruntime::xml::reader::new_reader;
//! let xml = br#"<PMML version="4.4"></PMML>"#;
//! let mut r = new_reader(xml);
//! assert!(r.is_ok());
//! ```

#![allow(
    clippy::never_loop,
    clippy::match_same_arms,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::semicolon_if_nothing_returned,
    clippy::used_underscore_binding,
    clippy::needless_continue,
    clippy::unnecessary_wraps,
    clippy::redundant_closure_for_method_calls
)]

pub mod reader;
pub mod unmarshal;

pub use reader::{new_reader, PmmlReader};
pub use unmarshal::*;
