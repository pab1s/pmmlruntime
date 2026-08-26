//! XML — hardened `quick-xml` 0.37 unmarshaling to `RawPmml`.

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
