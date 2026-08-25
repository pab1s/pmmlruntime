//! IR — optimized intermediate representation (`Ir`).

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::too_many_lines,
    clippy::pedantic,
    clippy::module_inception
)]

pub mod intern;
pub mod ir;
pub mod lower;
pub mod verify;

pub use intern::Interner;
pub use ir::*;
pub use lower::lower;
pub use verify::{verify_ir, verify_raw};
