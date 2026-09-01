//! IR — optimized intermediate representation for the session.
//!
//! The `Ir` is an `Arc`-immutable, posterior-optimized plan built from
//! `RawPmml` via [`lower::lower`]. It flattens models into `Vec<NodeIr>`/`Vec<Op>`
//! bytecode and interns strings via [`Interner`] (`lasso::Rodeo`) on the cold
//! path. Verified by [`verify::verify_ir`]/[`verify::verify_raw`].
//!
//! # Main types
//!
//! - [`Ir`] — root plan (`field_names`, `model`, `derived_fields`)
//! - [`Interner`] — string interning for `FieldId`/`SymbolId`
//! - [`lower::lower`] — `RawPmml` → `Ir` lowering
//! - [`verify_ir`] — IR invariant checks
//!
//! # Examples
//!
//! ```rust
//! use pmmlruntime::ir::{Ir, verify_ir};
//! // Ir is built via `lower::lower(raw)` and shared as `Arc<Ir>` in `Session`.
//! ```

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
