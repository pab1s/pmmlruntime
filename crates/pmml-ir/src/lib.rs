pub mod intern;
pub mod ir;
pub mod lower;
pub mod verify;

pub use intern::Interner;
pub use ir::{Ir, MissingValueStrategy, ModelIr, NoTrueChildStrategy, TreeIr};
pub use lower::lower;
pub use verify::{verify_ir, verify_raw};

pub fn placeholder() {}
