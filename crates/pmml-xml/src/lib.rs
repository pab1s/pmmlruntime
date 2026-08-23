#![allow(
    clippy::never_loop,
    clippy::match_same_arms,
    clippy::needless_range_loop
)]
pub mod reader;
pub mod unmarshal;

pub use reader::{new_reader, PmmlReader};
pub use unmarshal::{
    unmarshal, RawDataField, RawMiningField, RawNode, RawPmml, RawPredicate, RawTreeModel,
};

pub fn placeholder() {}
