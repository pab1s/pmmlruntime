#![allow(clippy::never_loop, clippy::match_same_arms, clippy::needless_range_loop)]
pub mod reader;
pub mod unmarshal;

pub use reader::{PmmlReader, new_reader};
pub use unmarshal::{RawDataField, RawMiningField, RawNode, RawPmml, RawPredicate, RawTreeModel, unmarshal};

pub fn placeholder() {}
