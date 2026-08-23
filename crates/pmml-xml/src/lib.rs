#![allow(
    clippy::never_loop,
    clippy::match_same_arms,
    clippy::needless_range_loop
)]
pub mod reader;
pub mod unmarshal;

pub use reader::{new_reader, PmmlReader};
pub use unmarshal::{
    unmarshal, RawCategoricalPredictor, RawDataField, RawMiningField, RawMiningModel, RawNode,
    RawNumericPredictor, RawOutputField, RawPmml, RawPredicate, RawRegressionModel,
    RawRegressionTable, RawScoreDistribution, RawSegment, RawSegmentModel, RawSegmentation,
    RawTreeModel,
};

pub fn placeholder() {}
