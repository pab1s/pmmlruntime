#![allow(
    clippy::never_loop,
    clippy::match_same_arms,
    clippy::needless_range_loop
)]
pub mod reader;
pub mod unmarshal;

pub use reader::{new_reader, PmmlReader};
pub use unmarshal::{
    unmarshal, RawAttribute, RawCategoricalPredictor, RawCharacteristic, RawCluster,
    RawClusteringModel, RawComparisonMeasure, RawDataField, RawGeneralRegressionModel,
    RawMiningField, RawMiningModel, RawNaiveBayesModel, RawNearestNeighborModel, RawNeuralNetwork,
    RawNode, RawNumericPredictor, RawOutputField, RawPmml, RawPredicate, RawRegressionModel,
    RawRegressionTable, RawScoreDistribution, RawScorecard, RawSegment, RawSegmentModel,
    RawSegmentation, RawSupportVectorMachineModel, RawTreeModel,
};

pub fn placeholder() {}
