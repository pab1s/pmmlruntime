#![allow(
    clippy::never_loop,
    clippy::match_same_arms,
    clippy::needless_range_loop
)]
pub mod reader;
pub mod unmarshal;

pub use reader::{new_reader, PmmlReader};
pub use unmarshal::{
    unmarshal, RawAssociationModel, RawAssociationRule, RawAttribute, RawBayesInput,
    RawCategoricalPredictor, RawCharacteristic, RawCluster, RawClusteringModel,
    RawComparisonMeasure, RawCon, RawDataField, RawGeneralRegressionModel, RawItem, RawItemset,
    RawMiningField, RawMiningModel, RawNaiveBayesModel, RawNearestNeighborModel, RawNeuralInput,
    RawNeuralLayer, RawNeuralNetwork, RawNeuron, RawNode, RawNumericPredictor, RawOutputField,
    RawPairCounts, RawPmml, RawPredicate, RawRegressionModel, RawRegressionTable, RawRuleSet,
    RawRuleSetModel, RawScoreDistribution, RawScorecard, RawSegment, RawSegmentModel,
    RawSegmentation, RawSimpleRule, RawSupportVectorMachineModel, RawTargetValueCount,
    RawTargetValueStat, RawTreeModel,
};

pub fn placeholder() {}
