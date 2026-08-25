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
    RawComparisonMeasure, RawCon, RawDataField, RawDefineFunction, RawDerivedField,
    RawDiscretizeBin, RawExpression, RawFieldColumnPair, RawGeneralRegressionModel, RawInterval,
    RawItem, RawItemset, RawLinearNorm, RawMiningField, RawMiningModel, RawNaiveBayesModel,
    RawNearestNeighborModel, RawNeuralInput, RawNeuralLayer, RawNeuralNetwork, RawNeuron, RawNode,
    RawNumericPredictor, RawOutputField, RawPairCounts, RawParameterField, RawPmml, RawPredicate,
    RawRegressionModel, RawRegressionTable, RawRuleSet, RawRuleSetModel, RawScoreDistribution,
    RawScorecard, RawSegment, RawSegmentModel, RawSegmentation, RawSimpleRule,
    RawSupportVectorMachineModel, RawTarget, RawTargetValue, RawTargetValueCount,
    RawTargetValueStat, RawTreeModel,
};

pub fn placeholder() {}
