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
