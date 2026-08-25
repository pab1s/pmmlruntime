//! `pmml-xml` — hardened `quick-xml` 0.37 unmarshaling to `RawPmml`.
//!
//! This crate is **cold path only**: `bytes: &[u8]` → `RawPmml` → `pmml-ir::lower` → `Ir`.
//! It mirrors `org.jpmml.model.SAXUtil` security: DTD/external entities disabled,
//! file cap `100 MB`, depth cap `512`, XXE hardened via `quick-xml` not expanding entities.
//!
//! # What belongs here
//!
//! - [`reader`] — [`PmmlReader`] and [`new_reader`] wrappers that enforce limits and `trim_text`.
//! - `unmarshal` — [`unmarshal`](crate::unmarshal::unmarshal) `bytes -> Result<RawPmml>` and `Raw*` structs (5758 LOC, 1:1 with `pmml.xsd:4490`
//!   `DataDictionary` + 12 model types + `TransformationDictionary` + `Extension` graceful storage).
//!
//! # Why `quick-xml` pull parser not `serde`
//!
//! PMML XSD has 304 elements, mixed `Attribute`/`Element` ordering, and `Extension` vendor payloads.
//! `quick-xml` pull gives precise control over depth/XXE and avoids `serde`'s `quick-xml` `serialize` overhead
//! forCold cold path (68µs for Iris 2.9 KB).
//!
//! # What to import
//!
//! ```
//! use pmml_xml::unmarshal;
//! let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
//! let raw = unmarshal(xml)?;
//! assert_eq!(raw.data_dictionary.len(), 1);
//! # Ok::<(), pmml_core::PmmlError>(())
//! ```
//!
//! # Security
//!
//! - `#![allow(clippy::never_loop)]` etc. are for `quick-xml` event loops that Clippy flags but are intentional
//!   (depth tracking loops need `continue` after `Event::Decl/Comment`).

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
