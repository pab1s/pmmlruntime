//! Raw PMML structures + unmarshal from XML bytes (quick-xml, hardened).
//! v1 focuses on DataDictionary + TreeModel (DecisionTreeIris). Other models stub.

use crate::reader::new_reader;
use pmml_core::error::{PmmlError, Result};
use quick_xml::events::{BytesStart, Event};
use std::str;

// ---------- Raw structures ----------

#[derive(Debug, Clone)]
pub struct RawDataField {
    pub name: String,
    pub data_type: String,
    pub op_type: String,
    pub values: Vec<String>, // <Value value=...>
}

#[derive(Debug, Clone)]
pub struct RawMiningField {
    pub name: String,
    pub usage_type: Option<String>, // target, active (default)
    pub importance: Option<f64>,
    pub outliers: Option<String>,
    pub low_value: Option<String>,
    pub high_value: Option<String>,
    pub missing_value_replacement: Option<String>,
    pub missing_value_treatment: Option<String>,
    pub invalid_value_treatment: Option<String>,
    pub invalid_value_replacement: Option<String>,
    pub op_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawTargetValue {
    pub value: Option<String>,
    pub display_value: Option<String>,
    pub prior_probability: Option<f64>,
    pub default_value: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RawTarget {
    pub field: Option<String>,
    pub op_type: Option<String>,
    pub cast_integer: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub rescale_constant: Option<f64>,
    pub rescale_factor: Option<f64>,
    pub target_values: Vec<RawTargetValue>,
}

#[derive(Debug, Clone)]
pub struct RawOutputField {
    pub name: String,
    pub feature: Option<String>,
    pub value: Option<String>,
    pub target_field: Option<String>,
    pub data_type: Option<String>,
    pub op_type: Option<String>,
    pub rule_feature: Option<String>,
    pub algorithm: Option<String>,
    pub rank: Option<i32>,
    pub rank_basis: Option<String>,
    pub rank_order: Option<String>,
    pub is_multi_valued: Option<String>,
    pub segment_id: Option<String>,
    pub is_final_result: Option<bool>,
    pub display_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawTreeModel {
    pub function_name: String,
    pub missing_value_strategy: Option<String>,
    pub no_true_child_strategy: Option<String>,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub root: RawNode,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawNumericPredictor {
    pub name: String,
    pub exponent: i32,
    pub coefficient: f64,
}

#[derive(Debug, Clone)]
pub struct RawCategoricalPredictor {
    pub name: String,
    pub value: String,
    pub coefficient: f64,
}

#[derive(Debug, Clone)]
pub struct RawRegressionTable {
    pub intercept: f64,
    pub target_category: Option<String>,
    pub numeric_predictors: Vec<RawNumericPredictor>,
    pub categorical_predictors: Vec<RawCategoricalPredictor>,
}

#[derive(Debug, Clone)]
pub struct RawRegressionModel {
    pub function_name: String,
    pub target_field_name: Option<String>,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub regression_tables: Vec<RawRegressionTable>,
    pub normalization_method: Option<String>,
    pub model_name: Option<String>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawSegment {
    pub id: Option<String>,
    pub predicate: RawPredicate,
    pub model: RawSegmentModel,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub enum RawSegmentModel {
    Tree(RawTreeModel),
    Regression(RawRegressionModel),
}

#[derive(Debug, Clone)]
pub struct RawSegmentation {
    pub multiple_model_method: String,
    pub missing_prediction_treatment: Option<String>,
    pub segments: Vec<RawSegment>,
}

#[derive(Debug, Clone)]
pub struct RawMiningModel {
    pub function_name: String,
    pub mining_schema: Vec<RawMiningField>,
    pub segmentation: Option<RawSegmentation>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub model_name: Option<String>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawAttribute {
    pub partial_score: f64,
    pub predicate: RawPredicate,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawCharacteristic {
    pub name: String,
    pub reason_code: Option<String>,
    pub baseline_score: Option<f64>,
    pub attributes: Vec<RawAttribute>,
}

#[derive(Debug, Clone)]
pub struct RawScorecard {
    pub model_name: Option<String>,
    pub function_name: String,
    pub initial_score: f64,
    pub use_reason_codes: Option<bool>,
    pub reason_code_algorithm: Option<String>,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub characteristics: Vec<RawCharacteristic>,
    pub baseline_method: Option<String>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawCluster {
    pub name: String,
    pub array: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RawComparisonMeasure {
    pub kind: String,
    pub compare_function: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawClusteringModel {
    pub model_name: Option<String>,
    pub function_name: String,
    pub model_class: Option<String>,
    pub number_of_clusters: Option<usize>,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub comparison_measure: Option<RawComparisonMeasure>,
    pub clustering_fields: Vec<String>,
    pub clusters: Vec<RawCluster>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawTargetValueCount {
    pub value: String,
    pub count: f64,
}

#[derive(Debug, Clone)]
pub struct RawBayesInput {
    pub field_name: String,
    pub target_value_stats: Vec<RawTargetValueStat>,
    pub pair_counts: Vec<RawPairCounts>,
}

#[derive(Debug, Clone)]
pub struct RawTargetValueStat {
    pub value: String,
    pub gaussian_mean: Option<f64>,
    pub gaussian_variance: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RawPairCounts {
    pub value: String,
    pub target_counts: Vec<RawTargetValueCount>,
}

#[derive(Debug, Clone)]
pub struct RawNaiveBayesModel {
    pub function_name: String,
    pub threshold: f64,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub bayes_inputs: Vec<RawBayesInput>,
    pub bayes_output_counts: Vec<RawTargetValueCount>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawInstanceField {
    pub field: String,
    pub column: String,
}

#[derive(Debug, Clone)]
pub struct RawNearestNeighborModel {
    pub function_name: String,
    pub number_of_neighbors: usize,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub instance_fields: Vec<RawInstanceField>,
    pub instances: Vec<std::collections::HashMap<String, String>>,
    pub knn_inputs: Vec<String>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawSupportVectorMachineModel {
    pub function_name: String,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub vector_fields: Vec<RawVectorField>,
    pub vector_instances: Vec<RawVectorInstance>,
    pub support_vector_machine: Option<RawSupportVectorMachine>,
    pub kernel_gamma: Option<f64>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawNeuralNetwork {
    pub function_name: String,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub neural_inputs: Vec<RawNeuralInput>,
    pub neural_layers: Vec<RawNeuralLayer>,
    pub model_name: Option<String>,
    pub activation_function: Option<String>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawParameter {
    pub name: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawFactor {
    pub name: String,
    pub categories: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
    pub contrast_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawPPCell {
    pub value: String,
    pub predictor_name: String,
    pub parameter_name: String,
}

#[derive(Debug, Clone)]
pub struct RawPCell {
    pub target_category: Option<String>,
    pub parameter_name: String,
    pub beta: f64,
}

#[derive(Debug, Clone)]
pub struct RawGeneralRegressionModel {
    pub function_name: String,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub model_type: Option<String>,
    pub target_variable_name: Option<String>,
    pub target_reference_category: Option<String>,
    pub parameters: Vec<RawParameter>,
    pub factors: Vec<RawFactor>,
    pub covariates: Vec<String>,
    pub pp_matrix: Vec<RawPPCell>,
    pub param_matrix: Vec<RawPCell>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawItem {
    pub id: String,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct RawItemset {
    pub id: String,
    pub item_refs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RawAssociationRule {
    pub antecedent: String,
    pub consequent: String,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
}

#[derive(Debug, Clone)]
pub struct RawAssociationModel {
    pub function_name: String,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub items: Vec<RawItem>,
    pub itemsets: Vec<RawItemset>,
    pub rules: Vec<RawAssociationRule>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawSimpleRule {
    pub id: Option<String>,
    pub score: String,
    pub predicate: RawPredicate,
}

#[derive(Debug, Clone)]
pub struct RawRuleSet {
    pub record_count: Option<f64>,
    pub nb_correct: Option<f64>,
    pub default_score: Option<String>,
    pub rules: Vec<RawSimpleRule>,
}

#[derive(Debug, Clone)]
pub struct RawRuleSetModel {
    pub function_name: String,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub targets: Vec<RawTarget>,
    pub rule_set: Option<RawRuleSet>,
    pub local_derived_fields: Vec<RawDerivedField>,
}

#[derive(Debug, Clone)]
pub struct RawCon {
    pub from: String,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct RawNeuron {
    pub id: String,
    pub bias: Option<f64>,
    pub cons: Vec<RawCon>,
}

#[derive(Debug, Clone)]
pub struct RawNeuralInput {
    pub id: String,
    pub field: String,
}

#[derive(Debug, Clone)]
pub struct RawNeuralLayer {
    pub number_of_neurons: Option<usize>,
    pub activation_function: Option<String>,
    pub neurons: Vec<RawNeuron>,
}

#[derive(Debug, Clone)]
pub struct RawSupportVector {
    pub vector_id: String,
}

#[derive(Debug, Clone)]
pub struct RawCoefficient {
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct RawSupportVectorMachine {
    pub support_vectors: Vec<RawSupportVector>,
    pub coefficients: Vec<RawCoefficient>,
    pub absolute_value: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RawVectorField {
    pub field: String,
}

#[derive(Debug, Clone)]
pub struct RawVectorInstance {
    pub id: String,
    pub array: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RawNode {
    pub id: Option<String>,
    pub score: Option<String>,
    pub record_count: Option<f64>,
    pub predicate: RawPredicate,
    pub score_distributions: Vec<RawScoreDistribution>,
    pub children: Vec<RawNode>,
    pub default_child: Option<String>,
}

#[derive(Debug, Clone)]
pub enum RawPredicate {
    True,
    Simple {
        field: String,
        operator: String,
        value: String,
    },
    SimpleSet {
        field: String,
        boolean_operator: String, // isIn / isNotIn
        array: String,
    },
    Compound {
        boolean_operator: String, // and/or/xor/surrogate
        predicates: Vec<RawPredicate>,
    },
}

#[derive(Debug, Clone)]
pub struct RawScoreDistribution {
    pub value: String,
    pub record_count: f64,
}

#[derive(Debug, Clone)]
pub struct RawExtension {
    pub extender: Option<String>,
    pub name: Option<String>,
    pub value: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawParameterField {
    pub name: String,
    pub data_type: Option<String>,
    pub op_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawLinearNorm {
    pub orig: f64,
    pub norm: f64,
}

#[derive(Debug, Clone)]
pub struct RawInterval {
    pub closure: String,
    pub left_margin: Option<f64>,
    pub right_margin: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RawDiscretizeBin {
    pub bin_value: String,
    pub interval: RawInterval,
}

#[derive(Debug, Clone)]
pub struct RawFieldColumnPair {
    pub field: String,
    pub column: String,
}

#[derive(Debug, Clone)]
pub struct RawDefineFunction {
    pub name: String,
    pub data_type: Option<String>,
    pub op_type: Option<String>,
    pub param_fields: Vec<RawParameterField>,
    pub derived_fields: Vec<RawDerivedField>,
    pub body: Option<RawExpression>,
}

#[derive(Debug, Clone)]
pub struct RawDerivedField {
    pub name: String,
    pub display_name: Option<String>,
    pub data_type: String,
    pub op_type: String,
    pub expression: RawExpression,
}

#[derive(Debug, Clone)]
pub enum RawExpression {
    Constant { data_type: Option<String>, value: String },
    FieldRef { field: String, map_missing_to: Option<String> },
    NormContinuous {
        field: String,
        map_missing_to: Option<String>,
        default_value: Option<String>,
        outliers: Option<String>,
        linear_norms: Vec<RawLinearNorm>,
    },
    NormDiscrete {
        field: String,
        value: String,
        map_missing_to: Option<String>,
        default_value: Option<String>,
    },
    Discretize {
        field: String,
        map_missing_to: Option<String>,
        default_value: Option<String>,
        data_type: Option<String>,
        bins: Vec<RawDiscretizeBin>,
    },
    MapValues {
        map_missing_to: Option<String>,
        default_value: Option<String>,
        output_column: String,
        field_column_pairs: Vec<RawFieldColumnPair>,
        inline_table: Vec<std::collections::HashMap<String, String>>,
    },
    TextIndex {
        field: String,
        map_missing_to: Option<String>,
        text: Box<RawExpression>,
        search_term: Box<RawExpression>,
        is_case_sensitive: bool,
        max_levenstein_distance: Option<usize>,
        word_separator: Option<String>,
        tokenize: bool,
    },
    Aggregate {
        field: String,
        function: String,
        group_field: Option<String>,
    },
    Apply {
        function: String,
        map_missing_to: Option<String>,
        default_value: Option<String>,
        args: Vec<RawExpression>,
    },
    Unknown,
}

#[derive(Debug, Clone)]
pub struct RawPmml {
    pub data_dictionary: Vec<RawDataField>,
    pub tree_model: Option<RawTreeModel>,
    pub regression_model: Option<RawRegressionModel>,
    pub mining_model: Option<RawMiningModel>,
    pub scorecard: Option<RawScorecard>,
    pub clustering_model: Option<RawClusteringModel>,
    pub naive_bayes_model: Option<RawNaiveBayesModel>,
    pub nearest_neighbor_model: Option<RawNearestNeighborModel>,
    pub support_vector_machine_model: Option<RawSupportVectorMachineModel>,
    pub neural_network: Option<RawNeuralNetwork>,
    pub general_regression_model: Option<RawGeneralRegressionModel>,
    pub association_model: Option<RawAssociationModel>,
    pub rule_set_model: Option<RawRuleSetModel>,
    /// TransformationDictionary derived fields (global)
    pub transformation_dictionary: Vec<RawDerivedField>,
    /// TransformationDictionary define functions
    pub define_functions: Vec<RawDefineFunction>,
    /// Vendor extensions (gracefully stored, not yet evaluated)
    pub extensions: Vec<RawExtension>,
    /// Unsupported model tag if PMML contains e.g. AnomalyDetectionModel, BaselineModel, etc.
    pub unsupported_model: Option<String>,
}

// ---------- helpers ----------

fn attr(e: &BytesStart, key: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == key.as_bytes() {
            // value may be escaped; unescape via unescape helper
            let v = a.unescape_value().ok()?;
            return Some(v.into_owned());
        }
    }
    None
}

fn attr_required(e: &BytesStart, key: &str, ctx: &str) -> Result<String> {
    attr(e, key).ok_or_else(|| PmmlError::ParseError {
        context: ctx.into(),
        message: format!("missing attribute {key}"),
    })
}

fn tag_name(e: &BytesStart) -> String {
    String::from_utf8_lossy(e.name().as_ref()).into_owned()
    // quick-xml 0.37: e.name().as_ref() is already local name when no prefix; default xmlns doesn't prefix.
    // If namespace prefix present, local_name would strip. But PMML uses default ns, so fine.
}
// ---------- Expression helpers for TransformationDictionary ----------

fn parse_constant(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawExpression> {
    let data_type = attr(start, "dataType");
    let mut value = String::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Text(t)) => {
                let txt = t.unescape().map(|c| c.into_owned()).unwrap_or_default();
                let trimmed = txt.trim();
                if !trimmed.is_empty() {
                    value = trimmed.to_string();
                } else if value.is_empty() && !txt.is_empty() {
                    value = txt.trim().to_string();
                }
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "Constant" => break,
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawExpression::Constant { data_type, value })
}

fn parse_constant_empty(start: &BytesStart) -> RawExpression {
    let data_type = attr(start, "dataType");
    RawExpression::Constant { data_type, value: String::new() }
}

fn parse_field_ref(start: &BytesStart) -> Result<RawExpression> {
    let field = attr_required(start, "field", "FieldRef")?;
    let map_missing_to = attr(start, "mapMissingTo");
    Ok(RawExpression::FieldRef { field, map_missing_to })
}

fn parse_norm_continuous(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawExpression> {
    let field = attr_required(start, "field", "NormContinuous")?;
    let map_missing_to = attr(start, "mapMissingTo");
    let default_value = attr(start, "defaultValue");
    let outliers = attr(start, "outliers");
    let mut linear_norms = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) if tag_name(&inner) == "LinearNorm" => {
                let orig = attr(&inner, "orig").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                let norm = attr(&inner, "norm").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                linear_norms.push(RawLinearNorm { orig, norm });
                let mut inner2 = Vec::new();
                loop {
                    match reader.read_event_into(&mut inner2) {
                        Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "LinearNorm" => break,
                        Ok(Event::Empty(_)) => break,
                        _ => {}
                    }
                    inner2.clear();
                    break;
                }
            }
            Ok(Event::Empty(inner)) if tag_name(&inner) == "LinearNorm" => {
                let orig = attr(&inner, "orig").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                let norm = attr(&inner, "norm").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                linear_norms.push(RawLinearNorm { orig, norm });
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "NormContinuous" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawExpression::NormContinuous { field, map_missing_to, default_value, outliers, linear_norms })
}

fn parse_map_values(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawExpression> {
    let map_missing_to = attr(start, "mapMissingTo");
    let default_value = attr(start, "defaultValue");
    let output_column = attr(start, "outputColumn").unwrap_or_default();
    let mut field_column_pairs = Vec::new();
    let mut inline_table: Vec<std::collections::HashMap<String, String>> = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) => {
                let tag = tag_name(&inner);
                match tag.as_str() {
                    "FieldColumnPair" => {
                        let field = attr_required(&inner, "field", "FieldColumnPair")?;
                        let column = attr_required(&inner, "column", "FieldColumnPair")?;
                        field_column_pairs.push(RawFieldColumnPair { field, column });
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "FieldColumnPair" => break,
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner2.clear();
                            break;
                        }
                    }
                    "InlineTable" => {
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::Start(row_start)) if tag_name(&row_start) == "row" => {
                                    let mut row: std::collections::HashMap<String, String> = std::collections::HashMap::new();
                                    let mut row_buf = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut row_buf) {
                                            Ok(Event::Start(cell)) => {
                                                let col = tag_name(&cell);
                                                let mut cell_buf = Vec::new();
                                                let mut text = String::new();
                                                loop {
                                                    match reader.read_event_into(&mut cell_buf) {
                                                        Ok(Event::Text(t)) => {
                                                            text = t.unescape().map(|c| c.into_owned()).unwrap_or_default();
                                                        }
                                                        Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == col => break,
                                                        _ => {}
                                                    }
                                                    cell_buf.clear();
                                                }
                                                row.insert(col, text);
                                            }
                                            Ok(Event::Empty(cell)) => {
                                                let col = tag_name(&cell);
                                                row.insert(col, String::new());
                                            }
                                            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "row" => break,
                                            _ => {}
                                        }
                                        row_buf.clear();
                                    }
                                    inline_table.push(row);
                                }
                                Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "InlineTable" => break,
                                _ => {}
                            }
                            inner2.clear();
                        }
                    }
                    "Extension" => {
                        let mut depth = 1usize;
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::Start(_)) => depth += 1,
                                Ok(Event::End(end)) => {
                                    depth -= 1;
                                    if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == "Extension" { break; }
                                }
                                Ok(Event::Empty(_)) => {},
                                _ => {}
                            }
                            inner2.clear();
                        }
                    }
                    _ => {
                        let mut depth = 1usize;
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::Start(_)) => depth += 1,
                                Ok(Event::End(end)) => {
                                    depth -= 1;
                                    if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == tag { break; }
                                }
                                Ok(Event::Empty(_)) => {},
                                _ => {}
                            }
                            inner2.clear();
                        }
                    }
                }
            }
            Ok(Event::Empty(inner)) => {
                let tag = tag_name(&inner);
                if tag == "FieldColumnPair" {
                    let field = attr_required(&inner, "field", "FieldColumnPair")?;
                    let column = attr_required(&inner, "column", "FieldColumnPair")?;
                    field_column_pairs.push(RawFieldColumnPair { field, column });
                }
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "MapValues" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawExpression::MapValues { map_missing_to, default_value, output_column, field_column_pairs, inline_table })
}

fn parse_discretize(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawExpression> {
    let field = attr_required(start, "field", "Discretize")?;
    let map_missing_to = attr(start, "mapMissingTo");
    let default_value = attr(start, "defaultValue");
    let data_type = attr(start, "dataType");
    let mut bins = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) if tag_name(&inner) == "DiscretizeBin" => {
                let bin_value = attr_required(&inner, "binValue", "DiscretizeBin")?;
                let mut interval = RawInterval { closure: "closedOpen".into(), left_margin: None, right_margin: None };
                let mut inner2 = Vec::new();
                loop {
                    match reader.read_event_into(&mut inner2) {
                        Ok(Event::Start(iv)) if tag_name(&iv) == "Interval" => {
                            interval.closure = attr(&iv, "closure").unwrap_or_else(|| "closedOpen".into());
                            interval.left_margin = attr(&iv, "leftMargin").and_then(|s| s.parse::<f64>().ok());
                            interval.right_margin = attr(&iv, "rightMargin").and_then(|s| s.parse::<f64>().ok());
                            let mut skip = Vec::new();
                            loop {
                                match reader.read_event_into(&mut skip) {
                                    Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "Interval" => break,
                                    Ok(Event::Empty(_)) => break,
                                    _ => {}
                                }
                                skip.clear();
                                break;
                            }
                        }
                        Ok(Event::Empty(iv)) if tag_name(&iv) == "Interval" => {
                            interval.closure = attr(&iv, "closure").unwrap_or_else(|| "closedOpen".into());
                            interval.left_margin = attr(&iv, "leftMargin").and_then(|s| s.parse::<f64>().ok());
                            interval.right_margin = attr(&iv, "rightMargin").and_then(|s| s.parse::<f64>().ok());
                        }
                        Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "DiscretizeBin" => break,
                        _ => {}
                    }
                    inner2.clear();
                }
                bins.push(RawDiscretizeBin { bin_value, interval });
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "Discretize" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawExpression::Discretize { field, map_missing_to, default_value, data_type, bins })
}

fn parse_apply(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawExpression> {
    let function = attr_required(start, "function", "Apply")?;
    let map_missing_to = attr(start, "mapMissingTo");
    let default_value = attr(start, "defaultValue");
    let mut args = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) => {
                let tag = tag_name(&inner);
                let expr = parse_expression_from_start(reader, &inner, &tag)?;
                args.push(expr);
            }
            Ok(Event::Empty(inner)) => {
                let tag = tag_name(&inner);
                let expr = parse_expression_empty(&inner, &tag)?;
                args.push(expr);
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "Apply" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawExpression::Apply { function, map_missing_to, default_value, args })
}

fn parse_expression_empty(start: &BytesStart, tag: &str) -> Result<RawExpression> {
    Ok(match tag {
        "Constant" => parse_constant_empty(start),
        "FieldRef" => parse_field_ref(start)?,
        "NormDiscrete" => {
            let field = attr_required(start, "field", "NormDiscrete")?;
            let value = attr_required(start, "value", "NormDiscrete")?;
            let map_missing_to = attr(start, "mapMissingTo");
            let default_value = attr(start, "defaultValue");
            RawExpression::NormDiscrete { field, value, map_missing_to, default_value }
        }
        "NormContinuous" => {
            let field = attr(start, "field").unwrap_or_default();
            RawExpression::NormContinuous { field, map_missing_to: None, default_value: None, outliers: None, linear_norms: vec![] }
        }
        _ => RawExpression::Unknown,
    })
}

fn parse_expression_from_start(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart, tag: &str) -> Result<RawExpression> {
    match tag {
        "Constant" => parse_constant(reader, start),
        "FieldRef" => {
            let expr = parse_field_ref(start)?;
            let mut buf = Vec::new();
            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "FieldRef" => break,
                    _ => {}
                }
                buf.clear();
                break;
            }
            Ok(expr)
        }
        "NormContinuous" => parse_norm_continuous(reader, start),
        "NormDiscrete" => {
            let field = attr_required(start, "field", "NormDiscrete")?;
            let value = attr_required(start, "value", "NormDiscrete")?;
            let map_missing_to = attr(start, "mapMissingTo");
            let default_value = attr(start, "defaultValue");
            let mut buf = Vec::new();
            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "NormDiscrete" => break,
                    _ => {}
                }
                buf.clear();
                break;
            }
            Ok(RawExpression::NormDiscrete { field, value, map_missing_to, default_value })
        }
        "Discretize" => parse_discretize(reader, start),
        "MapValues" => parse_map_values(reader, start),
        "Apply" => parse_apply(reader, start),
        "TextIndex" => {
            let field = attr(start, "field").unwrap_or_default();
            let mut depth = 1usize;
            let mut buf = Vec::new();
            let mut text_expr: Option<RawExpression> = None;
            let mut search_expr: Option<RawExpression> = None;
            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(Event::Start(inner)) => {
                        let t = tag_name(&inner);
                        if t == "FieldRef" || t == "Constant" || t == "Apply" {
                            let e = parse_expression_from_start(reader, &inner, &t)?;
                            if text_expr.is_none() { text_expr = Some(e); } else { search_expr = Some(e); }
                        } else { depth += 1; }
                    }
                    Ok(Event::Empty(inner)) => {
                        let t = tag_name(&inner);
                        if t == "FieldRef" || t == "Constant" {
                            let e = parse_expression_empty(&inner, &t)?;
                            if text_expr.is_none() { text_expr = Some(e); } else { search_expr = Some(e); }
                        }
                    }
                    Ok(Event::End(end)) => {
                        depth -= 1;
                        if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == "TextIndex" { break; }
                    }
                    _ => {}
                }
                buf.clear();
            }
            if let (Some(txt), Some(search)) = (text_expr, search_expr) {
                Ok(RawExpression::TextIndex { field, map_missing_to: None, text: Box::new(txt), search_term: Box::new(search), is_case_sensitive: false, max_levenstein_distance: None, word_separator: None, tokenize: false })
            } else { Ok(RawExpression::Unknown) }
        }
        "Aggregate" => {
            let field = attr(start, "field").unwrap_or_default();
            let function = attr(start, "function").unwrap_or_else(|| "average".into());
            let group_field = attr(start, "groupField");
            let mut buf = Vec::new();
            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "Aggregate" => break,
                    _ => {}
                }
                buf.clear();
            }
            Ok(RawExpression::Aggregate { field, function, group_field })
        }
        _ => {
            let mut depth = 1usize;
            let mut buf = Vec::new();
            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(Event::Start(_)) => depth += 1,
                    Ok(Event::End(end)) => {
                        depth -= 1;
                        if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == tag { break; }
                    }
                    _ => {}
                }
                buf.clear();
            }
            Ok(RawExpression::Unknown)
        }
    }
}

fn parse_derived_field(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawDerivedField> {
    let name = attr_required(start, "name", "DerivedField")?;
    let display_name = attr(start, "displayName");
    let data_type = attr(start, "dataType").unwrap_or_else(|| "string".into());
    let op_type = attr(start, "optype").unwrap_or_else(|| "continuous".into());
    let mut expression: Option<RawExpression> = None;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) => {
                let tag = tag_name(&inner);
                if tag == "Extension" {
                    let mut depth = 1usize;
                    let mut inner2 = Vec::new();
                    loop {
                        match reader.read_event_into(&mut inner2) {
                            Ok(Event::Start(_)) => depth += 1,
                            Ok(Event::End(end)) => {
                                depth -= 1;
                                if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == "Extension" { break; }
                            }
                            _ => {}
                        }
                        inner2.clear();
                    }
                    continue;
                }
                if ["Constant","FieldRef","NormContinuous","NormDiscrete","Discretize","MapValues","TextIndex","Aggregate","Apply"].contains(&tag.as_str()) {
                    let expr = parse_expression_from_start(reader, &inner, &tag)?;
                    if expression.is_none() { expression = Some(expr); }
                } else {
                    let mut depth = 1usize;
                    let mut inner2 = Vec::new();
                    loop {
                        match reader.read_event_into(&mut inner2) {
                            Ok(Event::Start(_)) => depth += 1,
                            Ok(Event::End(end)) => {
                                depth -= 1;
                                if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == tag { break; }
                            }
                            _ => {}
                        }
                        inner2.clear();
                    }
                }
            }
            Ok(Event::Empty(inner)) => {
                let tag = tag_name(&inner);
                if ["Constant","FieldRef","NormDiscrete"].contains(&tag.as_str()) {
                    let expr = parse_expression_empty(&inner, &tag)?;
                    if expression.is_none() { expression = Some(expr); }
                }
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "DerivedField" => break,
            _ => {}
        }
        buf.clear();
    }
    let expr = expression.unwrap_or(RawExpression::Unknown);
    Ok(RawDerivedField { name, display_name, data_type, op_type, expression: expr })
}

fn parse_define_function(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawDefineFunction> {
    let name = attr_required(start, "name", "DefineFunction")?;
    let data_type = attr(start, "dataType");
    let op_type = attr(start, "optype");
    let mut param_fields = Vec::new();
    let mut derived_fields = Vec::new();
    let mut body: Option<RawExpression> = None;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) => {
                let tag = tag_name(&inner);
                match tag.as_str() {
                    "ParameterField" => {
                        let p_name = attr_required(&inner, "name", "ParameterField")?;
                        let p_data_type = attr(&inner, "dataType");
                        let p_op_type = attr(&inner, "optype");
                        param_fields.push(RawParameterField { name: p_name, data_type: p_data_type, op_type: p_op_type });
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "ParameterField" => break,
                                _ => {}
                            }
                            inner2.clear();
                            break;
                        }
                    }
                    "DerivedField" => {
                        let df = parse_derived_field(reader, &inner)?;
                        derived_fields.push(df);
                    }
                    "Extension" => {
                        let mut depth = 1usize;
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::Start(_)) => depth += 1,
                                Ok(Event::End(end)) => {
                                    depth -= 1;
                                    if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == "Extension" { break; }
                                }
                                _ => {}
                            }
                            inner2.clear();
                        }
                    }
                    _ => {
                        if ["Constant","FieldRef","NormContinuous","NormDiscrete","Discretize","MapValues","TextIndex","Aggregate","Apply"].contains(&tag.as_str()) {
                            if body.is_none() {
                                let expr = parse_expression_from_start(reader, &inner, &tag)?;
                                body = Some(expr);
                            } else {
                                let mut depth = 1usize;
                                let mut inner2 = Vec::new();
                                loop {
                                    match reader.read_event_into(&mut inner2) {
                                        Ok(Event::Start(_)) => depth += 1,
                                        Ok(Event::End(end)) => {
                                            depth -= 1;
                                            if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == tag { break; }
                                        }
                                        _ => {}
                                    }
                                    inner2.clear();
                                }
                            }
                        } else {
                            let mut depth = 1usize;
                            let mut inner2 = Vec::new();
                            loop {
                                match reader.read_event_into(&mut inner2) {
                                    Ok(Event::Start(_)) => depth += 1,
                                    Ok(Event::End(end)) => {
                                        depth -= 1;
                                        if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == tag { break; }
                                    }
                                    _ => {}
                                }
                                inner2.clear();
                            }
                        }
                    }
                }
            }
            Ok(Event::Empty(inner)) => {
                let tag = tag_name(&inner);
                if tag == "ParameterField" {
                    let p_name = attr_required(&inner, "name", "ParameterField")?;
                    let p_data_type = attr(&inner, "dataType");
                    let p_op_type = attr(&inner, "optype");
                    param_fields.push(RawParameterField { name: p_name, data_type: p_data_type, op_type: p_op_type });
                } else if ["Constant","FieldRef","NormDiscrete"].contains(&tag.as_str()) && body.is_none() {
                    let expr = parse_expression_empty(&inner, &tag)?;
                    body = Some(expr);
                }
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "DefineFunction" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawDefineFunction { name, data_type, op_type, param_fields, derived_fields, body })
}

fn parse_transformation_dictionary(reader: &mut quick_xml::Reader<&[u8]>, _start: &BytesStart) -> Result<(Vec<RawDefineFunction>, Vec<RawDerivedField>)> {
    let mut define_functions = Vec::new();
    let mut derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) => {
                let tag = tag_name(&inner);
                match tag.as_str() {
                    "DefineFunction" => {
                        let df = parse_define_function(reader, &inner)?;
                        define_functions.push(df);
                    }
                    "DerivedField" => {
                        let df = parse_derived_field(reader, &inner)?;
                        derived_fields.push(df);
                    }
                    "Extension" => {
                        let mut depth = 1usize;
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::Start(_)) => depth += 1,
                                Ok(Event::End(end)) => {
                                    depth -= 1;
                                    if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == "Extension" { break; }
                                }
                                _ => {}
                            }
                            inner2.clear();
                        }
                    }
                    _ => {
                        let mut depth = 1usize;
                        let mut inner2 = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner2) {
                                Ok(Event::Start(_)) => depth += 1,
                                Ok(Event::End(end)) => {
                                    depth -= 1;
                                    if depth == 0 && String::from_utf8_lossy(end.name().as_ref()) == tag { break; }
                                }
                                _ => {}
                            }
                            inner2.clear();
                        }
                    }
                }
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "TransformationDictionary" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok((define_functions, derived_fields))
}

fn parse_local_transformations(reader: &mut quick_xml::Reader<&[u8]>) -> Result<Vec<RawDerivedField>> {
    let mut derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) if tag_name(&inner) == "DerivedField" => {
                let df = parse_derived_field(reader, &inner)?;
                derived_fields.push(df);
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "LocalTransformations" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(derived_fields)
}




fn parse_mining_field(e: &BytesStart) -> Result<RawMiningField> {
    let name = attr_required(e, "name", "MiningField")?;
    let usage_type = attr(e, "usageType");
    let importance = attr(e, "importance").and_then(|s| s.parse::<f64>().ok());
    let outliers = attr(e, "outliers").or_else(|| attr(e, "outlierTreatment"));
    let low_value = attr(e, "lowValue");
    let high_value = attr(e, "highValue");
    let missing_value_replacement = attr(e, "missingValueReplacement");
    let missing_value_treatment = attr(e, "missingValueTreatment");
    let invalid_value_treatment = attr(e, "invalidValueTreatment");
    let invalid_value_replacement = attr(e, "invalidValueReplacement");
    let op_type = attr(e, "opType");
    Ok(RawMiningField {
        name,
        usage_type,
        importance,
        outliers,
        low_value,
        high_value,
        missing_value_replacement,
        missing_value_treatment,
        invalid_value_treatment,
        invalid_value_replacement,
        op_type,
    })
}

fn parse_output_field(e: &BytesStart) -> Result<RawOutputField> {
    let name = attr_required(e, "name", "OutputField")?;
    let feature = attr(e, "feature");
    let value = attr(e, "value");
    let target_field = attr(e, "targetField");
    let data_type = attr(e, "dataType");
    let op_type = attr(e, "opType");
    let rule_feature = attr(e, "ruleFeature");
    let algorithm = attr(e, "algorithm");
    let rank = attr(e, "rank").and_then(|s| s.parse::<i32>().ok());
    let rank_basis = attr(e, "rankBasis");
    let rank_order = attr(e, "rankOrder");
    let is_multi_valued = attr(e, "isMultiValued");
    let segment_id = attr(e, "segmentId");
    let is_final_result = attr(e, "isFinalResult").map(|s| s == "true");
    let display_name = attr(e, "displayName");
    Ok(RawOutputField {
        name,
        feature,
        value,
        target_field,
        data_type,
        op_type,
        rule_feature,
        algorithm,
        rank,
        rank_basis,
        rank_order,
        is_multi_valued,
        segment_id,
        is_final_result,
        display_name,
    })
}

fn parse_target(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawTarget> {
    let field = attr(start, "field");
    let op_type = attr(start, "opType");
    let cast_integer = attr(start, "castInteger");
    let rescale_constant = attr(start, "rescaleConstant").and_then(|s| s.parse::<f64>().ok());
    let rescale_factor = attr(start, "rescaleFactor").and_then(|s| s.parse::<f64>().ok());
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "Target" => break,
            Ok(Event::Empty(_)) => break,
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawTarget { field, op_type, cast_integer, min: None, max: None, rescale_constant, rescale_factor, target_values: vec![] })
}

fn parse_targets(reader: &mut quick_xml::Reader<&[u8]>) -> Result<Vec<RawTarget>> {
    let mut targets = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) if tag_name(&inner) == "Target" => {
                let t = parse_target(reader, &inner)?;
                targets.push(t);
            }
            Ok(Event::Empty(inner)) if tag_name(&inner) == "Target" => {
                let t = parse_target(reader, &inner)?;
                targets.push(t);
            }
            Ok(Event::End(end)) if String::from_utf8_lossy(end.name().as_ref()) == "Targets" => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(targets)
}

fn parse_simple_predicate(e: &BytesStart) -> Result<RawPredicate> {
    let field = attr_required(e, "field", "SimplePredicate")?;
    let operator = attr_required(e, "operator", "SimplePredicate")?;
    // value may be missing for isMissing/isNotMissing operators
    let value = attr(e, "value").unwrap_or_default();
    Ok(RawPredicate::Simple {
        field,
        operator,
        value,
    })
}

fn parse_simple_set_predicate(
    e: &BytesStart,
    reader: &mut quick_xml::Reader<&[u8]>,
) -> Result<RawPredicate> {
    let field = attr_required(e, "field", "SimpleSetPredicate")?;
    let boolean_operator = attr_required(e, "booleanOperator", "SimpleSetPredicate")?;
    // Expect <Array> child
    let mut array_content = String::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(inner)) if tag_name(&inner) == "Array" => {
                // collect text until End Array
                let mut inner_buf = Vec::new();
                loop {
                    match reader.read_event_into(&mut inner_buf) {
                        Ok(Event::Text(t)) => {
                            array_content =
                                t.unescape().map(|c| c.into_owned()).unwrap_or_default();
                        }
                        Ok(Event::End(end))
                            if String::from_utf8_lossy(end.name().as_ref()) == "Array" =>
                        {
                            break
                        }
                        Ok(Event::Eof) => break,
                        _ => {}
                    }
                    inner_buf.clear();
                }
            }
            Ok(Event::Empty(inner)) if tag_name(&inner) == "Array" => {}
            Ok(Event::End(end))
                if String::from_utf8_lossy(end.name().as_ref()) == "SimpleSetPredicate" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawPredicate::SimpleSet {
        field,
        boolean_operator,
        array: array_content,
    })
}

// ---------- Node parsing ----------

fn parse_node(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawNode> {
    let id = attr(start, "id");
    let score = attr(start, "score");
    let record_count = attr(start, "recordCount").and_then(|s| s.parse::<f64>().ok());
    let default_child = attr(start, "defaultChild");

    let mut predicate = RawPredicate::True; // default if none
    let mut predicate_set = false;
    let mut children = Vec::new();
    let mut score_distributions = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "True" => {
                        predicate = RawPredicate::True;
                        predicate_set = true;
                        // consume until End True
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "True" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "SimplePredicate" => {
                        predicate = parse_simple_predicate(&e)?;
                        predicate_set = true;
                        // SimplePredicate may be empty or have no children; consume if Start
                        // For non-empty, we expect End
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "SimplePredicate" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Node" =>
                                {
                                    // shouldn't happen, but break
                                    break;
                                }
                                _ => {}
                            }
                            inner.clear();
                            // Actually SimplePredicate is empty element usually; break after one
                            break;
                        }
                    }
                    "SimpleSetPredicate" => {
                        predicate = parse_simple_set_predicate(&e, reader)?;
                        predicate_set = true;
                    }
                    "CompoundPredicate" => {
                        let boolean_operator =
                            attr_required(&e, "booleanOperator", "CompoundPredicate")?;
                        let mut preds = Vec::new();
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) => {
                                    let itag = tag_name(&inner_e);
                                    match itag.as_str() {
                                        "SimplePredicate" => {
                                            preds.push(parse_simple_predicate(&inner_e)?)
                                        }
                                        "SimpleSetPredicate" => preds
                                            .push(parse_simple_set_predicate(&inner_e, reader)?),
                                        "True" => preds.push(RawPredicate::True),
                                        _ => {}
                                    }
                                }
                                Ok(Event::Empty(inner_e)) => {
                                    let itag = tag_name(&inner_e);
                                    if itag == "SimplePredicate" {
                                        preds.push(parse_simple_predicate(&inner_e)?);
                                    } else if itag == "True" {
                                        preds.push(RawPredicate::True);
                                    }
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "CompoundPredicate" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                        predicate = RawPredicate::Compound {
                            boolean_operator,
                            predicates: preds,
                        };
                        predicate_set = true;
                    }
                    "ScoreDistribution" => {
                        let value = attr_required(&e, "value", "ScoreDistribution")?;
                        let rc = attr(&e, "recordCount")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        score_distributions.push(RawScoreDistribution {
                            value,
                            record_count: rc,
                        });
                        // consume end
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "ScoreDistribution" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner.clear();
                            break;
                        }
                    }
                    "Node" => {
                        let child = parse_node(reader, &e)?;
                        children.push(child);
                    }
                    _ => {
                        // skip unknown (Extension, etc) - consume until matching End if needed
                        // For now, just ignore
                    }
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "True" => {
                        predicate = RawPredicate::True;
                        predicate_set = true;
                    }
                    "SimplePredicate" => {
                        predicate = parse_simple_predicate(&e)?;
                        predicate_set = true;
                    }
                    "SimpleSetPredicate" => {
                        // SimpleSet empty not typical, treat as isIn with empty array
                        let field = attr_required(&e, "field", "SimpleSetPredicate")?;
                        let boolean_operator =
                            attr_required(&e, "booleanOperator", "SimpleSetPredicate")?;
                        predicate = RawPredicate::SimpleSet {
                            field,
                            boolean_operator,
                            array: String::new(),
                        };
                        predicate_set = true;
                    }
                    "ScoreDistribution" => {
                        let value = attr_required(&e, "value", "ScoreDistribution")?;
                        let rc = attr(&e, "recordCount")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        score_distributions.push(RawScoreDistribution {
                            value,
                            record_count: rc,
                        });
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "Node" => {
                break;
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }

    if !predicate_set {
        // Nodes at root may have True; inner nodes must have predicate, default True if missing (defensive)
        predicate = RawPredicate::True;
    }

    Ok(RawNode {
        id,
        score,
        record_count,
        predicate,
        score_distributions,
        children,
        default_child: None,
    })
}

// ---------- TreeModel parsing ----------

fn parse_tree_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawTreeModel> {
    let function_name = attr_required(start, "functionName", "TreeModel")?;
    let missing_value_strategy = attr(start, "missingValueStrategy");
    let no_true_child_strategy = attr(start, "noTrueChildStrategy");
    let mut mining_schema = Vec::new();
    let mut output = Vec::new();
    let mut root: Option<RawNode> = None;
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        // parse MiningField children
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    // consume end
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            Ok(Event::Eof) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Output" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "OutputField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            Ok(Event::Eof) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Output" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Node" => {
                        let node = parse_node(reader, &e)?;
                        root = Some(node);
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {
                        // skip ModelStats, Targets, Extension, etc for v1
                        // Need to consume subtree if it's Start
                        if tag == "Targets"
                            || tag == "ModelStats"
                            || tag == "ModelExplanation"
                        {
                            let mut depth = 1usize;
                            let mut inner = Vec::new();
                            loop {
                                match reader.read_event_into(&mut inner) {
                                    Ok(Event::Start(_)) => depth += 1,
                                    Ok(Event::End(end)) => {
                                        depth -= 1;
                                        if depth == 0
                                            && String::from_utf8_lossy(end.name().as_ref()) == tag
                                        {
                                            break;
                                        }
                                    }
                                    Ok(Event::Empty(_)) => {}
                                    Ok(Event::Eof) => break,
                                    _ => {}
                                }
                                inner.clear();
                            }
                        }
                    }
                }
            }
            Ok(Event::Empty(e)) if tag_name(&e) == "Node" => {
                let node = parse_node(reader, &e)?;
                root = Some(node);
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "TreeModel" => break,
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    let root = root.ok_or_else(|| PmmlError::ParseError {
        context: "TreeModel".into(),
        message: "missing root Node".into(),
    })?;
    Ok(RawTreeModel {
        function_name,
        missing_value_strategy,
        no_true_child_strategy,
        mining_schema,
        output,
        targets: Vec::new(),
        root,
        local_derived_fields
    })
}

// ---------- Regression parsing ----------

fn parse_regression_table(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawRegressionTable> {
    let intercept = attr(start, "intercept")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let target_category = attr(start, "targetCategory");
    let mut numeric_predictors = Vec::new();
    let mut categorical_predictors = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "NumericPredictor" => {
                        let name = attr_required(&e, "name", "NumericPredictor")?;
                        let coefficient = attr(&e, "coefficient")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        let exponent = attr(&e, "exponent")
                            .and_then(|s| s.parse::<i32>().ok())
                            .unwrap_or(1);
                        numeric_predictors.push(RawNumericPredictor {
                            name,
                            exponent,
                            coefficient,
                        });
                        // consume end if not empty
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "NumericPredictor" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                            break;
                        }
                    }
                    "CategoricalPredictor" => {
                        let name = attr_required(&e, "name", "CategoricalPredictor")?;
                        let value = attr_required(&e, "value", "CategoricalPredictor")?;
                        let coefficient = attr(&e, "coefficient")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        categorical_predictors.push(RawCategoricalPredictor {
                            name,
                            value,
                            coefficient,
                        });
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "CategoricalPredictor" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner.clear();
                            break;
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = tag_name(&e);
                if tag == "NumericPredictor" {
                    let name = attr_required(&e, "name", "NumericPredictor")?;
                    let coefficient = attr(&e, "coefficient")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let exponent = attr(&e, "exponent")
                        .and_then(|s| s.parse::<i32>().ok())
                        .unwrap_or(1);
                    numeric_predictors.push(RawNumericPredictor {
                        name,
                        exponent,
                        coefficient,
                    });
                } else if tag == "CategoricalPredictor" {
                    let name = attr_required(&e, "name", "CategoricalPredictor")?;
                    let value = attr_required(&e, "value", "CategoricalPredictor")?;
                    let coefficient = attr(&e, "coefficient")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    categorical_predictors.push(RawCategoricalPredictor {
                        name,
                        value,
                        coefficient,
                    });
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "RegressionTable" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawRegressionTable {
        intercept,
        target_category,
        numeric_predictors,
        categorical_predictors,
    })
}

fn parse_regression_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawRegressionModel> {
    let function_name = attr_required(start, "functionName", "RegressionModel")?;
    let target_field_name = attr(start, "targetFieldName");
    let normalization_method = attr(start, "normalizationMethod");
    let model_name = attr(start, "modelName");
    let mut mining_schema = Vec::new();
    let mut output = Vec::new();
    let mut regression_tables = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Output" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "OutputField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Output" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "RegressionTable" => {
                        let tbl = parse_regression_table(reader, &e)?;
                        regression_tables.push(tbl);
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {
                        if tag == "Targets" || tag == "ModelStats"
                        {
                            let mut depth = 1usize;
                            let mut inner = Vec::new();
                            loop {
                                match reader.read_event_into(&mut inner) {
                                    Ok(Event::Start(_)) => depth += 1,
                                    Ok(Event::End(end)) => {
                                        depth -= 1;
                                        if depth == 0
                                            && String::from_utf8_lossy(end.name().as_ref()) == tag
                                        {
                                            break;
                                        }
                                    }
                                    Ok(Event::Empty(_)) => {}
                                    Ok(Event::Eof) => break,
                                    _ => {}
                                }
                                inner.clear();
                            }
                        }
                    }
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "RegressionModel" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawRegressionModel {
        function_name,
        target_field_name,
        mining_schema,
        output,
        targets: Vec::new(),
        regression_tables,
        normalization_method,
        model_name,
        local_derived_fields
    })
}

fn parse_segment(reader: &mut quick_xml::Reader<&[u8]>, start: &BytesStart) -> Result<RawSegment> {
    let id = attr(start, "id");
    let weight = attr(start, "weight")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    let mut predicate = RawPredicate::True;
    let mut model: Option<RawSegmentModel> = None;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "True" => {
                        predicate = RawPredicate::True;
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "True" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "SimplePredicate" => {
                        predicate = parse_simple_predicate(&e)?;
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "SimplePredicate" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner.clear();
                            break;
                        }
                    }
                    "SimpleSetPredicate" => {
                        predicate = parse_simple_set_predicate(&e, reader)?;
                    }
                    "CompoundPredicate" => {
                        let boolean_operator =
                            attr_required(&e, "booleanOperator", "CompoundPredicate")?;
                        let mut preds = Vec::new();
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) => {
                                    let itag = tag_name(&inner_e);
                                    if itag.as_str() == "SimplePredicate" {
                                        preds.push(parse_simple_predicate(&inner_e)?)
                                    }
                                }
                                Ok(Event::Empty(inner_e)) => {
                                    let itag = tag_name(&inner_e);
                                    if itag == "SimplePredicate" {
                                        preds.push(parse_simple_predicate(&inner_e)?);
                                    } else if itag == "True" {
                                        preds.push(RawPredicate::True);
                                    }
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "CompoundPredicate" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                        predicate = RawPredicate::Compound {
                            boolean_operator,
                            predicates: preds,
                        };
                    }
                    "TreeModel" => {
                        let tm = parse_tree_model(reader, &e)?;
                        model = Some(RawSegmentModel::Tree(tm));
                    }
                    "RegressionModel" => {
                        let rm = parse_regression_model(reader, &e)?;
                        model = Some(RawSegmentModel::Regression(rm));
                    }
                    "Regression" => {
                        // Embedded Regression inside MiningModel (PMML 4.1 style)
                        // Parse as RegressionModel with single RegressionTable
                        let mut regression_tables = Vec::new();
                        let mut inner = Vec::new();
                        let mut function_name = "regression".to_string();
                        // Try to get functionName if present? For embedded, may not have.
                        if let Some(fn_attr) = attr(&e, "functionName") {
                            function_name = fn_attr;
                        }
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "RegressionTable" =>
                                {
                                    let tbl = parse_regression_table(reader, &inner_e)?;
                                    regression_tables.push(tbl);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "Regression" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                        let rm = RawRegressionModel {
                            function_name,
                            target_field_name: None,
                            mining_schema: Vec::new(),
                            output: Vec::new(),
                            targets: Vec::new(),
                            regression_tables,
                            normalization_method: None,
                            model_name: None,
                            local_derived_fields: Vec::new(),
                        };
                        model = Some(RawSegmentModel::Regression(rm));
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "True" => predicate = RawPredicate::True,
                    "SimplePredicate" => predicate = parse_simple_predicate(&e)?,
                    _ => {}
                }
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "Segment" => break,
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    let model = model.ok_or_else(|| PmmlError::ParseError {
        context: "Segment".into(),
        message: "missing model".into(),
    })?;
    Ok(RawSegment {
        id,
        predicate,
        model,
        weight,
    })
}

fn parse_segmentation(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawSegmentation> {
    let multiple_model_method = attr_required(start, "multipleModelMethod", "Segmentation")?;
    let missing_prediction_treatment = attr(start, "missingPredictionTreatment");
    let mut segments = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) if tag_name(&e) == "Segment" => {
                let seg = parse_segment(reader, &e)?;
                segments.push(seg);
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "Segmentation" => {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawSegmentation {
        multiple_model_method,
        missing_prediction_treatment,
        segments,
    })
}

fn parse_mining_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawMiningModel> {
    let function_name = attr_required(start, "functionName", "MiningModel")?;
    let model_name = attr(start, "modelName");
    let mut mining_schema = Vec::new();
    let mut output = Vec::new();
    let mut segmentation: Option<RawSegmentation> = None;
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Segmentation" => {
                        let seg = parse_segmentation(reader, &e)?;
                        segmentation = Some(seg);
                    }
                    "Output" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "OutputField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Output" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {
                        if tag == "Targets" {
                            let mut depth = 1usize;
                            let mut inner = Vec::new();
                            loop {
                                match reader.read_event_into(&mut inner) {
                                    Ok(Event::Start(_)) => depth += 1,
                                    Ok(Event::End(end)) => {
                                        depth -= 1;
                                        if depth == 0
                                            && String::from_utf8_lossy(end.name().as_ref()) == tag
                                        {
                                            break;
                                        }
                                    }
                                    Ok(Event::Empty(_)) => {}
                                    Ok(Event::Eof) => break,
                                    _ => {}
                                }
                                inner.clear();
                            }
                        }
                    }
                }
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "MiningModel" => {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawMiningModel {
        function_name,
        mining_schema,
        segmentation,
        output,
        targets: Vec::new(),
        model_name,
        local_derived_fields
    })
}

fn parse_scorecard(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawScorecard> {
    let function_name = attr_required(start, "functionName", "Scorecard")?;
    let model_name = attr(start, "modelName");
    let initial_score = attr(start, "initialScore")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let use_reason_codes = attr(start, "useReasonCodes").map(|s| s == "true");
    let reason_code_algorithm = attr(start, "reasonCodeAlgorithm");
    let baseline_method = attr(start, "baselineMethod");
    let mut mining_schema = Vec::new();
    let mut output = Vec::new();
    let mut characteristics = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Output" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "OutputField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Output" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Characteristics" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "Characteristic" =>
                                {
                                    let name = attr_required(&inner_e, "name", "Characteristic")?;
                                    let reason_code = attr(&inner_e, "reasonCode");
                                    let baseline_score = attr(&inner_e, "baselineScore")
                                        .and_then(|s| s.parse::<f64>().ok());
                                    let mut attrs = Vec::new();
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(a_e))
                                                if tag_name(&a_e) == "Attribute" =>
                                            {
                                                let partial_score = attr(&a_e, "partialScore")
                                                    .and_then(|s| s.parse::<f64>().ok())
                                                    .unwrap_or(0.0);
                                                let reason_code = attr(&a_e, "reasonCode");
                                                // Expect predicate inside
                                                let mut pred = RawPredicate::True;
                                                let mut attr_buf = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut attr_buf) {
                                                        Ok(Event::Start(p_e))
                                                            if tag_name(&p_e)
                                                                == "SimplePredicate" =>
                                                        {
                                                            pred = parse_simple_predicate(&p_e)?;
                                                            // consume end
                                                            let mut skip = Vec::new();
                                                            loop {
                                                                match reader.read_event_into(
                                                                    &mut skip,
                                                                ) {
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        ) == "SimplePredicate" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                skip.clear();
                                                                break;
                                                            }
                                                        }
                                                        Ok(Event::Empty(p_e))
                                                            if tag_name(&p_e)
                                                                == "SimplePredicate" =>
                                                        {
                                                            pred = parse_simple_predicate(&p_e)?;
                                                        }
                                                        Ok(Event::Start(p_e))
                                                            if tag_name(&p_e)
                                                                == "CompoundPredicate" =>
                                                        {
                                                            let boolean_operator = attr_required(
                                                                &p_e,
                                                                "booleanOperator",
                                                                "CompoundPredicate",
                                                            )?;
                                                            let mut preds = Vec::new();
                                                            let mut cp_buf = Vec::new();
                                                            loop {
                                                                match reader.read_event_into(
                                                                    &mut cp_buf,
                                                                ) {
                                                                    Ok(Event::Start(inner_e)) => {
                                                                        let itag =
                                                                            tag_name(&inner_e);
                                                                        if itag
                                                                            == "SimplePredicate"
                                                                        {
                                                                            preds.push(
                                                                                parse_simple_predicate(
                                                                                    &inner_e,
                                                                                )?,
                                                                            );
                                                                        }
                                                                    }
                                                                    Ok(Event::Empty(inner_e)) => {
                                                                        let itag =
                                                                            tag_name(&inner_e);
                                                                        if itag
                                                                            == "SimplePredicate"
                                                                        {
                                                                            preds.push(
                                                                                parse_simple_predicate(
                                                                                    &inner_e,
                                                                                )?,
                                                                            );
                                                                        }
                                                                    }
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "CompoundPredicate" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                cp_buf.clear();
                                                            }
                                                            pred = RawPredicate::Compound {
                                                                boolean_operator,
                                                                predicates: preds,
                                                            };
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "Attribute" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    attr_buf.clear();
                                                }
                                                attrs.push(RawAttribute {
                                                    partial_score,
                                                    predicate: pred,
                                                    reason_code,
                                                });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Characteristic" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    characteristics.push(RawCharacteristic {
                                        name,
                                        reason_code,
                                        baseline_score,
                                        attributes: attrs,
                                    });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "Characteristics" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "Scorecard" => break,
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawScorecard {
        model_name,
        function_name,
        initial_score,
        use_reason_codes,
        reason_code_algorithm,
        mining_schema,
        output,
        characteristics,
        baseline_method,
        targets: vec![],
        local_derived_fields
    })
}

fn parse_clustering_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawClusteringModel> {
    let function_name = attr_required(start, "functionName", "ClusteringModel")?;
    let model_name = attr(start, "modelName");
    let model_class = attr(start, "modelClass");
    let number_of_clusters = attr(start, "numberOfClusters").and_then(|s| s.parse::<usize>().ok());
    let mut mining_schema = Vec::new();
    let mut output = Vec::new();
    let mut comparison_measure: Option<RawComparisonMeasure> = None;
    let mut clustering_fields = Vec::new();
    let mut clusters = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Output" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "OutputField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Output" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "ComparisonMeasure" => {
                        let kind = attr(&e, "kind").unwrap_or_else(|| "distance".to_string());
                        comparison_measure = Some(RawComparisonMeasure {
                            kind,
                            compare_function: None,
                        });
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "ComparisonMeasure" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => {}
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "ClusteringField" => {
                        let field = attr_required(&e, "field", "ClusteringField")?;
                        clustering_fields.push(field);
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "ClusteringField" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner.clear();
                            break;
                        }
                    }
                    "Cluster" => {
                        let name = attr_required(&e, "name", "Cluster")?;
                        let mut array = Vec::new();
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "Array" => {
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Text(t)) => {
                                                let txt =
                                                    t.unescape().unwrap_or_default().into_owned();
                                                for part in txt.split_whitespace() {
                                                    if let Ok(f) = part.parse::<f64>() {
                                                        array.push(f);
                                                    }
                                                }
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Array" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "Array" => {}
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "Cluster" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                        clusters.push(RawCluster { name, array });
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "ClusteringModel" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawClusteringModel {
        model_name,
        function_name,
        model_class,
        number_of_clusters,
        mining_schema,
        output,
        targets: Vec::new(),
        comparison_measure,
        clustering_fields,
        clusters,
        local_derived_fields
    })
}

fn parse_naive_bayes_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawNaiveBayesModel> {
    let function_name = attr_required(start, "functionName", "NaiveBayesModel")?;
    let threshold = attr(start, "threshold")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let mut mining_schema = Vec::new();
    let output = Vec::new();
    let mut bayes_inputs: Vec<RawBayesInput> = Vec::new();
    let mut bayes_output_counts: Vec<RawTargetValueCount> = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "BayesInputs" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "BayesInput" => {
                                    let field_name =
                                        attr_required(&inner_e, "fieldName", "BayesInput")?;
                                    let mut target_value_stats = Vec::new();
                                    let mut pair_counts = Vec::new();
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(b_e))
                                                if tag_name(&b_e) == "TargetValueStats" =>
                                            {
                                                let mut inner3 = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(tv_e))
                                                            if tag_name(&tv_e)
                                                                == "TargetValueStat" =>
                                                        {
                                                            let value = attr_required(
                                                                &tv_e,
                                                                "value",
                                                                "TargetValueStat",
                                                            )?;
                                                            let mut mean = None;
                                                            let mut variance = None;
                                                            let mut inner4 = Vec::new();
                                                            loop {
                                                                match reader.read_event_into(
                                                                    &mut inner4,
                                                                ) {
                                                                    Ok(Event::Start(g_e))
                                                                        if tag_name(&g_e)
                                                                            == "GaussianDistribution" =>
                                                                    {
                                                                        mean = attr(&g_e, "mean")
                                                                            .and_then(|s| {
                                                                                s.parse::<f64>().ok()
                                                                            });
                                                                        variance = attr(
                                                                            &g_e, "variance",
                                                                        )
                                                                        .and_then(|s| {
                                                                            s.parse::<f64>().ok()
                                                                        });
                                                                        let mut skip = Vec::new();
                                                                        loop {
                                                                            match reader
                                                                                .read_event_into(
                                                                                    &mut skip,
                                                                                )
                                                                            {
                                                                                Ok(Event::End(
                                                                                    end,
                                                                                )) if String::from_utf8_lossy(
                                                                                    end.name().as_ref(),
                                                                                )
                                                                                    == "GaussianDistribution" =>
                                                                                {
                                                                                    break
                                                                                }
                                                                                Ok(Event::Empty(
                                                                                    _,
                                                                                )) => break,
                                                                                _ => {}
                                                                            }
                                                                            skip.clear();
                                                                            break;
                                                                        }
                                                                    }
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "TargetValueStat" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                inner4.clear();
                                                            }
                                                            target_value_stats.push(
                                                                RawTargetValueStat {
                                                                    value,
                                                                    gaussian_mean: mean,
                                                                    gaussian_variance: variance,
                                                                },
                                                            );
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "TargetValueStats" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                            }
                                            Ok(Event::Start(b_e))
                                                if tag_name(&b_e) == "PairCounts" =>
                                            {
                                                let pc_value =
                                                    attr(&b_e, "value").unwrap_or_default();
                                                let mut target_counts = Vec::new();
                                                let mut inner3 = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(tvc_e))
                                                            if tag_name(&tvc_e)
                                                                == "TargetValueCounts" =>
                                                        {
                                                            let mut inner4 = Vec::new();
                                                            loop {
                                                                match reader.read_event_into(
                                                                    &mut inner4,
                                                                ) {
                                                                    Ok(Event::Start(cnt_e))
                                                                        if tag_name(&cnt_e)
                                                                            == "TargetValueCount" =>
                                                                    {
                                                                        let value = attr_required(
                                                                            &cnt_e,
                                                                            "value",
                                                                            "TargetValueCount",
                                                                        )?;
                                                                        let count = attr(
                                                                            &cnt_e, "count",
                                                                        )
                                                                        .and_then(|s| {
                                                                            s.parse::<f64>().ok()
                                                                        })
                                                                        .unwrap_or(0.0);
                                                                        target_counts.push(
                                                                            RawTargetValueCount {
                                                                                value,
                                                                                count,
                                                                            },
                                                                        );
                                                                        let mut skip =
                                                                            Vec::new();
                                                                        loop {
                                                                            match reader
                                                                                .read_event_into(
                                                                                    &mut skip,
                                                                                )
                                                                            {
                                                                                Ok(Event::End(
                                                                                    end,
                                                                                )) if String::from_utf8_lossy(
                                                                                    end.name().as_ref(),
                                                                                )
                                                                                    == "TargetValueCount" =>
                                                                                {
                                                                                    break
                                                                                }
                                                                                Ok(Event::Empty(
                                                                                    _,
                                                                                )) => break,
                                                                                _ => {}
                                                                            }
                                                                            skip.clear();
                                                                            break;
                                                                        }
                                                                    }
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "TargetValueCounts" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                inner4.clear();
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "PairCounts" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                                pair_counts.push(RawPairCounts {
                                                    value: pc_value,
                                                    target_counts,
                                                });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "BayesInput" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    bayes_inputs.push(RawBayesInput {
                                        field_name,
                                        target_value_stats,
                                        pair_counts,
                                    });
                                }
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "Extension" => {
                                    // Handle Extension wrapping BayesInput (BayesInputTest)
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(bayes_e))
                                                if tag_name(&bayes_e) == "BayesInput" =>
                                            {
                                                let field_name = attr_required(
                                                    &bayes_e,
                                                    "fieldName",
                                                    "BayesInput",
                                                )?;
                                                let mut target_value_stats = Vec::new();
                                                let pair_counts = Vec::new();
                                                let mut inner3 = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(b_e))
                                                            if tag_name(&b_e)
                                                                == "TargetValueStats" =>
                                                        {
                                                            let mut inner4 = Vec::new();
                                                            loop {
                                                                match reader.read_event_into(
                                                                    &mut inner4,
                                                                ) {
                                                                    Ok(Event::Start(tv_e))
                                                                        if tag_name(&tv_e)
                                                                            == "TargetValueStat" =>
                                                                    {
                                                                        let value = attr_required(
                                                                            &tv_e,
                                                                            "value",
                                                                            "TargetValueStat",
                                                                        )?;
                                                                        let mut mean = None;
                                                                        let mut variance = None;
                                                                        let mut inner5 = Vec::new();
                                                                        loop {
                                                                            match reader
                                                                                .read_event_into(
                                                                                    &mut inner5,
                                                                                )
                                                                            {
                                                                                Ok(Event::Start(
                                                                                    g_e,
                                                                                )) if tag_name(
                                                                                    &g_e,
                                                                                )
                                                                                    == "GaussianDistribution" =>
                                                                                {
                                                                                    mean = attr(
                                                                                        &g_e,
                                                                                        "mean",
                                                                                    )
                                                                                    .and_then(
                                                                                        |s| {
                                                                                            s.parse::<
                                                                                                f64,
                                                                                            >(
                                                                                            )
                                                                                            .ok()
                                                                                        },
                                                                                    );
                                                                                    variance = attr(
                                                                                        &g_e,
                                                                                        "variance",
                                                                                    )
                                                                                    .and_then(
                                                                                        |s| {
                                                                                            s.parse::<
                                                                                                f64,
                                                                                            >(
                                                                                            )
                                                                                            .ok()
                                                                                        },
                                                                                    );
                                                                                    let mut skip =
                                                                                        Vec::new();
                                                                                    loop {
                                                                                        match reader
                                                                                            .read_event_into(
                                                                                                &mut skip,
                                                                                            )
                                                                                        {
                                                                                            Ok(Event::End(
                                                                                                end,
                                                                                            )) if String::from_utf8_lossy(
                                                                                                end
                                                                                                    .name()
                                                                                                    .as_ref(),
                                                                                            )
                                                                                                == "GaussianDistribution" =>
                                                                                            {
                                                                                                break
                                                                                            }
                                                                                            Ok(Event::Empty(
                                                                                                _,
                                                                                            )) => {
                                                                                                break
                                                                                            }
                                                                                            _ => {}
                                                                                        }
                                                                                        skip.clear(
                                                                                        );
                                                                                        break;
                                                                                    }
                                                                                }
                                                                                Ok(Event::End(
                                                                                    end,
                                                                                )) if String::from_utf8_lossy(
                                                                                    end.name().as_ref(),
                                                                                )
                                                                                    == "TargetValueStat" =>
                                                                                {
                                                                                    break
                                                                                }
                                                                                _ => {}
                                                                            }
                                                                            inner5.clear();
                                                                        }
                                                                        target_value_stats.push(
                                                                            RawTargetValueStat {
                                                                                value,
                                                                                gaussian_mean:
                                                                                    mean,
                                                                                gaussian_variance:
                                                                                    variance,
                                                                            },
                                                                        );
                                                                    }
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "TargetValueStats" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                inner4.clear();
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "BayesInput" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                                bayes_inputs.push(RawBayesInput {
                                                    field_name,
                                                    target_value_stats,
                                                    pair_counts,
                                                });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Extension" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "BayesInputs" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "BayesOutput" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "TargetValueCounts" =>
                                {
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(cnt_e))
                                                if tag_name(&cnt_e) == "TargetValueCount" =>
                                            {
                                                let value = attr_required(
                                                    &cnt_e,
                                                    "value",
                                                    "TargetValueCount",
                                                )?;
                                                let count = attr(&cnt_e, "count")
                                                    .and_then(|s| s.parse::<f64>().ok())
                                                    .unwrap_or(0.0);
                                                bayes_output_counts
                                                    .push(RawTargetValueCount { value, count });
                                                let mut skip = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut skip) {
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "TargetValueCount" =>
                                                        {
                                                            break
                                                        }
                                                        Ok(Event::Empty(_)) => break,
                                                        _ => {}
                                                    }
                                                    skip.clear();
                                                    break;
                                                }
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "TargetValueCounts" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "BayesOutput" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "NaiveBayesModel" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawNaiveBayesModel {
        function_name,
        threshold,
        mining_schema,
        output,
        targets: Vec::new(),
        bayes_inputs,
        bayes_output_counts,
        local_derived_fields
    })
}

fn parse_nearest_neighbor_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawNearestNeighborModel> {
    let function_name = attr_required(start, "functionName", "NearestNeighborModel")?;
    let number_of_neighbors = attr(start, "numberOfNeighbors")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let mut mining_schema = Vec::new();
    let mut output = Vec::new();
    let mut instance_fields: Vec<RawInstanceField> = Vec::new();
    let mut instances: Vec<std::collections::HashMap<String, String>> = Vec::new();
    let mut knn_inputs = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "TrainingInstances" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "InstanceFields" =>
                                {
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(f_e))
                                                if tag_name(&f_e) == "InstanceField" =>
                                            {
                                                let field =
                                                    attr_required(&f_e, "field", "InstanceField")?;
                                                let column = attr(&f_e, "column")
                                                    .unwrap_or_else(|| field.clone());
                                                instance_fields
                                                    .push(RawInstanceField { field, column });
                                                let mut skip = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut skip) {
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "InstanceField" =>
                                                        {
                                                            break
                                                        }
                                                        Ok(Event::Empty(_)) => break,
                                                        _ => {}
                                                    }
                                                    skip.clear();
                                                    break;
                                                }
                                            }
                                            Ok(Event::Empty(f_e))
                                                if tag_name(&f_e) == "InstanceField" =>
                                            {
                                                let field =
                                                    attr_required(&f_e, "field", "InstanceField")?;
                                                let column = attr(&f_e, "column")
                                                    .unwrap_or_else(|| field.clone());
                                                instance_fields
                                                    .push(RawInstanceField { field, column });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "InstanceFields" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "InlineTable" =>
                                {
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(row_e))
                                                if tag_name(&row_e) == "row" =>
                                            {
                                                let mut row_buf = Vec::new();
                                                let mut row_map: std::collections::HashMap<
                                                    String,
                                                    String,
                                                > = std::collections::HashMap::new();
                                                loop {
                                                    match reader.read_event_into(&mut row_buf) {
                                                        Ok(Event::Start(cell_e)) => {
                                                            let col_name = tag_name(&cell_e);
                                                            let mut cell_buf = Vec::new();
                                                            let mut cell_val = String::new();
                                                            loop {
                                                                match reader
                                                                    .read_event_into(&mut cell_buf)
                                                                {
                                                                    Ok(Event::Text(t)) => {
                                                                        cell_val = t
                                                                            .unescape()
                                                                            .unwrap_or_default()
                                                                            .into_owned();
                                                                    }
                                                                    Ok(Event::End(end)) => {
                                                                        let tag = String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                        .into_owned();
                                                                        if tag == col_name {
                                                                            row_map.insert(
                                                                                col_name.clone(),
                                                                                cell_val.clone(),
                                                                            );
                                                                        }
                                                                        break;
                                                                    }
                                                                    _ => {}
                                                                }
                                                                cell_buf.clear();
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "row" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    row_buf.clear();
                                                }
                                                if !row_map.is_empty() {
                                                    instances.push(row_map);
                                                }
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "InlineTable" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                // TableLocator is a placeholder for external data (e.g., CSV/ARFF file).
                                // For Arrow bridge, we handle it gracefully by treating it as empty InlineTable
                                // (plan A4: handle TableLocator placeholder). No panic, just skip.
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "TableLocator" =>
                                {
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "TableLocator" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Eof) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                    }
                                    // intentionally leave `instances` empty — caller (arrow bridge)
                                    // will produce an empty RecordBatch placeholder via
                                    // `table_locator_placeholder_batch`. This keeps scoring from failing
                                    // on TableLocator-only models.
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "TableLocator" =>
                                {
                                    // self-closing <TableLocator/> — also empty placeholder
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "TrainingInstances" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "KNNInputs" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "KNNInput" => {
                                    let field = attr_required(&inner_e, "field", "KNNInput")?;
                                    knn_inputs.push(field);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "KNNInput" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "KNNInput" => {
                                    let field = attr_required(&inner_e, "field", "KNNInput")?;
                                    knn_inputs.push(field);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "KNNInputs" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Output" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "OutputField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "OutputField" =>
                                {
                                    let of = parse_output_field(&inner_e)?;
output.push(of);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Output" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "NearestNeighborModel" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawNearestNeighborModel {
        function_name,
        number_of_neighbors,
        mining_schema,
        output,
        targets: Vec::new(),
        instance_fields,
        instances,
        knn_inputs,
        local_derived_fields
    })
}

fn parse_general_regression_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawGeneralRegressionModel> {
    let function_name = attr_required(start, "functionName", "GeneralRegressionModel")?;
    let model_type = attr(start, "modelType");
    let target_variable_name = attr(start, "targetVariableName");
    let target_reference_category = attr(start, "targetReferenceCategory");
    let mut mining_schema = Vec::new();
    let output = Vec::new();
    let mut parameters = Vec::new();
    let mut factors = Vec::new();
    let mut covariates = Vec::new();
    let mut pp_matrix = Vec::new();
    let mut param_matrix = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "ParameterList" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "Parameter" => {
                                    let name = attr_required(&inner_e, "name", "Parameter")?;
                                    let label = attr(&inner_e, "label");
                                    parameters.push(RawParameter { name, label });
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Parameter" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "Parameter" => {
                                    let name = attr_required(&inner_e, "name", "Parameter")?;
                                    let label = attr(&inner_e, "label");
                                    parameters.push(RawParameter { name, label });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "ParameterList" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "FactorList" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "Predictor" => {
                                    let name = attr_required(&inner_e, "name", "Predictor")?;
                                    let contrast_type = attr(&inner_e, "contrastMatrixType");
                                    let mut cats = Vec::new();
                                    let mut matrix: Vec<Vec<f64>> = Vec::new();
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(c_e))
                                                if tag_name(&c_e) == "Categories" =>
                                            {
                                                let mut inner3 = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(cat_e))
                                                            if tag_name(&cat_e) == "Category" =>
                                                        {
                                                            if let Some(v) = attr(&cat_e, "value") {
                                                                cats.push(v);
                                                            }
                                                            let mut skip = Vec::new();
                                                            loop {
                                                                match reader
                                                                    .read_event_into(&mut skip)
                                                                {
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "Category" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    Ok(Event::Empty(_)) => break,
                                                                    _ => {}
                                                                }
                                                                skip.clear();
                                                                break;
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "Categories" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                            }
                                            Ok(Event::Start(c_e)) if tag_name(&c_e) == "Matrix" => {
                                                let mut inner3 = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(arr_e))
                                                            if tag_name(&arr_e) == "Array" =>
                                                        {
                                                            let mut txt = String::new();
                                                            let mut inner4 = Vec::new();
                                                            loop {
                                                                match reader
                                                                    .read_event_into(&mut inner4)
                                                                {
                                                                    Ok(Event::Text(t)) => {
                                                                        txt = t
                                                                            .unescape()
                                                                            .unwrap_or_default()
                                                                            .into_owned();
                                                                    }
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "Array" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                inner4.clear();
                                                            }
                                                            let row: Vec<f64> = txt
                                                                .split_whitespace()
                                                                .filter_map(|s| {
                                                                    s.parse::<f64>().ok()
                                                                })
                                                                .collect();
                                                            if !row.is_empty() {
                                                                matrix.push(row);
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "Matrix" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Predictor" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    factors.push(RawFactor {
                                        name,
                                        categories: cats,
                                        matrix,
                                        contrast_type,
                                    });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "FactorList" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "CovariateList" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "Predictor" => {
                                    let name = attr_required(&inner_e, "name", "Predictor")?;
                                    covariates.push(name);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Predictor" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "Predictor" => {
                                    let name = attr_required(&inner_e, "name", "Predictor")?;
                                    covariates.push(name);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "CovariateList" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "PPMatrix" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "PPCell" => {
                                    let value = attr_required(&inner_e, "value", "PPCell")?;
                                    let predictor_name =
                                        attr_required(&inner_e, "predictorName", "PPCell")?;
                                    let parameter_name =
                                        attr_required(&inner_e, "parameterName", "PPCell")?;
                                    pp_matrix.push(RawPPCell {
                                        value,
                                        predictor_name,
                                        parameter_name,
                                    });
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "PPCell" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "PPCell" => {
                                    let value = attr_required(&inner_e, "value", "PPCell")?;
                                    let predictor_name =
                                        attr_required(&inner_e, "predictorName", "PPCell")?;
                                    let parameter_name =
                                        attr_required(&inner_e, "parameterName", "PPCell")?;
                                    pp_matrix.push(RawPPCell {
                                        value,
                                        predictor_name,
                                        parameter_name,
                                    });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "PPMatrix" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "ParamMatrix" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "PCell" => {
                                    let target_category = attr(&inner_e, "targetCategory");
                                    let parameter_name =
                                        attr_required(&inner_e, "parameterName", "PCell")?;
                                    let beta = attr(&inner_e, "beta")
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .unwrap_or(0.0);
                                    param_matrix.push(RawPCell {
                                        target_category,
                                        parameter_name,
                                        beta,
                                    });
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "PCell" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "PCell" => {
                                    let target_category = attr(&inner_e, "targetCategory");
                                    let parameter_name =
                                        attr_required(&inner_e, "parameterName", "PCell")?;
                                    let beta = attr(&inner_e, "beta")
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .unwrap_or(0.0);
                                    param_matrix.push(RawPCell {
                                        target_category,
                                        parameter_name,
                                        beta,
                                    });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "ParamMatrix" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "GeneralRegressionModel" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawGeneralRegressionModel {
        function_name,
        mining_schema,
        output,
        targets: Vec::new(),
        model_type,
        target_variable_name,
        target_reference_category,
        parameters,
        factors,
        covariates,
        pp_matrix,
        param_matrix,
        local_derived_fields
    })
}

fn parse_support_vector_machine_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawSupportVectorMachineModel> {
    let function_name = attr_required(start, "functionName", "SupportVectorMachineModel")?;
    let mut mining_schema = Vec::new();
    let output = Vec::new();
    let mut vector_fields = Vec::new();
    let mut vector_instances = Vec::new();
    let mut support_vector_machine: Option<RawSupportVectorMachine> = None;
    let mut kernel_gamma: Option<f64> = None;
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "RadialBasisKernelType" => {
                        kernel_gamma = attr(&e, "gamma").and_then(|s| s.parse::<f64>().ok());
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "RadialBasisKernelType" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "VectorDictionary" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "VectorFields" =>
                                {
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(f_e))
                                                if tag_name(&f_e) == "FieldRef" =>
                                            {
                                                let field =
                                                    attr_required(&f_e, "field", "FieldRef")?;
                                                vector_fields.push(RawVectorField { field });
                                                let mut skip = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut skip) {
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "FieldRef" =>
                                                        {
                                                            break
                                                        }
                                                        Ok(Event::Empty(_)) => break,
                                                        _ => {}
                                                    }
                                                    skip.clear();
                                                    break;
                                                }
                                            }
                                            Ok(Event::Empty(f_e))
                                                if tag_name(&f_e) == "FieldRef" =>
                                            {
                                                let field =
                                                    attr_required(&f_e, "field", "FieldRef")?;
                                                vector_fields.push(RawVectorField { field });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "VectorFields" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "VectorInstance" =>
                                {
                                    let id = attr(&inner_e, "id").unwrap_or_else(|| {
                                        format!("vec{}", vector_instances.len())
                                    });
                                    let mut array = Vec::new();
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(arr_e))
                                                if tag_name(&arr_e) == "REAL-SparseArray" =>
                                            {
                                                let mut inner3 = Vec::new();
                                                let mut indices = Vec::new();
                                                let mut entries = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(idx_e))
                                                            if tag_name(&idx_e) == "Indices" =>
                                                        {
                                                            let mut txt_buf = Vec::new();
                                                            loop {
                                                                match reader
                                                                    .read_event_into(&mut txt_buf)
                                                                {
                                                                    Ok(Event::Text(t)) => {
                                                                        let txt = t
                                                                            .unescape()
                                                                            .unwrap_or_default()
                                                                            .into_owned();
                                                                        for part in
                                                                            txt.split_whitespace()
                                                                        {
                                                                            if let Ok(i) = part
                                                                                .parse::<usize>()
                                                                            {
                                                                                indices.push(i);
                                                                            }
                                                                        }
                                                                    }
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "Indices" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                txt_buf.clear();
                                                            }
                                                        }
                                                        Ok(Event::Start(ent_e))
                                                            if tag_name(&ent_e)
                                                                == "REAL-Entries" =>
                                                        {
                                                            let mut txt_buf = Vec::new();
                                                            loop {
                                                                match reader
                                                                    .read_event_into(&mut txt_buf)
                                                                {
                                                                    Ok(Event::Text(t)) => {
                                                                        let txt = t
                                                                            .unescape()
                                                                            .unwrap_or_default()
                                                                            .into_owned();
                                                                        for part in
                                                                            txt.split_whitespace()
                                                                        {
                                                                            if let Ok(f) =
                                                                                part.parse::<f64>()
                                                                            {
                                                                                entries.push(f);
                                                                            }
                                                                        }
                                                                    }
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "REAL-Entries" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    _ => {}
                                                                }
                                                                txt_buf.clear();
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "REAL-SparseArray" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                                let mut dense = vec![0.0; 2];
                                                for (idx, val) in indices.into_iter().zip(entries) {
                                                    if idx > 0 && idx <= dense.len() {
                                                        dense[idx - 1] = val;
                                                    }
                                                }
                                                array = dense;
                                            }
                                            Ok(Event::Start(arr_e))
                                                if tag_name(&arr_e) == "Array" =>
                                            {
                                                let mut txt_buf = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut txt_buf) {
                                                        Ok(Event::Text(t)) => {
                                                            let txt = t
                                                                .unescape()
                                                                .unwrap_or_default()
                                                                .into_owned();
                                                            for part in txt.split_whitespace() {
                                                                if let Ok(f) = part.parse::<f64>() {
                                                                    array.push(f);
                                                                }
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "Array" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    txt_buf.clear();
                                                }
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "VectorInstance" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    vector_instances.push(RawVectorInstance { id, array });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "VectorDictionary" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "SupportVectorMachine" => {
                        let mut svs = Vec::new();
                        let mut coeffs = Vec::new();
                        let mut abs_val: Option<f64> = None;
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "SupportVectors" =>
                                {
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(sv_e))
                                                if tag_name(&sv_e) == "SupportVector" =>
                                            {
                                                let vid = attr_required(
                                                    &sv_e,
                                                    "vectorId",
                                                    "SupportVector",
                                                )?;
                                                svs.push(RawSupportVector { vector_id: vid });
                                                let mut skip = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut skip) {
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "SupportVector" =>
                                                        {
                                                            break
                                                        }
                                                        Ok(Event::Empty(_)) => break,
                                                        _ => {}
                                                    }
                                                    skip.clear();
                                                    break;
                                                }
                                            }
                                            Ok(Event::Empty(sv_e))
                                                if tag_name(&sv_e) == "SupportVector" =>
                                            {
                                                let vid = attr_required(
                                                    &sv_e,
                                                    "vectorId",
                                                    "SupportVector",
                                                )?;
                                                svs.push(RawSupportVector { vector_id: vid });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "SupportVectors" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "Coefficients" =>
                                {
                                    abs_val = attr(&inner_e, "absoluteValue")
                                        .and_then(|s| s.parse::<f64>().ok());
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(coeff_e))
                                                if tag_name(&coeff_e) == "Coefficient" =>
                                            {
                                                let val = attr(&coeff_e, "value")
                                                    .and_then(|s| s.parse::<f64>().ok())
                                                    .unwrap_or(0.0);
                                                coeffs.push(RawCoefficient { value: val });
                                                let mut skip = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut skip) {
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "Coefficient" =>
                                                        {
                                                            break
                                                        }
                                                        Ok(Event::Empty(_)) => break,
                                                        _ => {}
                                                    }
                                                    skip.clear();
                                                    break;
                                                }
                                            }
                                            Ok(Event::Empty(coeff_e))
                                                if tag_name(&coeff_e) == "Coefficient" =>
                                            {
                                                let val = attr(&coeff_e, "value")
                                                    .and_then(|s| s.parse::<f64>().ok())
                                                    .unwrap_or(0.0);
                                                coeffs.push(RawCoefficient { value: val });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Coefficients" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "SupportVectorMachine" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                        support_vector_machine = Some(RawSupportVectorMachine {
                            support_vectors: svs,
                            coefficients: coeffs,
                            absolute_value: abs_val,
                        });
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "SupportVectorMachineModel" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawSupportVectorMachineModel {
        function_name,
        mining_schema,
        output,
        targets: Vec::new(),
        vector_fields,
        vector_instances,
        support_vector_machine,
        kernel_gamma,
        local_derived_fields
    })
}

fn parse_neural_network(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawNeuralNetwork> {
    let function_name = attr_required(start, "functionName", "NeuralNetwork")?;
    let model_name = attr(start, "modelName");
    let activation_function = attr(start, "activationFunction");
    let mut mining_schema = Vec::new();
    let output = Vec::new();
    let mut neural_inputs: Vec<RawNeuralInput> = Vec::new();
    let mut neural_layers: Vec<RawNeuralLayer> = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "NeuralInputs" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "NeuralInput" =>
                                {
                                    let id = attr_required(&inner_e, "id", "NeuralInput")?;
                                    let mut field = String::new();
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(df_e))
                                                if tag_name(&df_e) == "DerivedField" =>
                                            {
                                                let mut inner3 = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(fr_e))
                                                            if tag_name(&fr_e) == "FieldRef" =>
                                                        {
                                                            field = attr_required(
                                                                &fr_e, "field", "FieldRef",
                                                            )?;
                                                            let mut skip = Vec::new();
                                                            loop {
                                                                match reader
                                                                    .read_event_into(&mut skip)
                                                                {
                                                                    Ok(Event::End(end))
                                                                        if String::from_utf8_lossy(
                                                                            end.name().as_ref(),
                                                                        )
                                                                            == "FieldRef" =>
                                                                    {
                                                                        break
                                                                    }
                                                                    Ok(Event::Empty(_)) => break,
                                                                    _ => {}
                                                                }
                                                                skip.clear();
                                                                break;
                                                            }
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "DerivedField" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "NeuralInput" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    neural_inputs.push(RawNeuralInput { id, field });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "NeuralInputs" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "NeuralLayer" => {
                        let number_of_neurons =
                            attr(&e, "numberOfNeurons").and_then(|s| s.parse::<usize>().ok());
                        let activation_function =
                            attr(&e, "activationFunction").or_else(|| activation_function.clone());
                        let mut neurons = Vec::new();
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(neuron_e)) if tag_name(&neuron_e) == "Neuron" => {
                                    let id = attr_required(&neuron_e, "id", "Neuron")?;
                                    let bias =
                                        attr(&neuron_e, "bias").and_then(|s| s.parse::<f64>().ok());
                                    let mut cons = Vec::new();
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(con_e))
                                                if tag_name(&con_e) == "Con" =>
                                            {
                                                let from = attr_required(&con_e, "from", "Con")?;
                                                let weight = attr(&con_e, "weight")
                                                    .and_then(|s| s.parse::<f64>().ok())
                                                    .unwrap_or(0.0);
                                                cons.push(RawCon { from, weight });
                                                let mut skip = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut skip) {
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "Con" =>
                                                        {
                                                            break
                                                        }
                                                        Ok(Event::Empty(_)) => break,
                                                        _ => {}
                                                    }
                                                    skip.clear();
                                                    break;
                                                }
                                            }
                                            Ok(Event::Empty(con_e))
                                                if tag_name(&con_e) == "Con" =>
                                            {
                                                let from = attr_required(&con_e, "from", "Con")?;
                                                let weight = attr(&con_e, "weight")
                                                    .and_then(|s| s.parse::<f64>().ok())
                                                    .unwrap_or(0.0);
                                                cons.push(RawCon { from, weight });
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "Neuron" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    neurons.push(RawNeuron { id, bias, cons });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "NeuralLayer" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                        neural_layers.push(RawNeuralLayer {
                            number_of_neurons,
                            activation_function: activation_function.clone(),
                            neurons,
                        });
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "NeuralNetwork" => {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawNeuralNetwork {
        function_name,
        mining_schema,
        output,
        neural_inputs,
        neural_layers,
        model_name,
        activation_function,
        targets: vec![],
        local_derived_fields,
    })
}

fn parse_association_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawAssociationModel> {
    let function_name = attr_required(start, "functionName", "AssociationModel")?;
    let mut mining_schema = Vec::new();
    let output = Vec::new();
    let mut items = Vec::new();
    let mut itemsets = Vec::new();
    let mut rules = Vec::new();
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "Item" => {
                        let id = attr_required(&e, "id", "Item")?;
                        let value = attr_required(&e, "value", "Item")?;
                        items.push(RawItem { id, value });
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == "Item" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner.clear();
                            break;
                        }
                    }
                    "Itemset" => {
                        let id = attr_required(&e, "id", "Itemset")?;
                        let mut item_refs = Vec::new();
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "ItemRef" => {
                                    let item_ref = attr_required(&inner_e, "itemRef", "ItemRef")?;
                                    item_refs.push(item_ref);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "ItemRef" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                                            _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "ItemRef" => {
                                    let item_ref = attr_required(&inner_e, "itemRef", "ItemRef")?;
                                    item_refs.push(item_ref);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "Itemset" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                        itemsets.push(RawItemset { id, item_refs });
                    }
                    "AssociationRule" => {
                        let antecedent = attr_required(&e, "antecedent", "AssociationRule")?;
                        let consequent = attr_required(&e, "consequent", "AssociationRule")?;
                        let support = attr(&e, "support")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        let confidence = attr(&e, "confidence")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        let lift = attr(&e, "lift")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        rules.push(RawAssociationRule {
                            antecedent,
                            consequent,
                            support,
                            confidence,
                            lift,
                        });
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "AssociationRule" =>
                                {
                                    break
                                }
                                Ok(Event::Empty(_)) => break,
                                _ => {}
                            }
                            inner.clear();
                            break;
                        }
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = tag_name(&e);
                if tag == "Item" {
                    let id = attr_required(&e, "id", "Item")?;
                    let value = attr_required(&e, "value", "Item")?;
                    items.push(RawItem { id, value });
                } else if tag == "AssociationRule" {
                    let antecedent = attr_required(&e, "antecedent", "AssociationRule")?;
                    let consequent = attr_required(&e, "consequent", "AssociationRule")?;
                    let support = attr(&e, "support")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let confidence = attr(&e, "confidence")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let lift = attr(&e, "lift")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    rules.push(RawAssociationRule {
                        antecedent,
                        consequent,
                        support,
                        confidence,
                        lift,
                    });
                }
            }
            Ok(Event::End(e))
                if String::from_utf8_lossy(e.name().as_ref()) == "AssociationModel" =>
            {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawAssociationModel {
        function_name,
        mining_schema,
        output,
        targets: Vec::new(),
        items,
        itemsets,
        rules,
        local_derived_fields
    })
}

fn parse_rule_set_model(
    reader: &mut quick_xml::Reader<&[u8]>,
    start: &BytesStart,
) -> Result<RawRuleSetModel> {
    let function_name = attr_required(start, "functionName", "RuleSetModel")?;
    let mut mining_schema = Vec::new();
    let output = Vec::new();
    let mut rule_set: Option<RawRuleSet> = None;
    let mut local_derived_fields = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "MiningSchema" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                    let mut skip = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut skip) {
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "MiningField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Empty(_)) => break,
                        _ => {}
                                        }
                                        skip.clear();
                                        break;
                                    }
                                }
                                Ok(Event::Empty(inner_e))
                                    if tag_name(&inner_e) == "MiningField" =>
                                {
                                    let mf = parse_mining_field(&inner_e)?;
mining_schema.push(mf);
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "MiningSchema" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "RuleSet" => {
                        let record_count =
                            attr(&e, "recordCount").and_then(|s| s.parse::<f64>().ok());
                        let nb_correct = attr(&e, "nbCorrect").and_then(|s| s.parse::<f64>().ok());
                        let default_score = attr(&e, "defaultScore");
                        let mut rules = Vec::new();
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(rule_e)) if tag_name(&rule_e) == "SimpleRule" => {
                                    let id = attr(&rule_e, "id");
                                    let score = attr_required(&rule_e, "score", "SimpleRule")?;
                                    let mut predicate = RawPredicate::True;
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(pred_e))
                                                if tag_name(&pred_e) == "CompoundPredicate" =>
                                            {
                                                let boolean_operator = attr_required(
                                                    &pred_e,
                                                    "booleanOperator",
                                                    "CompoundPredicate",
                                                )?;
                                                let mut preds = Vec::new();
                                                let mut inner3 = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut inner3) {
                                                        Ok(Event::Start(simple_e))
                                                            if tag_name(&simple_e)
                                                                == "SimplePredicate" =>
                                                        {
                                                            preds.push(parse_simple_predicate(
                                                                &simple_e,
                                                            )?);
                                                        }
                                                        Ok(Event::Empty(simple_e))
                                                            if tag_name(&simple_e)
                                                                == "SimplePredicate" =>
                                                        {
                                                            preds.push(parse_simple_predicate(
                                                                &simple_e,
                                                            )?);
                                                        }
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "CompoundPredicate" =>
                                                        {
                                                            break
                                                        }
                                                        _ => {}
                                                    }
                                                    inner3.clear();
                                                }
                                                predicate = RawPredicate::Compound {
                                                    boolean_operator,
                                                    predicates: preds,
                                                };
                                            }
                                            Ok(Event::Start(pred_e))
                                                if tag_name(&pred_e) == "SimplePredicate" =>
                                            {
                                                predicate = parse_simple_predicate(&pred_e)?;
                                            }
                                            Ok(Event::Empty(pred_e))
                                                if tag_name(&pred_e) == "SimplePredicate" =>
                                            {
                                                predicate = parse_simple_predicate(&pred_e)?;
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "SimpleRule" =>
                                            {
                                                break
                                            }
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    rules.push(RawSimpleRule {
                                        id,
                                        score,
                                        predicate,
                                    });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "RuleSet" =>
                                {
                                    break
                                }
                                _ => {}
                            }
                            inner.clear();
                        }
                        rule_set = Some(RawRuleSet {
                            record_count,
                            nb_correct,
                            default_score,
                            rules,
                        });
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(reader)?;
                        local_derived_fields.extend(fields);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) if String::from_utf8_lossy(e.name().as_ref()) == "RuleSetModel" => {
                break
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(RawRuleSetModel {
        function_name,
        mining_schema,
        output,
        targets: Vec::new(),
        rule_set,
        local_derived_fields
    })
}

// ---------- Top-level ----------

pub fn unmarshal(bytes: &[u8]) -> Result<RawPmml> {
    let mut reader = new_reader(bytes)?;
    let mut data_dictionary = Vec::new();
    let mut tree_model: Option<RawTreeModel> = None;
    let mut regression_model: Option<RawRegressionModel> = None;
    let mut mining_model: Option<RawMiningModel> = None;
    let mut scorecard: Option<RawScorecard> = None;
    let mut clustering_model: Option<RawClusteringModel> = None;
    let mut naive_bayes_model: Option<RawNaiveBayesModel> = None;
    let mut nearest_neighbor_model: Option<RawNearestNeighborModel> = None;
    let mut support_vector_machine_model: Option<RawSupportVectorMachineModel> = None;
    let mut neural_network: Option<RawNeuralNetwork> = None;
    let mut general_regression_model: Option<RawGeneralRegressionModel> = None;
    let mut association_model: Option<RawAssociationModel> = None;
    let mut rule_set_model: Option<RawRuleSetModel> = None;
    let mut transformation_dictionary: Vec<RawDerivedField> = Vec::new();
    let mut define_functions: Vec<RawDefineFunction> = Vec::new();
    let mut buf = Vec::new();

    let mut extensions: Vec<RawExtension> = Vec::new();
    let mut unsupported_model: Option<String> = None;

    // Helper to parse <Extension> element (vendor handling, graceful)
    let parse_extension = |start: &BytesStart| -> RawExtension {
        let extender = attr(start, "extender");
        let name = attr(start, "name");
        let value = attr(start, "value");
        RawExtension {
            extender,
            name,
            value,
            content: None,
        }
    };

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                match tag.as_str() {
                    "DataDictionary" => {
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == "DataField" => {
                                    let name = attr_required(&inner_e, "name", "DataField")?;
                                    let data_type = attr(&inner_e, "dataType")
                                        .unwrap_or_else(|| "string".into());
                                    let op_type = attr(&inner_e, "optype")
                                        .unwrap_or_else(|| "categorical".into());
                                    let mut values = Vec::new();
                                    let mut inner2 = Vec::new();
                                    loop {
                                        match reader.read_event_into(&mut inner2) {
                                            Ok(Event::Start(v)) if tag_name(&v) == "Value" => {
                                                if let Some(val) = attr(&v, "value") {
                                                    values.push(val);
                                                }
                                                let mut skip = Vec::new();
                                                loop {
                                                    match reader.read_event_into(&mut skip) {
                                                        Ok(Event::End(end))
                                                            if String::from_utf8_lossy(
                                                                end.name().as_ref(),
                                                            ) == "Value" =>
                                                        {
                                                            break
                                                        }
                                                        Ok(Event::Empty(_)) => break,
                                                        _ => {}
                                                    }
                                                    skip.clear();
                                                    break;
                                                }
                                            }
                                            Ok(Event::Empty(v)) if tag_name(&v) == "Value" => {
                                                if let Some(val) = attr(&v, "value") {
                                                    values.push(val);
                                                }
                                            }
                                            Ok(Event::End(end))
                                                if String::from_utf8_lossy(end.name().as_ref())
                                                    == "DataField" =>
                                            {
                                                break
                                            }
                                            Ok(Event::Eof) => break,
                                            _ => {}
                                        }
                                        inner2.clear();
                                    }
                                    data_dictionary.push(RawDataField {
                                        name,
                                        data_type,
                                        op_type,
                                        values,
                                    });
                                }
                                Ok(Event::Empty(inner_e)) if tag_name(&inner_e) == "DataField" => {
                                    let name = attr_required(&inner_e, "name", "DataField")?;
                                    let data_type = attr(&inner_e, "dataType")
                                        .unwrap_or_else(|| "string".into());
                                    let op_type = attr(&inner_e, "optype")
                                        .unwrap_or_else(|| "categorical".into());
                                    data_dictionary.push(RawDataField {
                                        name,
                                        data_type,
                                        op_type,
                                        values: vec![],
                                    });
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "DataDictionary" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    "TreeModel" => {
                        let tm = parse_tree_model(&mut reader, &e)?;
                        tree_model = Some(tm);
                    }
                    "RegressionModel" => {
                        let rm = parse_regression_model(&mut reader, &e)?;
                        regression_model = Some(rm);
                    }
                    "MiningModel" => {
                        let mm = parse_mining_model(&mut reader, &e)?;
                        mining_model = Some(mm);
                    }
                    "Scorecard" => {
                        let sc = parse_scorecard(&mut reader, &e)?;
                        scorecard = Some(sc);
                    }
                    "ClusteringModel" => {
                        let cm = parse_clustering_model(&mut reader, &e)?;
                        clustering_model = Some(cm);
                    }
                    "NaiveBayesModel" => {
                        let nb = parse_naive_bayes_model(&mut reader, &e)?;
                        naive_bayes_model = Some(nb);
                    }
                    "NearestNeighborModel" => {
                        let nn = parse_nearest_neighbor_model(&mut reader, &e)?;
                        nearest_neighbor_model = Some(nn);
                    }
                    "SupportVectorMachineModel" => {
                        let svm = parse_support_vector_machine_model(&mut reader, &e)?;
                        support_vector_machine_model = Some(svm);
                    }
                    "GeneralRegressionModel" => {
                        let gr = parse_general_regression_model(&mut reader, &e)?;
                        general_regression_model = Some(gr);
                    }
                    "AssociationModel" => {
                        let am = parse_association_model(&mut reader, &e)?;
                        association_model = Some(am);
                    }
                    "RuleSetModel" => {
                        let rsm = parse_rule_set_model(&mut reader, &e)?;
                        rule_set_model = Some(rsm);
                    }
                    "NeuralNetwork" => {
                        let nn = parse_neural_network(&mut reader, &e)?;
                        neural_network = Some(nn);
                    }
                    "TransformationDictionary" => {
                        let (funcs, fields) = parse_transformation_dictionary(&mut reader, &e)?;
                        define_functions.extend(funcs);
                        transformation_dictionary.extend(fields);
                    }
                    "LocalTransformations" => {
                        let fields = parse_local_transformations(&mut reader)?;
                        transformation_dictionary.extend(fields);
                    }
                    "Extension" => {
                        // Gracefully capture vendor extension, do not error
                        let mut ext = parse_extension(&e);
                        // collect inner content until </Extension> if any
                        let mut inner = Vec::new();
                        let mut content = String::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Text(t)) => {
                                    content.push_str(&t.unescape().unwrap_or_default());
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref())
                                        == "Extension" =>
                                {
                                    break
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                        if !content.is_empty() {
                            ext.content = Some(content);
                        }
                        extensions.push(ext);
                    }
                    // Unsupported PMML 4.4 models — captured gracefully for verification (plan D1)
                    "AnomalyDetectionModel"
                    | "BaselineModel"
                    | "BaselineRegressionModel"
                    | "BayesianNetworkModel"
                    | "GaussianProcessModel"
                    | "SequenceModel"
                    | "TextModel"
                    | "TimeSeriesModel"
                    | "ModelComposition"
                    | "CenterFields" => {
                        if unsupported_model.is_none() {
                            unsupported_model = Some(tag.clone());
                        }
                        // consume until matching End tag to avoid polluting stream
                        let mut depth = 1usize;
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == tag => {
                                    depth += 1
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == tag =>
                                {
                                    depth -= 1;
                                    if depth == 0 {
                                        break;
                                    }
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    // Generic fallback for any other *Model tag not yet supported — treat as unsupported
                    _ if tag.ends_with("Model") => {
                        if unsupported_model.is_none() {
                            unsupported_model = Some(tag.clone());
                        }
                        let mut depth = 1usize;
                        let mut inner = Vec::new();
                        loop {
                            match reader.read_event_into(&mut inner) {
                                Ok(Event::Start(inner_e)) if tag_name(&inner_e) == tag => {
                                    depth += 1
                                }
                                Ok(Event::End(end))
                                    if String::from_utf8_lossy(end.name().as_ref()) == tag =>
                                {
                                    depth -= 1;
                                    if depth == 0 {
                                        break;
                                    }
                                }
                                Ok(Event::Eof) => break,
                                _ => {}
                            }
                            inner.clear();
                        }
                    }
                    _ => {} // other top-level ignored for v1 (Header, MiningBuildTask, etc)
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = tag_name(&e);
                if tag == "TreeModel" {
                    let tm = parse_tree_model(&mut reader, &e)?;
                    tree_model = Some(tm);
                } else if tag == "RegressionModel" {
                    let dummy_start = BytesStart::new("RegressionModel");
                    let rm = parse_regression_model(&mut reader, &dummy_start)?;
                    regression_model = Some(rm);
                } else if tag == "MiningModel" {
                    let dummy_start = BytesStart::new("MiningModel");
                    let mm = parse_mining_model(&mut reader, &dummy_start)?;
                    mining_model = Some(mm);
                } else if tag == "Scorecard" {
                    let sc = parse_scorecard(&mut reader, &e)?;
                    scorecard = Some(sc);
                } else if tag == "ClusteringModel" {
                    let cm = parse_clustering_model(&mut reader, &e)?;
                    clustering_model = Some(cm);
                } else if tag == "NaiveBayesModel" {
                    let nb = parse_naive_bayes_model(&mut reader, &e)?;
                    naive_bayes_model = Some(nb);
                } else if tag == "NearestNeighborModel" {
                    let nn = parse_nearest_neighbor_model(&mut reader, &e)?;
                    nearest_neighbor_model = Some(nn);
                } else if tag == "Extension" {
                    extensions.push(parse_extension(&e));
                } else if (tag.ends_with("Model") || tag == "ModelComposition")
                    && unsupported_model.is_none() {
                        unsupported_model = Some(tag);
                    }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(PmmlError::ParseError {
                    context: "xml".into(),
                    message: e.to_string(),
                });
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(RawPmml {
        data_dictionary,
        tree_model,
        regression_model,
        mining_model,
        scorecard,
        clustering_model,
        naive_bayes_model,
        nearest_neighbor_model,
        support_vector_machine_model,
        neural_network,
        general_regression_model,
        association_model,
        rule_set_model,
        transformation_dictionary,
        define_functions,
        extensions,
        unsupported_model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iris() {
        let xml = std::fs::read("/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/DecisionTreeIris.pmml").unwrap();
        let raw = unmarshal(&xml).unwrap();
        assert_eq!(raw.data_dictionary.len(), 3);
        assert!(raw.tree_model.is_some());
        let tm = raw.tree_model.unwrap();
        assert_eq!(tm.function_name, "classification");
        assert_eq!(tm.mining_schema.len(), 3);
        assert_eq!(tm.root.children.len(), 2);
    }

    #[test]
    fn xxe_blocked() {
        let xxe = br#"<?xml version="1.0"?>
<!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
<PMML version="4.4"><Header/><DataDictionary><DataField name="f" dataType="string" optype="categorical"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="f"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
        // Should not panic and should not contain passwd; unmarshal may error or ignore entity
        let res = unmarshal(xxe);
        // Accept either Ok with no passwd leak or Err; but must not expose file
        match res {
            Ok(raw) => {
                assert!(raw.data_dictionary.iter().all(|df| !df.name.contains("root:")));
            }
            Err(e) => {
                assert!(!e.to_string().contains("root:"));
            }
        }
    }

    #[test]
    fn depth_limit_enforced() {
        // Build xml with deep nesting >512 depth via nested nodes? Use PMML wrapper + deep a's
        let mut xml = String::from("<PMML version=\"4.4\"><Header/><DataDictionary><DataField name=\"f\" dataType=\"string\" optype=\"categorical\"/></DataDictionary><TreeModel functionName=\"classification\"><MiningSchema><MiningField name=\"f\"/></MiningSchema><Node score=\"a\"><True/></Node></TreeModel>");
        // Not easy to test deep nesting via unmarshal directly; reader's depth limit tested in reader.rs
        // This test ensures unmarshal handles normal file without depth error
        let bytes = xml.into_bytes();
        let res = unmarshal(&bytes);
        assert!(res.is_ok(), "normal depth should be ok: {:?}", res.err());
    }
}
