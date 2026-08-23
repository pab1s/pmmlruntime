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
}

#[derive(Debug, Clone)]
pub struct RawOutputField {
    pub name: String,
    pub feature: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawTreeModel {
    pub function_name: String,
    pub missing_value_strategy: Option<String>,
    pub no_true_child_strategy: Option<String>,
    pub mining_schema: Vec<RawMiningField>,
    pub output: Vec<RawOutputField>,
    pub root: RawNode,
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
    pub regression_tables: Vec<RawRegressionTable>,
    pub normalization_method: Option<String>,
    pub model_name: Option<String>,
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
    pub model_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawNode {
    pub id: Option<String>,
    pub score: Option<String>,
    pub record_count: Option<f64>,
    pub predicate: RawPredicate,
    pub score_distributions: Vec<RawScoreDistribution>,
    pub children: Vec<RawNode>,
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
pub struct RawPmml {
    pub data_dictionary: Vec<RawDataField>,
    pub tree_model: Option<RawTreeModel>,
    pub regression_model: Option<RawRegressionModel>,
    pub mining_model: Option<RawMiningModel>,
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
                                    let name = attr_required(&inner_e, "name", "MiningField")?;
                                    let usage_type = attr(&inner_e, "usageType");
                                    let importance = attr(&inner_e, "importance")
                                        .and_then(|s| s.parse::<f64>().ok());
                                    mining_schema.push(RawMiningField {
                                        name,
                                        usage_type,
                                        importance,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "MiningField")?;
                                    let usage_type = attr(&inner_e, "usageType");
                                    let importance = attr(&inner_e, "importance")
                                        .and_then(|s| s.parse::<f64>().ok());
                                    mining_schema.push(RawMiningField {
                                        name,
                                        usage_type,
                                        importance,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "OutputField")?;
                                    let feature = attr(&inner_e, "feature");
                                    let value = attr(&inner_e, "value");
                                    output.push(RawOutputField {
                                        name,
                                        feature,
                                        value,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "OutputField")?;
                                    let feature = attr(&inner_e, "feature");
                                    let value = attr(&inner_e, "value");
                                    output.push(RawOutputField {
                                        name,
                                        feature,
                                        value,
                                    });
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
                    _ => {
                        // skip LocalTransformations, ModelStats, Targets, Extension, etc for v1
                        // Need to consume subtree if it's Start
                        if tag == "LocalTransformations"
                            || tag == "Targets"
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
        root,
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
                                    let name = attr_required(&inner_e, "name", "MiningField")?;
                                    let usage_type = attr(&inner_e, "usageType");
                                    let importance = attr(&inner_e, "importance")
                                        .and_then(|s| s.parse::<f64>().ok());
                                    mining_schema.push(RawMiningField {
                                        name,
                                        usage_type,
                                        importance,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "MiningField")?;
                                    let usage_type = attr(&inner_e, "usageType");
                                    let importance = attr(&inner_e, "importance")
                                        .and_then(|s| s.parse::<f64>().ok());
                                    mining_schema.push(RawMiningField {
                                        name,
                                        usage_type,
                                        importance,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "OutputField")?;
                                    let feature = attr(&inner_e, "feature");
                                    let value = attr(&inner_e, "value");
                                    output.push(RawOutputField {
                                        name,
                                        feature,
                                        value,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "OutputField")?;
                                    let feature = attr(&inner_e, "feature");
                                    let value = attr(&inner_e, "value");
                                    output.push(RawOutputField {
                                        name,
                                        feature,
                                        value,
                                    });
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
                    _ => {
                        if tag == "LocalTransformations" || tag == "Targets" || tag == "ModelStats"
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
        regression_tables,
        normalization_method,
        model_name,
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
                                    match itag.as_str() {
                                        "SimplePredicate" => {
                                            preds.push(parse_simple_predicate(&inner_e)?)
                                        }
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
                            mining_schema: vec![],
                            output: vec![],
                            regression_tables,
                            normalization_method: None,
                            model_name: None,
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
                                    let name = attr_required(&inner_e, "name", "MiningField")?;
                                    let usage_type = attr(&inner_e, "usageType");
                                    mining_schema.push(RawMiningField {
                                        name,
                                        usage_type,
                                        importance: None,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "MiningField")?;
                                    let usage_type = attr(&inner_e, "usageType");
                                    mining_schema.push(RawMiningField {
                                        name,
                                        usage_type,
                                        importance: None,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "OutputField")?;
                                    let feature = attr(&inner_e, "feature");
                                    let value = attr(&inner_e, "value");
                                    output.push(RawOutputField {
                                        name,
                                        feature,
                                        value,
                                    });
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
                                    let name = attr_required(&inner_e, "name", "OutputField")?;
                                    let feature = attr(&inner_e, "feature");
                                    let value = attr(&inner_e, "value");
                                    output.push(RawOutputField {
                                        name,
                                        feature,
                                        value,
                                    });
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
                    _ => {
                        if tag == "LocalTransformations" || tag == "Targets" {
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
        model_name,
    })
}

// ---------- Top-level ----------

pub fn unmarshal(bytes: &[u8]) -> Result<RawPmml> {
    let mut reader = new_reader(bytes)?;
    let mut data_dictionary = Vec::new();
    let mut tree_model: Option<RawTreeModel> = None;
    let mut regression_model: Option<RawRegressionModel> = None;
    let mut mining_model: Option<RawMiningModel> = None;
    let mut buf = Vec::new();

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
                    _ => {} // other top-level models ignored for v1
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = tag_name(&e);
                if tag == "TreeModel" {
                    let tm = parse_tree_model(&mut reader, &e)?;
                    tree_model = Some(tm);
                } else if tag == "RegressionModel" {
                    // Empty RegressionModel unlikely, but handle
                    let dummy_start = BytesStart::new("RegressionModel");
                    let rm = parse_regression_model(&mut reader, &dummy_start)?;
                    regression_model = Some(rm);
                } else if tag == "MiningModel" {
                    let dummy_start = BytesStart::new("MiningModel");
                    let mm = parse_mining_model(&mut reader, &dummy_start)?;
                    mining_model = Some(mm);
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
}
