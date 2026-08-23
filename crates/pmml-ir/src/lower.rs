//! Lower RawPmml -> Ir (optimized).

use crate::intern::Interner;
use crate::ir::*;
use pmml_core::error::{PmmlError, Result};
use pmml_core::field::{DataType, OpType};
use pmml_core::{FieldId, SymbolId};
use pmml_xml::{RawPmml, RawPredicate};
use std::collections::HashMap;

fn parse_data_type(s: &str) -> Result<DataType> {
    s.parse::<DataType>().map_err(|e| PmmlError::ParseError {
        context: "DataType".into(),
        message: e,
    })
}
fn parse_op_type(s: &str) -> Result<OpType> {
    s.parse::<OpType>().map_err(|e| PmmlError::ParseError {
        context: "OpType".into(),
        message: e,
    })
}

fn parse_missing_strategy(s: Option<&str>) -> MissingValueStrategy {
    match s.unwrap_or("nullPrediction") {
        "lastPrediction" => MissingValueStrategy::LastPrediction,
        "nullPrediction" => MissingValueStrategy::NullPrediction,
        "defaultChild" => MissingValueStrategy::DefaultChild,
        _ => MissingValueStrategy::NullPrediction,
    }
}
fn parse_no_true_child(s: Option<&str>) -> NoTrueChildStrategy {
    match s.unwrap_or("returnNullPrediction") {
        "returnNullPrediction" => NoTrueChildStrategy::ReturnNullPrediction,
        "returnLastPrediction" => NoTrueChildStrategy::ReturnLastPrediction,
        _ => NoTrueChildStrategy::ReturnNullPrediction,
    }
}

fn parse_simple_operator(op: &str) -> Result<SimpleOperator> {
    Ok(match op {
        "equal" => SimpleOperator::Equal,
        "notEqual" => SimpleOperator::NotEqual,
        "lessThan" => SimpleOperator::LessThan,
        "lessOrEqual" => SimpleOperator::LessOrEqual,
        "greaterThan" => SimpleOperator::GreaterThan,
        "greaterOrEqual" => SimpleOperator::GreaterOrEqual,
        "isMissing" => SimpleOperator::IsMissing,
        "isNotMissing" => SimpleOperator::IsNotMissing,
        _ => {
            return Err(PmmlError::ParseError {
                context: "SimplePredicate".into(),
                message: format!("unknown operator {op}"),
            })
        }
    })
}

fn value_to_symbol_or_continuous(
    val_str: &str,
    data_type: DataType,
    interner: &mut Interner,
) -> SymbolIdOrContinuous {
    // For string/categorical -> Discrete, otherwise try parse as f64
    match data_type {
        DataType::String => SymbolIdOrContinuous::Symbol(interner.intern_symbol(val_str)),
        DataType::Integer | DataType::Float | DataType::Double => {
            if let Ok(f) = val_str.parse::<f64>() {
                SymbolIdOrContinuous::Continuous(f)
            } else {
                SymbolIdOrContinuous::Symbol(interner.intern_symbol(val_str))
            }
        }
        DataType::Boolean => {
            // boolean could be true/false or 0/1
            SymbolIdOrContinuous::Symbol(interner.intern_symbol(val_str))
        }
        _ => {
            // dates etc -> symbol
            SymbolIdOrContinuous::Symbol(interner.intern_symbol(val_str))
        }
    }
}

fn lower_predicate(
    raw: &RawPredicate,
    interner: &mut Interner,
    field_meta_map: &HashMap<FieldId, FieldMeta>,
    field_name_to_id: &HashMap<String, FieldId>,
) -> Result<PredicateIr> {
    match raw {
        RawPredicate::True => Ok(PredicateIr::True),
        RawPredicate::Simple {
            field,
            operator,
            value,
        } => {
            let fid = *field_name_to_id
                .get(field)
                .ok_or_else(|| PmmlError::MissingField(field.clone()))?;
            let meta = field_meta_map
                .get(&fid)
                .ok_or_else(|| PmmlError::MissingField(field.clone()))?;
            let op = parse_simple_operator(operator)?;
            let val = if matches!(op, SimpleOperator::IsMissing | SimpleOperator::IsNotMissing) {
                SymbolIdOrContinuous::Missing
            } else {
                value_to_symbol_or_continuous(value, meta.data_type, interner)
            };
            Ok(PredicateIr::Simple {
                field: fid,
                operator: op,
                value: val,
            })
        }
        RawPredicate::SimpleSet {
            field,
            boolean_operator,
            array,
        } => {
            let fid = *field_name_to_id
                .get(field)
                .ok_or_else(|| PmmlError::MissingField(field.clone()))?;
            let meta = field_meta_map
                .get(&fid)
                .cloned()
                .unwrap_or_else(|| FieldMeta {
                    field_id: fid,
                    name: field.clone(),
                    data_type: DataType::String,
                    op_type: OpType::Categorical,
                    values: vec![],
                });
            let is_in = boolean_operator == "isIn";
            let vals: Vec<SymbolIdOrContinuous> = array
                .split_whitespace()
                .map(|v| value_to_symbol_or_continuous(v, meta.data_type, interner))
                .collect();
            Ok(PredicateIr::SimpleSet {
                field: fid,
                is_in,
                array: vals,
            })
        }
        RawPredicate::Compound {
            boolean_operator,
            predicates,
        } => {
            let op = match boolean_operator.as_str() {
                "and" => CompoundOperator::And,
                "or" => CompoundOperator::Or,
                "xor" => CompoundOperator::Xor,
                "surrogate" => CompoundOperator::Surrogate,
                _ => {
                    return Err(PmmlError::ParseError {
                        context: "CompoundPredicate".into(),
                        message: format!("unknown operator {boolean_operator}"),
                    })
                }
            };
            let preds = predicates
                .iter()
                .map(|p| lower_predicate(p, interner, field_meta_map, field_name_to_id))
                .collect::<Result<Vec<_>>>()?;
            Ok(PredicateIr::Compound {
                operator: op,
                predicates: preds,
            })
        }
    }
}

fn flatten_node(
    raw: &pmml_xml::RawNode,
    interner: &mut Interner,
    field_meta_map: &HashMap<FieldId, FieldMeta>,
    field_name_to_id: &HashMap<String, FieldId>,
    out: &mut Vec<NodeIr>,
) -> Result<usize> {
    let idx = out.len();
    // placeholder to hold place
    out.push(NodeIr {
        id: raw.id.clone(),
        score: None,
        predicate: PredicateIr::True,
        children: vec![],
        score_distributions: vec![],
    });

    // predicate
    let pred = lower_predicate(&raw.predicate, interner, field_meta_map, field_name_to_id)?;
    // score conversion: if score string exists, intern or parse
    let score = raw.score.as_ref().map(|s| {
        // score could be discrete (classification) or continuous (regression)
        // Try parse as f64, fallback to symbol
        if let Ok(f) = s.parse::<f64>() {
            SymbolIdOrContinuous::Continuous(f)
        } else {
            SymbolIdOrContinuous::Symbol(interner.intern_symbol(s))
        }
    });

    let sds = raw
        .score_distributions
        .iter()
        .map(|sd| ScoreDistributionIr {
            value: interner.intern_symbol(&sd.value),
            record_count: sd.record_count,
        })
        .collect();

    // children indices
    let mut child_indices = Vec::new();
    for child_raw in &raw.children {
        let child_idx = flatten_node(child_raw, interner, field_meta_map, field_name_to_id, out)?;
        child_indices.push(child_idx);
    }

    // update node at idx
    out[idx] = NodeIr {
        id: raw.id.clone(),
        score,
        predicate: pred,
        children: child_indices,
        score_distributions: sds,
    };
    Ok(idx)
}

fn parse_regression_norm(s: Option<&str>) -> RegressionNormalizationMethod {
    match s.unwrap_or("none") {
        "none" => RegressionNormalizationMethod::None,
        "simplemax" => RegressionNormalizationMethod::SimpleMax,
        "softmax" => RegressionNormalizationMethod::Softmax,
        "logit" => RegressionNormalizationMethod::Logit,
        "probit" => RegressionNormalizationMethod::Probit,
        "cloglog" => RegressionNormalizationMethod::ClogLog,
        "exp" => RegressionNormalizationMethod::Exp,
        "loglog" => RegressionNormalizationMethod::Loglog,
        "cauchit" => RegressionNormalizationMethod::Cauchit,
        _ => RegressionNormalizationMethod::None,
    }
}

fn parse_multiple_model_method(s: &str) -> MultipleModelMethod {
    match s {
        "majorityVote" => MultipleModelMethod::MajorityVote,
        "weightedMajorityVote" => MultipleModelMethod::WeightedMajorityVote,
        "average" => MultipleModelMethod::Average,
        "weightedAverage" => MultipleModelMethod::WeightedAverage,
        "median" => MultipleModelMethod::Median,
        "weightedMedian" => MultipleModelMethod::WeightedMedian,
        "max" => MultipleModelMethod::Max,
        "sum" => MultipleModelMethod::Sum,
        "weightedSum" => MultipleModelMethod::WeightedSum,
        "selectFirst" => MultipleModelMethod::SelectFirst,
        "selectAll" => MultipleModelMethod::SelectAll,
        "modelChain" => MultipleModelMethod::ModelChain,
        _ => MultipleModelMethod::Average,
    }
}

fn parse_missing_pred(s: Option<&str>) -> MissingPredictionTreatment {
    match s.unwrap_or("continue") {
        "returnMissing" => MissingPredictionTreatment::ReturnMissing,
        "skipSegment" => MissingPredictionTreatment::SkipSegment,
        "continue" => MissingPredictionTreatment::Continue,
        _ => MissingPredictionTreatment::Continue,
    }
}

fn lower_mining_schema(
    raw_fields: &[pmml_xml::RawMiningField],
    field_name_to_id: &mut HashMap<String, FieldId>,
    field_meta_map: &mut HashMap<FieldId, FieldMeta>,
    interner: &mut Interner,
) -> Result<MiningSchemaIr> {
    let mut active_fields = Vec::new();
    let mut target_field: Option<FieldId> = None;
    let mut field_metas = Vec::new();
    for mf in raw_fields {
        let fid = if let Some(&id) = field_name_to_id.get(&mf.name) {
            id
        } else {
            // For modelChain, segment's mining_schema may reference output fields of previous segment (e.g., Probability_setosa)
            // Intern them as synthetic continuous double fields
            let id = interner.intern_field(&mf.name);
            field_name_to_id.insert(mf.name.clone(), id);
            let meta = FieldMeta {
                field_id: id,
                name: mf.name.clone(),
                data_type: DataType::Double,
                op_type: OpType::Continuous,
                values: vec![],
            };
            field_meta_map.insert(id, meta.clone());
            id
        };
        let meta = field_meta_map
            .get(&fid)
            .cloned()
            .ok_or_else(|| PmmlError::MissingField(mf.name.clone()))?;
        match mf.usage_type.as_deref() {
            Some("target") | Some("predicted") => target_field = Some(fid),
            Some("supplementary") => {} // not active
            _ => active_fields.push(fid),
        }
        field_metas.push(meta);
    }
    Ok(MiningSchemaIr {
        active_fields,
        target_field,
        field_metas,
        missing_value_replacement: None,
    })
}

fn lower_output(
    raw_output: &[pmml_xml::RawOutputField],
    field_name_to_id: &HashMap<String, FieldId>,
    interner: &mut Interner,
) -> Vec<OutputFieldIr> {
    raw_output
        .iter()
        .map(|of| {
            let feature = of
                .feature
                .as_deref()
                .unwrap_or("predictedValue")
                .parse::<pmml_core::field::ResultFeature>()
                .unwrap_or(pmml_core::field::ResultFeature::PredictedValue);
            let val = of.value.as_ref().map(|v| interner.intern_symbol(v));
            let field = field_name_to_id.get(&of.name).copied();
            OutputFieldIr {
                name: of.name.clone(),
                feature,
                value: val,
                field,
            }
        })
        .collect()
}

fn lower_regression(
    raw: &pmml_xml::RawRegressionModel,
    field_name_to_id: &mut HashMap<String, FieldId>,
    field_meta_map: &mut HashMap<FieldId, FieldMeta>,
    interner: &mut Interner,
) -> Result<RegressionIr> {
    let mining_schema = lower_mining_schema(
        &raw.mining_schema,
        field_name_to_id,
        field_meta_map,
        interner,
    )?;
    let output = lower_output(&raw.output, field_name_to_id, interner);
    let mut tables = Vec::new();
    for tbl in &raw.regression_tables {
        let mut numeric_predictors = Vec::new();
        for np in &tbl.numeric_predictors {
            let fid = if let Some(&id) = field_name_to_id.get(&np.name) {
                id
            } else {
                let id = interner.intern_field(&np.name);
                field_name_to_id.insert(np.name.clone(), id);
                let meta = FieldMeta {
                    field_id: id,
                    name: np.name.clone(),
                    data_type: DataType::Double,
                    op_type: OpType::Continuous,
                    values: vec![],
                };
                field_meta_map.insert(id, meta);
                id
            };
            numeric_predictors.push(NumericPredictorIr {
                field: fid,
                coefficient: np.coefficient,
                exponent: np.exponent,
            });
        }
        let mut categorical_predictors = Vec::new();
        for cp in &tbl.categorical_predictors {
            let fid = if let Some(&id) = field_name_to_id.get(&cp.name) {
                id
            } else {
                let id = interner.intern_field(&cp.name);
                field_name_to_id.insert(cp.name.clone(), id);
                let meta = FieldMeta {
                    field_id: id,
                    name: cp.name.clone(),
                    data_type: DataType::String,
                    op_type: OpType::Categorical,
                    values: vec![],
                };
                field_meta_map.insert(id, meta);
                id
            };
            let val = interner.intern_symbol(&cp.value);
            categorical_predictors.push(CategoricalPredictorIr {
                field: fid,
                value: val,
                coefficient: cp.coefficient,
            });
        }
        let target_category = tbl
            .target_category
            .as_ref()
            .map(|s| interner.intern_symbol(s));
        tables.push(RegressionTableIr {
            intercept: tbl.intercept,
            target_category,
            numeric_predictors,
            categorical_predictors,
        });
    }
    Ok(RegressionIr {
        function_name: raw.function_name.clone(),
        mining_schema,
        regression_tables: tables,
        normalization_method: parse_regression_norm(raw.normalization_method.as_deref()),
        targets: vec![],
        output,
    })
}

fn lower_tree_raw(
    raw: &pmml_xml::RawTreeModel,
    field_name_to_id: &mut HashMap<String, FieldId>,
    field_meta_map: &mut HashMap<FieldId, FieldMeta>,
    interner: &mut Interner,
) -> Result<TreeIr> {
    let mining_schema = lower_mining_schema(
        &raw.mining_schema,
        field_name_to_id,
        field_meta_map,
        interner,
    )?;
    let output = lower_output(&raw.output, field_name_to_id, interner);
    let mut nodes = Vec::new();
    flatten_node(
        &raw.root,
        interner,
        field_meta_map,
        field_name_to_id,
        &mut nodes,
    )?;
    Ok(TreeIr {
        function_name: raw.function_name.clone(),
        missing_value_strategy: parse_missing_strategy(raw.missing_value_strategy.as_deref()),
        no_true_child_strategy: parse_no_true_child(raw.no_true_child_strategy.as_deref()),
        nodes,
        mining_schema,
        targets: vec![],
        output,
    })
}

fn lower_segment_model(
    raw: &pmml_xml::RawSegmentModel,
    field_name_to_id: &mut HashMap<String, FieldId>,
    field_meta_map: &mut HashMap<FieldId, FieldMeta>,
    interner: &mut Interner,
) -> Result<ModelIr> {
    match raw {
        pmml_xml::RawSegmentModel::Tree(tm) => {
            let tree_ir = lower_tree_raw(tm, field_name_to_id, field_meta_map, interner)?;
            Ok(ModelIr::Tree(tree_ir))
        }
        pmml_xml::RawSegmentModel::Regression(rm) => {
            let reg_ir = lower_regression(rm, field_name_to_id, field_meta_map, interner)?;
            Ok(ModelIr::Regression(reg_ir))
        }
    }
}

pub fn lower(raw: RawPmml) -> Result<Ir> {
    let mut interner = Interner::new();
    let mut field_name_to_id: HashMap<String, FieldId> = HashMap::new();
    let mut data_dictionary: Vec<FieldMeta> = Vec::new();
    let mut field_meta_map: HashMap<FieldId, FieldMeta> = HashMap::new();

    for df in &raw.data_dictionary {
        let fid = interner.intern_field(&df.name);
        field_name_to_id.insert(df.name.clone(), fid);
        let dt = parse_data_type(&df.data_type)?;
        if dt.is_unsupported() {
            return Err(PmmlError::UnsupportedMarkup(format!(
                "unsupported DATATYPE {}",
                df.data_type
            )));
        }
        let ot = parse_op_type(&df.op_type)?;
        let vals: Vec<SymbolId> = df
            .values
            .iter()
            .map(|v| interner.intern_symbol(v))
            .collect();
        let meta = FieldMeta {
            field_id: fid,
            name: df.name.clone(),
            data_type: dt,
            op_type: ot,
            values: vals.clone(),
        };
        field_meta_map.insert(fid, meta.clone());
        data_dictionary.push(meta);
    }

    // Build Ir — handle Tree, Regression, Mining
    let (model, derived_fields) = if let Some(tm) = raw.tree_model {
        let tree_ir = lower_tree_raw(
            &tm,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        (ModelIr::Tree(tree_ir), vec![])
    } else if let Some(rm) = raw.regression_model {
        let reg_ir = lower_regression(
            &rm,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        (ModelIr::Regression(reg_ir), vec![])
    } else if let Some(mm) = raw.mining_model {
        // MiningModel: need to lower its mining_schema + segmentation
        let mining_schema = lower_mining_schema(
            &mm.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&mm.output, &field_name_to_id, &mut interner);
        let segmentation = if let Some(seg_raw) = mm.segmentation {
            let mut segments = Vec::new();
            for seg in &seg_raw.segments {
                let pred = lower_predicate(
                    &seg.predicate,
                    &mut interner,
                    &field_meta_map,
                    &field_name_to_id,
                )?;
                let model_ir = lower_segment_model(
                    &seg.model,
                    &mut field_name_to_id,
                    &mut field_meta_map,
                    &mut interner,
                )?;
                segments.push(SegmentIr {
                    id: seg.id.clone(),
                    predicate: pred,
                    weight: seg.weight,
                    model: Box::new(model_ir),
                });
            }
            SegmentationIr {
                multiple_model_method: parse_multiple_model_method(&seg_raw.multiple_model_method),
                missing_prediction_treatment: parse_missing_pred(
                    seg_raw.missing_prediction_treatment.as_deref(),
                ),
                segments,
            }
        } else {
            return Err(PmmlError::UnsupportedMarkup(
                "MiningModel without Segmentation not supported".into(),
            ));
        };
        let mining_ir = MiningIr {
            function_name: mm.function_name.clone(),
            mining_schema,
            segmentation,
            targets: vec![],
            output,
        };
        (ModelIr::Mining(mining_ir), vec![])
    } else if let Some(sc) = raw.scorecard {
        let mining_schema = lower_mining_schema(
            &sc.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&sc.output, &field_name_to_id, &mut interner);
        let mut characteristics = Vec::new();
        for ch in &sc.characteristics {
            let mut attrs = Vec::new();
            for attr in &ch.attributes {
                let pred = lower_predicate(
                    &attr.predicate,
                    &mut interner,
                    &field_meta_map,
                    &field_name_to_id,
                )?;
                attrs.push(AttributeIr {
                    partial_score: attr.partial_score,
                    predicate: pred,
                    reason_code: attr.reason_code.clone(),
                });
            }
            characteristics.push(CharacteristicIr {
                name: ch.name.clone(),
                reason_code: ch.reason_code.clone(),
                baseline_score: ch.baseline_score.unwrap_or(0.0),
                attributes: attrs,
            });
        }
        let scorecard_ir = ScorecardIr {
            function_name: sc.function_name.clone(),
            initial_score: sc.initial_score,
            use_reason_codes: sc.use_reason_codes.unwrap_or(false),
            reason_code_algorithm: sc
                .reason_code_algorithm
                .unwrap_or_else(|| "pointsAbove".to_string()),
            mining_schema,
            characteristics,
            output,
        };
        (ModelIr::Scorecard(scorecard_ir), vec![])
    } else if let Some(cm) = raw.clustering_model {
        let mining_schema = lower_mining_schema(
            &cm.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&cm.output, &field_name_to_id, &mut interner);
        let mut clusters = Vec::new();
        for cl in &cm.clusters {
            let sym = interner.intern_symbol(&cl.name);
            clusters.push(ClusterIr {
                name: sym,
                name_str: cl.name.clone(),
                array: cl.array.clone(),
            });
        }
        let mut clustering_fields = Vec::new();
        for f in &cm.clustering_fields {
            let fid = if let Some(&id) = field_name_to_id.get(f) {
                id
            } else {
                let id = interner.intern_field(f);
                field_name_to_id.insert(f.clone(), id);
                let meta = FieldMeta {
                    field_id: id,
                    name: f.clone(),
                    data_type: DataType::Double,
                    op_type: OpType::Continuous,
                    values: vec![],
                };
                field_meta_map.insert(id, meta);
                id
            };
            clustering_fields.push(fid);
        }
        let clustering_ir = ClusteringIr {
            function_name: cm.function_name.clone(),
            model_class: cm
                .model_class
                .clone()
                .unwrap_or_else(|| "centerBased".to_string()),
            number_of_clusters: cm.number_of_clusters.unwrap_or(clusters.len()),
            mining_schema,
            comparison_measure: cm
                .comparison_measure
                .as_ref()
                .map(|c| c.kind.clone())
                .unwrap_or_else(|| "euclidean".to_string()),
            clustering_fields,
            clusters,
            output,
        };
        (ModelIr::Clustering(clustering_ir), vec![])
    } else if let Some(nb) = raw.naive_bayes_model {
        let mining_schema = lower_mining_schema(
            &nb.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&nb.output, &field_name_to_id, &mut interner);
        let mut bayes_inputs = Vec::new();
        for bi in &nb.bayes_inputs {
            let fid = if let Some(&id) = field_name_to_id.get(&bi.field_name) {
                id
            } else {
                let id = interner.intern_field(&bi.field_name);
                field_name_to_id.insert(bi.field_name.clone(), id);
                let meta = FieldMeta {
                    field_id: id,
                    name: bi.field_name.clone(),
                    data_type: DataType::String,
                    op_type: OpType::Categorical,
                    values: vec![],
                };
                field_meta_map.insert(id, meta);
                id
            };
            let mut target_value_stats = Vec::new();
            for tvs in &bi.target_value_stats {
                let sid = interner.intern_symbol(&tvs.value);
                target_value_stats.push(TargetValueStatIr {
                    value: sid,
                    mean: tvs.gaussian_mean,
                    variance: tvs.gaussian_variance,
                });
            }
            let mut pair_counts = Vec::new();
            for pc in &bi.pair_counts {
                let pc_sid = interner.intern_symbol(&pc.value);
                let mut target_counts = Vec::new();
                for tc in &pc.target_counts {
                    let t_sid = interner.intern_symbol(&tc.value);
                    target_counts.push(TargetValueCountIr {
                        value: t_sid,
                        count: tc.count,
                    });
                }
                pair_counts.push(PairCountsIr {
                    value: pc_sid,
                    target_counts,
                });
            }
            bayes_inputs.push(BayesInputIr {
                field: fid,
                target_value_stats,
                pair_counts,
            });
        }
        let mut bayes_output_counts = Vec::new();
        for tc in &nb.bayes_output_counts {
            let sid = interner.intern_symbol(&tc.value);
            bayes_output_counts.push(TargetValueCountIr {
                value: sid,
                count: tc.count,
            });
        }
        let nb_ir = NaiveBayesIr {
            function_name: nb.function_name.clone(),
            threshold: nb.threshold,
            mining_schema,
            output,
            bayes_inputs,
            bayes_output_counts,
        };
        (ModelIr::NaiveBayes(nb_ir), vec![])
    } else if let Some(nn) = raw.nearest_neighbor_model {
        let mining_schema = lower_mining_schema(
            &nn.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&nn.output, &field_name_to_id, &mut interner);
        // knn_inputs
        let mut knn_inputs = Vec::new();
        for f in &nn.knn_inputs {
            let fid = if let Some(&id) = field_name_to_id.get(f) {
                id
            } else {
                let id = interner.intern_field(f);
                field_name_to_id.insert(f.clone(), id);
                let meta = FieldMeta {
                    field_id: id,
                    name: f.clone(),
                    data_type: DataType::Double,
                    op_type: OpType::Continuous,
                    values: vec![],
                };
                field_meta_map.insert(id, meta);
                id
            };
            knn_inputs.push(fid);
        }
        // instances: convert HashMap<String,String> to HashMap<FieldId, Value>
        let mut instances = Vec::new();
        let mut instance_ids = Vec::new();
        for row in &nn.instances {
            let mut map: std::collections::HashMap<pmml_core::FieldId, pmml_core::Value> =
                std::collections::HashMap::new();
            for inst_field in &nn.instance_fields {
                let col = &inst_field.column;
                let field_name = &inst_field.field;
                if let Some(val_str) = row.get(col) {
                    let fid = if let Some(&id) = field_name_to_id.get(field_name) {
                        id
                    } else {
                        let id = interner.intern_field(field_name);
                        field_name_to_id.insert(field_name.clone(), id);
                        let meta = FieldMeta {
                            field_id: id,
                            name: field_name.clone(),
                            data_type: DataType::String,
                            op_type: OpType::Categorical,
                            values: vec![],
                        };
                        field_meta_map.insert(id, meta);
                        id
                    };
                    // Try to parse as f64, else as discrete
                    let val = if let Ok(f) = val_str.parse::<f64>() {
                        pmml_core::Value::Continuous(f)
                    } else {
                        let sid = interner.intern_symbol(val_str);
                        pmml_core::Value::Discrete(sid)
                    };
                    map.insert(fid, val);
                    if field_name == "ID" || field_name == "output" || field_name == "output" {
                        // Capture ID if needed
                    }
                }
            }
            // Extract ID if present (field "ID" or first instance field)
            let id_val = row
                .values()
                .next()
                .cloned()
                .unwrap_or_else(|| format!("{}", instances.len()));
            instance_ids.push(id_val);
            instances.push(map);
        }
        let nn_ir = NearestNeighborIr {
            function_name: nn.function_name.clone(),
            number_of_neighbors: nn.number_of_neighbors,
            mining_schema,
            output,
            knn_inputs,
            instances,
            instance_ids,
        };
        (ModelIr::NearestNeighbor(nn_ir), vec![])
    } else if let Some(gr) = raw.general_regression_model {
        let mining_schema = lower_mining_schema(
            &gr.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&gr.output, &field_name_to_id, &mut interner);
        let gr_ir = GeneralRegressionIr {
            function_name: gr.function_name.clone(),
            mining_schema,
            output,
        };
        (ModelIr::GeneralRegression(gr_ir), vec![])
    } else if let Some(svm) = raw.support_vector_machine_model {
        let mining_schema = lower_mining_schema(
            &svm.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&svm.output, &field_name_to_id, &mut interner);
        let svm_ir = SupportVectorMachineIr {
            function_name: svm.function_name.clone(),
            mining_schema,
            output,
        };
        (ModelIr::SupportVectorMachine(svm_ir), vec![])
    } else if let Some(am) = raw.association_model {
        let mining_schema = lower_mining_schema(
            &am.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&am.output, &field_name_to_id, &mut interner);
        let assoc_ir = AssociationIr {
            function_name: am.function_name.clone(),
            mining_schema,
            output,
        };
        (ModelIr::Association(assoc_ir), vec![])
    } else if let Some(rs) = raw.rule_set_model {
        let mining_schema = lower_mining_schema(
            &rs.mining_schema,
            &mut field_name_to_id,
            &mut field_meta_map,
            &mut interner,
        )?;
        let output = lower_output(&rs.output, &field_name_to_id, &mut interner);
        let rs_ir = RuleSetIr {
            function_name: rs.function_name.clone(),
            mining_schema,
            output,
        };
        (ModelIr::RuleSet(rs_ir), vec![])
    } else if raw.neural_network.is_some() {
        return Err(PmmlError::UnsupportedMarkup(
            "NeuralNetwork not yet fully supported (stub)".into(),
        ));
    } else {
        return Err(PmmlError::UnsupportedMarkup(
            "no supported model found".into(),
        ));
    };

    // Build field_names / symbol_names snapshot
    let mut field_names: HashMap<FieldId, String> = HashMap::new();
    for (name, fid) in &field_name_to_id {
        field_names.insert(*fid, name.clone());
    }
    let mut symbol_names: HashMap<SymbolId, String> = HashMap::new();
    for (s, id) in interner.symbol_map() {
        symbol_names.insert(*id, s.clone());
    }

    Ok(Ir {
        data_dictionary,
        derived_fields,
        model,
        field_names,
        symbol_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_xml::unmarshal;

    #[test]
    fn lower_iris() {
        let xml = std::fs::read("/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/DecisionTreeIris.pmml").unwrap();
        let raw = unmarshal(&xml).unwrap();
        let ir = lower(raw).unwrap();
        assert_eq!(ir.data_dictionary.len(), 3);
        match ir.model {
            ModelIr::Tree(ref t) => {
                assert_eq!(t.nodes.len(), 5); // root + 2 + 2
                assert_eq!(t.mining_schema.active_fields.len(), 2);
            }
            _ => panic!("expected tree"),
        }
    }
}
