//! Lower RawPmml -> Ir (optimized).

use crate::intern::Interner;
use crate::ir::*;
use pmml_core::error::{PmmlError, Result};
use pmml_core::field::{DataType, OpType};
use pmml_core::{FieldId, SymbolId};
use pmml_xml::{RawPmml, RawPredicate};
use std::collections::HashMap;

fn parse_data_type(s: &str) -> Result<DataType> {
    s.parse::<DataType>().map_err(|e| PmmlError::ParseError { context: "DataType".into(), message: e })
}
fn parse_op_type(s: &str) -> Result<OpType> {
    s.parse::<OpType>().map_err(|e| PmmlError::ParseError { context: "OpType".into(), message: e })
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
        _ => return Err(PmmlError::ParseError { context: "SimplePredicate".into(), message: format!("unknown operator {op}") }),
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
        RawPredicate::Simple { field, operator, value } => {
            let fid = *field_name_to_id.get(field).ok_or_else(|| PmmlError::MissingField(field.clone()))?;
            let meta = field_meta_map.get(&fid).ok_or_else(|| PmmlError::MissingField(field.clone()))?;
            let op = parse_simple_operator(operator)?;
            let val = if matches!(op, SimpleOperator::IsMissing | SimpleOperator::IsNotMissing) {
                SymbolIdOrContinuous::Missing
            } else {
                value_to_symbol_or_continuous(value, meta.data_type, interner)
            };
            Ok(PredicateIr::Simple { field: fid, operator: op, value: val })
        }
        RawPredicate::SimpleSet { field, boolean_operator, array } => {
            let fid = *field_name_to_id.get(field).ok_or_else(|| PmmlError::MissingField(field.clone()))?;
            let meta = field_meta_map.get(&fid).cloned().unwrap_or_else(|| FieldMeta {
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
            Ok(PredicateIr::SimpleSet { field: fid, is_in, array: vals })
        }
        RawPredicate::Compound { boolean_operator, predicates } => {
            let op = match boolean_operator.as_str() {
                "and" => CompoundOperator::And,
                "or" => CompoundOperator::Or,
                "xor" => CompoundOperator::Xor,
                "surrogate" => CompoundOperator::Surrogate,
                _ => return Err(PmmlError::ParseError { context: "CompoundPredicate".into(), message: format!("unknown operator {boolean_operator}") }),
            };
            let preds = predicates.iter().map(|p| lower_predicate(p, interner, field_meta_map, field_name_to_id)).collect::<Result<Vec<_>>>()?;
            Ok(PredicateIr::Compound { operator: op, predicates: preds })
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

    let sds = raw.score_distributions.iter().map(|sd| {
        ScoreDistributionIr { value: interner.intern_symbol(&sd.value), record_count: sd.record_count }
    }).collect();

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
            return Err(PmmlError::UnsupportedMarkup(format!("unsupported DATATYPE {}", df.data_type)));
        }
        let ot = parse_op_type(&df.op_type)?;
        let vals: Vec<SymbolId> = df.values.iter().map(|v| interner.intern_symbol(v)).collect();
        let meta = FieldMeta { field_id: fid, name: df.name.clone(), data_type: dt, op_type: ot, values: vals.clone() };
        field_meta_map.insert(fid, meta.clone());
        data_dictionary.push(meta);
    }

    // Build Ir
    let (model, derived_fields) = if let Some(tm) = raw.tree_model {
        // mining schema
        let mut active_fields = Vec::new();
        let mut target_field: Option<FieldId> = None;
        let mut mining_field_metas: Vec<FieldMeta> = Vec::new();
        for mf in &tm.mining_schema {
            let fid = *field_name_to_id.get(&mf.name).ok_or_else(|| PmmlError::MissingField(mf.name.clone()))?;
            let meta = field_meta_map.get(&fid).cloned().ok_or_else(|| PmmlError::MissingField(mf.name.clone()))?;
            match mf.usage_type.as_deref() {
                Some("target") => target_field = Some(fid),
                _ => active_fields.push(fid),
            }
            mining_field_metas.push(meta);
        }
        // if no mining schema explicitly says target, try to infer via DataDictionary target? For now keep as is.

        let mining_schema_ir = MiningSchemaIr {
            active_fields: active_fields.clone(),
            target_field,
            field_metas: mining_field_metas,
            missing_value_replacement: None,
        };

        // output
        let output_ir: Vec<OutputFieldIr> = tm.output.iter().map(|of| {
            let feature = of.feature.as_deref().unwrap_or("predictedValue").parse::<pmml_core::field::ResultFeature>().unwrap_or(pmml_core::field::ResultFeature::PredictedValue);
            let val = of.value.as_ref().map(|v| interner.intern_symbol(v));
            let field = of.name.parse::<String>().ok().and_then(|n| field_name_to_id.get(&n).copied());
            // Actually output field name is not necessarily data field; keep None for field
            OutputFieldIr { name: of.name.clone(), feature, value: val, field }
        }).collect();

        // flatten tree
        let mut nodes: Vec<NodeIr> = Vec::new();
        flatten_node(&tm.root, &mut interner, &field_meta_map, &field_name_to_id, &mut nodes)?;

        let tree_ir = TreeIr {
            function_name: tm.function_name.clone(),
            missing_value_strategy: parse_missing_strategy(tm.missing_value_strategy.as_deref()),
            no_true_child_strategy: parse_no_true_child(tm.no_true_child_strategy.as_deref()),
            nodes,
            mining_schema: mining_schema_ir,
            targets: vec![], // v1 skip Targets
            output: output_ir,
        };

        (ModelIr::Tree(tree_ir), vec![])
    } else {
        return Err(PmmlError::UnsupportedMarkup("no TreeModel found — only TreeModel supported in v1".into()));
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
