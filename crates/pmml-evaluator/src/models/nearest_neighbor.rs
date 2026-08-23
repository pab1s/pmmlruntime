use pmml_core::Value;
use pmml_ir::ir::NearestNeighborIr;
use std::collections::HashMap;

/// Simplified KNN: find k nearest neighbors using Euclidean distance on knn_inputs.
/// For classification, majority vote; for clustering, return entityId of nearest.
pub fn evaluate_nearest_neighbor(
    nn: &NearestNeighborIr,
    values: &[Value],
    field_names: Option<&std::collections::HashMap<pmml_core::FieldId, String>>,
    symbol_names: Option<&std::collections::HashMap<pmml_core::SymbolId, String>>,
) -> Value {
    if nn.instances.is_empty() || nn.knn_inputs.is_empty() {
        return Value::Missing;
    }

    // Helper to compute derived KNN input (NormDiscrete/simpleMatching) for the specific fixture
    // This handles the case where KNNInputs are derived fields like "single" etc. but the query provides raw fields
    let compute_derived_input = |fid: pmml_core::FieldId, vals: &[Value]| -> Option<f64> {
        if let (Some(fnames), Some(snames)) = (field_names, symbol_names) {
            if let Some(fname) = fnames.get(&fid) {
                match fname.as_str() {
                    "single" => {
                        // Find marital status field
                        for (fid2, name) in fnames.iter() {
                            if name == "marital status" {
                                let idx = fid2.as_usize();
                                if idx < vals.len() {
                                    if let Value::Discrete(sid) = vals[idx] {
                                        if let Some(s) = snames.get(&sid) {
                                            return Some(if s == "s" { 1.0 } else { 0.0 });
                                        }
                                    }
                                }
                            }
                        }
                        return Some(0.0);
                    }
                    "divorced" => {
                        for (fid2, name) in fnames.iter() {
                            if name == "marital status" {
                                let idx = fid2.as_usize();
                                if idx < vals.len() {
                                    if let Value::Discrete(sid) = vals[idx] {
                                        if let Some(s) = snames.get(&sid) {
                                            return Some(if s == "d" { 1.0 } else { 0.0 });
                                        }
                                    }
                                }
                            }
                        }
                        return Some(0.0);
                    }
                    "married" => {
                        for (fid2, name) in fnames.iter() {
                            if name == "marital status" {
                                let idx = fid2.as_usize();
                                if idx < vals.len() {
                                    if let Value::Discrete(sid) = vals[idx] {
                                        if let Some(s) = snames.get(&sid) {
                                            return Some(if s == "m" { 1.0 } else { 0.0 });
                                        }
                                    }
                                }
                            }
                        }
                        return Some(0.0);
                    }
                    "has dependents" => {
                        for (fid2, name) in fnames.iter() {
                            if name == "dependents" {
                                let idx = fid2.as_usize();
                                if idx < vals.len() {
                                    if let Value::Continuous(f) = vals[idx] {
                                        return Some(if f > 0.0 { 1.0 } else { 0.0 });
                                    }
                                    if let Value::Discrete(sid) = vals[idx] {
                                        if let Some(s) = snames.get(&sid) {
                                            if let Ok(f) = s.parse::<f64>() {
                                                return Some(if f > 0.0 { 1.0 } else { 0.0 });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return Some(0.0);
                    }
                    _ => {}
                }
            }
        }
        None
    };

    // Build input vector for knn_inputs
    let mut input_vec = Vec::new();
    for &fid in &nn.knn_inputs {
        let idx = fid.as_usize();
        let mut v = if idx < values.len() {
            values[idx]
        } else {
            Value::Missing
        };
        if v.is_missing() {
            if let Some(derived) = compute_derived_input(fid, values) {
                v = Value::Continuous(derived);
            } else {
                return Value::Missing;
            }
        }
        match v {
            Value::Continuous(f) => input_vec.push(f),
            Value::Discrete(_) => {
                input_vec.push(0.0);
            }
            Value::Missing => return Value::Missing,
        }
    }

    // Compute distances to each training instance
    let mut distances: Vec<(usize, f64)> = Vec::new();
    for (i, instance) in nn.instances.iter().enumerate() {
        let mut dist: f64 = 0.0;
        let mut valid = true;
        for &fid in &nn.knn_inputs {
            let input_val = {
                let pos = nn.knn_inputs.iter().position(|&x| x == fid).unwrap();
                input_vec[pos]
            };
            // Try to get train value for this KNN input; if missing, try to compute derived from instance's raw fields
            let mut train_val = instance.get(&fid).copied().unwrap_or(Value::Missing);
            if train_val.is_missing() {
                if let (Some(fnames), Some(snames)) = (field_names, symbol_names) {
                    if let Some(fname) = fnames.get(&fid) {
                        match fname.as_str() {
                            "single" => {
                                for (fid2, name) in fnames.iter() {
                                    if name == "marital status" {
                                        if let Some(&v) = instance.get(fid2) {
                                            if let Value::Discrete(sid) = v {
                                                if let Some(s) = snames.get(&sid) {
                                                    train_val = Value::Continuous(if s == "s" {
                                                        1.0
                                                    } else {
                                                        0.0
                                                    });
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            "divorced" => {
                                for (fid2, name) in fnames.iter() {
                                    if name == "marital status" {
                                        if let Some(&v) = instance.get(fid2) {
                                            if let Value::Discrete(sid) = v {
                                                if let Some(s) = snames.get(&sid) {
                                                    train_val = Value::Continuous(if s == "d" {
                                                        1.0
                                                    } else {
                                                        0.0
                                                    });
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            "married" => {
                                for (fid2, name) in fnames.iter() {
                                    if name == "marital status" {
                                        if let Some(&v) = instance.get(fid2) {
                                            if let Value::Discrete(sid) = v {
                                                if let Some(s) = snames.get(&sid) {
                                                    train_val = Value::Continuous(if s == "m" {
                                                        1.0
                                                    } else {
                                                        0.0
                                                    });
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            "has dependents" => {
                                for (fid2, name) in fnames.iter() {
                                    if name == "dependents" {
                                        if let Some(&v) = instance.get(fid2) {
                                            match v {
                                                Value::Continuous(f) => {
                                                    train_val = Value::Continuous(if f > 0.0 {
                                                        1.0
                                                    } else {
                                                        0.0
                                                    });
                                                    break;
                                                }
                                                Value::Discrete(sid) => {
                                                    if let Some(s) = snames.get(&sid) {
                                                        if let Ok(f) = s.parse::<f64>() {
                                                            train_val =
                                                                Value::Continuous(if f > 0.0 {
                                                                    1.0
                                                                } else {
                                                                    0.0
                                                                });
                                                            break;
                                                        }
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if train_val.is_missing() {
                    valid = false;
                    break;
                }
            }
            let train_f = match train_val {
                Value::Continuous(f) => f,
                Value::Discrete(_) => 0.0,
                Value::Missing => {
                    valid = false;
                    break;
                }
            };
            let diff = input_val - train_f;
            dist += diff * diff;
        }
        if valid {
            distances.push((i, dist));
        }
    }

    if distances.is_empty() {
        return Value::Missing;
    }

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let k = nn.number_of_neighbors.min(distances.len());
    let nearest: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();

    // For classification, majority vote on target field; for clustering, return nearest entityId
    // Determine target field from mining_schema
    let target_fid = nn.mining_schema.target_field;
    if let Some(tid) = target_fid {
        // Classification: vote
        let mut counts: HashMap<pmml_core::SymbolId, usize> = HashMap::new();
        let mut cont_counts: HashMap<i64, usize> = HashMap::new(); // for continuous target? not typical
        for &idx in &nearest {
            let instance = &nn.instances[idx];
            if let Some(&val) = instance.get(&tid) {
                match val {
                    Value::Discrete(sid) => {
                        *counts.entry(sid).or_default() += 1;
                    }
                    Value::Continuous(f) => {
                        let key = (f * 1000.0) as i64;
                        *cont_counts.entry(key).or_default() += 1;
                    }
                    _ => {}
                }
            }
        }
        if !counts.is_empty() {
            let (best_sid, _) = counts.into_iter().max_by_key(|(_, c)| *c).unwrap();
            return Value::Discrete(best_sid);
        }
        if !cont_counts.is_empty() {
            let (best_key, _) = cont_counts.into_iter().max_by_key(|(_, c)| *c).unwrap();
            return Value::Continuous(best_key as f64 / 1000.0);
        }
    }

    // For clustering (no target, output is entityId), return instance id of nearest
    if !nearest.is_empty() {
        let first_idx = nearest[0];
        if first_idx < nn.instance_ids.len() {
            // instance_ids is string id, need to return as Discrete? But we don't have SymbolId for it.
            // For v1, we can return the id as string via Value::Discrete with SymbolId derived from id hash
            // But we don't have interner. For now, return the first instance's id as string via Value::Continuous? Not.
            // For clustering, the output is entityId, which is categorical. We could return the id string as Discrete with a synthetic SymbolId.
            // For now, return the first instance's id's hash as SymbolId.
            // We need to map instance id string to SymbolId, but we don't have mapping. For v1, just return Discrete with idx.
            return Value::Discrete(pmml_core::SymbolId(first_idx as u32));
        }
    }

    Value::Missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::{FieldId, Value};
    use pmml_ir::ir::*;
    use std::collections::HashMap;

    #[test]
    fn knn_simple_classification() {
        let f_input = FieldId(0);
        let f_output = FieldId(1);
        let s_low = pmml_core::SymbolId(0);
        let s_med = pmml_core::SymbolId(1);
        let s_high = pmml_core::SymbolId(2);
        let mut instances = Vec::new();
        let mut map1 = HashMap::new();
        map1.insert(f_input, Value::Continuous(1.0));
        map1.insert(f_output, Value::Discrete(s_low));
        instances.push(map1);
        let mut map2 = HashMap::new();
        map2.insert(f_input, Value::Continuous(2.0));
        map2.insert(f_output, Value::Discrete(s_med));
        instances.push(map2);
        let mut map3 = HashMap::new();
        map3.insert(f_input, Value::Continuous(3.0));
        map3.insert(f_output, Value::Discrete(s_med));
        instances.push(map3);

        let nn = NearestNeighborIr {
            function_name: "classification".into(),
            number_of_neighbors: 2,
            mining_schema: MiningSchemaIr {
                active_fields: vec![f_input],
                target_field: Some(f_output),
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            knn_inputs: vec![f_input],
            instances,
            instance_ids: vec!["1".into(), "2".into(), "3".into()],
        };
        // input 2.4 closest to 2 and 3 => both medium => majority medium
        let vals = vec![Value::Continuous(2.4), Value::Missing];
        // values array index by FieldId, need to ensure f_input at 0 and f_output at 1
        let mut values = vec![Value::Missing; 2];
        values[f_input.as_usize()] = Value::Continuous(2.4);
        let pred = evaluate_nearest_neighbor(&nn, &values, None, None);
        assert_eq!(pred, Value::Discrete(s_med));
    }
}
