use pmml_core::Value;
use pmml_ir::ir::{ClusterIr, ClusteringIr};

pub fn evaluate_clustering(clustering: &ClusteringIr, values: &[Value]) -> Value {
    if clustering.clusters.is_empty() || clustering.clustering_fields.is_empty() {
        return Value::Missing;
    }

    let mut input_vec: Vec<f64> = Vec::new();
    for &fid in &clustering.clustering_fields {
        let idx = fid.as_usize();
        let v = if idx < values.len() {
            values[idx]
        } else {
            Value::Missing
        };
        match v {
            Value::Continuous(f) => input_vec.push(f),
            Value::Missing => return Value::Missing,
            Value::Discrete(_) => return Value::Missing,
        }
    }

    let mut best_idx: Option<usize> = None;
    let mut best_dist = f64::INFINITY;

    for (i, cluster) in clustering.clusters.iter().enumerate() {
        let dist = distance(&input_vec, &cluster.array, &clustering.comparison_measure);
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(i);
        }
    }

    if let Some(idx) = best_idx {
        return Value::Discrete(clustering.clusters[idx].name);
    }

    Value::Missing
}

fn distance(a: &[f64], b: &[f64], measure: &str) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    match measure {
        "squaredEuclidean" => a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum(),
        "euclidean" => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt(),
        "manhattan" => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        "chebyshev" => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max),
        _ => a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::{FieldId, SymbolId, Value};
    use pmml_ir::ir::*;

    #[test]
    fn clustering_1d() {
        let f = FieldId(0);
        let s_neg = SymbolId(0);
        let s_neu = SymbolId(1);
        let s_pos = SymbolId(2);
        let clustering = ClusteringIr {
            function_name: "clustering".into(),
            model_class: "centerBased".into(),
            number_of_clusters: 3,
            mining_schema: MiningSchemaIr {
                active_fields: vec![f],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            comparison_measure: "squaredEuclidean".into(),
            clustering_fields: vec![f],
            clusters: vec![
                ClusterIr {
                    name: s_neg,
                    name_str: "negative".into(),
                    array: vec![-3.0],
                },
                ClusterIr {
                    name: s_neu,
                    name_str: "neutral".into(),
                    array: vec![0.0],
                },
                ClusterIr {
                    name: s_pos,
                    name_str: "positive".into(),
                    array: vec![3.0],
                },
            ],
            output: vec![],
        };
        let vals = vec![Value::Continuous(2.8)];
        let pred = evaluate_clustering(&clustering, &vals);
        assert_eq!(pred, Value::Discrete(s_pos));
        let vals2 = vec![Value::Continuous(-2.9)];
        let pred2 = evaluate_clustering(&clustering, &vals2);
        assert_eq!(pred2, Value::Discrete(s_neg));
    }
}
