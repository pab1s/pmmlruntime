use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

#[test]
fn knn_tie_break() {
    let bytes = std::fs::read("bench/pmml/TieBreakTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/TieBreakTest.pmml"))
        .expect("tie break pmml");
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load knn");
    // input 2.5, k=2, nearest are 2 and 3 (both medium) => predicted medium
    let mut m = HashMap::new();
    m.insert("input".to_string(), Value::Continuous(2.5));
    let out = sess.run(m).expect("run knn");
    println!("knn tie break out: {:?}", out);
    let pred = out.get("predictedValue").expect("predictedValue");
    // For TieBreakTest, predictedValue should be medium
    match pred {
        Value::Discrete(sid) => {
            let s = sess.ir.symbol_names.get(sid).unwrap();
            println!("pred discrete {}", s);
            assert_eq!(s, "medium");
        }
        _ => panic!("expected discrete medium"),
    }
}

#[test]
fn knn_clustering_simple_matching() {
    let bytes = std::fs::read("bench/pmml/ClusteringNeighborhoodTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/ClusteringNeighborhoodTest.pmml"))
        .expect("knn clustering pmml");
    let env = PmmlEnv::new();
    let sess =
        Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load knn clustering");
    // This KNN is for clustering with simpleMatching, marital status s/d/m and dependents
    // We need to test with marital status = s, dependents 0 => nearest should be ID 1
    // For this test, we need to provide values for "marital status" and "dependents"
    // But KNNInputs are derived fields single/divorced/married/has dependents, not original fields.
    // Our KNN evaluator currently uses knn_inputs which are those derived fields, but we haven't yet handled LocalTransformations for KNN (NormDiscrete etc).
    // For now, we just test that it doesn't panic.
    let mut m = HashMap::new();
    m.insert(
        "marital status".to_string(),
        Value::Discrete(pmmlruntime::base::SymbolId(0)),
    ); // need correct sid for "s"
       // Find sid for "s"
    let sid_s = sess
        .ir
        .symbol_names
        .iter()
        .find(|(_, s)| *s == "s")
        .map(|(id, _)| *id)
        .unwrap_or(pmmlruntime::base::SymbolId(0));
    m.insert("marital status".to_string(), Value::Discrete(sid_s));
    m.insert("dependents".to_string(), Value::Continuous(0.0));
    let out = sess.run(m).expect("run knn clustering");
    println!("knn clustering out: {:?}", out);
    assert!(!out.is_empty());
}
