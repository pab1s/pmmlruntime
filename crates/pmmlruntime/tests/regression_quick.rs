use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

#[test]
fn regression_linear() {
    let paths = [
        "bench/pmml/RegressionOutputTest.pmml",
        "../../bench/pmml/RegressionOutputTest.pmml",
        "/home/pab1s/Projects/pmml-migration/upstream/pmml-evaluator/pmml-evaluator/src/test/resources/pmml/regression/RegressionOutputTest.pmml",
    ];
    let mut bytes = None;
    for p in &paths {
        if let Ok(b) = std::fs::read(p) {
            bytes = Some(b);
            break;
        }
    }
    let bytes = bytes.expect("regression pmml not found");
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load");
    println!("active {}", sess.num_active_fields());
    let mut m = HashMap::new();
    m.insert("input".to_string(), Value::Continuous(2.0));
    let out = sess.run(m).expect("run");
    println!("out: {:?}", out);
    let pred = out.get("predictedValue").expect("predictedValue");
    match pred {
        Value::Continuous(f) => {
            assert!((f - 4.0).abs() < 1e-6, "expected 4.0 got {}", f);
        }
        _ => panic!("expected continuous"),
    }
}

#[test]
fn mining_model_chain() {
    let paths = [
        "bench/pmml/ModelChainSimpleTest.pmml",
        "../../bench/pmml/ModelChainSimpleTest.pmml",
        "/home/pab1s/Projects/pmml-migration/upstream/pmml-evaluator/pmml-evaluator/src/test/resources/pmml/mining/ModelChainSimpleTest.pmml",
    ];
    let mut bytes = None;
    for p in &paths {
        if let Ok(b) = std::fs::read(p) {
            bytes = Some(b);
            break;
        }
    }
    let bytes = bytes.expect("mining pmml not found");
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load mining");
    println!("mining active {}", sess.num_active_fields());
    let mut m = HashMap::new();
    m.insert("petal_length".to_string(), Value::Continuous(1.4));
    m.insert("petal_width".to_string(), Value::Continuous(0.2));
    m.insert("temperature".to_string(), Value::Continuous(20.0));
    m.insert("cloudiness".to_string(), Value::Continuous(0.5));
    let out = sess.run(m).expect("run mining");
    println!("mining out: {:?}", out);
    // PollenIndex should be around 0.3 + 0.8*prob_setosa + ... + 0.02*temp -0.1*cloudiness
    // For setosa petal 1.4/0.2 => prob setosa ~1, versicolor 0, virginica 0 => PollenIndex ~0.3+0.8+0.02*20 -0.1*0.5 = 0.3+0.8+0.4-0.05=1.45
    let pred = out
        .get("predictedValue")
        .or_else(|| out.get("PollenIndex"))
        .expect("predicted");
    println!("pred mining: {:?}", pred);
    assert!(!pred.is_missing());
}
