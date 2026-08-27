use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

#[test]
fn simple_nn() {
    let bytes = std::fs::read("bench/pmml/SimpleNeuralNetwork.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/SimpleNeuralNetwork.pmml"))
        .expect("nn pmml not found");
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load nn");
    let mut m = HashMap::new();
    m.insert("x1".to_string(), Value::Continuous(0.5));
    m.insert("x2".to_string(), Value::Continuous(0.5));
    let out = sess
        .run(&m as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .expect("run nn");
    println!("nn out: {out:?}");
    let pred = out.get("predictedValue").expect("predictedValue");
    match pred {
        Value::Continuous(f) => {
            // hidden1 logistic(1.0) 0.731, hidden2 logistic(0.5) 0.622, output 1.353
            assert!((f - 1.353).abs() < 0.05, "expected ~1.353 got {f}");
        }
        _ => panic!("expected continuous"),
    }
}
