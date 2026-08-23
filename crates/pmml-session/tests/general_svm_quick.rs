use pmml_core::Value;
use pmml_session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

#[test]
fn general_regression_contrast() {
    let bytes = std::fs::read("bench/pmml/ContrastMatrixTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/ContrastMatrixTest.pmml"))
        .expect("general regression pmml not found");
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load gr");
    println!(
        "symbol_names: {:?}",
        sess.ir.symbol_names.values().collect::<Vec<_>>()
    );
    let mut m = HashMap::new();
    // gender f, educ 19, jobcat 3, salbegin 45000
    let sid_f = sess
        .ir
        .symbol_names
        .iter()
        .find(|(_, s)| *s == "f")
        .map(|(id, _)| *id)
        .unwrap();
    let sid_3 = sess
        .ir
        .symbol_names
        .iter()
        .find(|(_, s)| *s == "3")
        .map(|(id, _)| *id)
        .unwrap();
    m.insert("gender".to_string(), Value::Discrete(sid_f));
    m.insert("educ".to_string(), Value::Continuous(19.0));
    m.insert("jobcat".to_string(), Value::Discrete(sid_3));
    m.insert("salbegin".to_string(), Value::Continuous(45000.0));
    let out = sess.run(m).expect("run gr");
    println!("general regression out: {:?}", out);
    let pred = out.get("predictedValue").expect("predictedValue");
    match pred {
        Value::Discrete(sid) => {
            let s = sess.ir.symbol_names.get(sid).unwrap();
            println!("pred discrete {}", s);
            assert_eq!(s, "Low");
        }
        _ => panic!("expected discrete Low"),
    }
}

#[test]
fn svm_xor() {
    let bytes = std::fs::read("bench/pmml/VectorInstanceTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/VectorInstanceTest.pmml"))
        .expect("svm pmml not found");
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load svm");
    let mut m = HashMap::new();
    m.insert("x1".to_string(), Value::Continuous(0.0));
    m.insert("x2".to_string(), Value::Continuous(0.0));
    let out = sess.run(m).expect("run svm");
    println!("svm out: {:?}", out);
    assert!(!out.is_empty());
}
