#![allow(clippy::unreadable_literal)]
use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
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
    let out = sess.run(&m as &dyn pmmlruntime::session::batch::Batch).unwrap().into_single().expect("run gr");
    println!("general regression out: {out:?}");
    let pred = out.get("predictedValue").expect("predictedValue");
    match pred {
        Value::Discrete(sid) => {
            let s = sess.ir.symbol_names.get(sid).unwrap();
            println!("pred discrete {s}");
            assert_eq!(s, "Low");
        }
        _ => panic!("expected discrete Low"),
    }
    // Check probabilities: expected 0.81956470 for Low, 0.18043530 for High
    let prob_low = out
        .get("Probability_Low")
        .or_else(|| out.get("Low"))
        .expect("Probability_Low");
    if let Value::Continuous(p) = prob_low {
        assert!(
            (p - 0.81956470).abs() < 1e-6,
            "expected Probability_Low 0.819 got {p}"
        );
    } else {
        panic!("expected continuous prob");
    }
    let prob_high = out
        .get("Probability_High")
        .or_else(|| out.get("High"))
        .expect("Probability_High");
    if let Value::Continuous(p) = prob_high {
        assert!(
            (p - 0.18043530).abs() < 1e-6,
            "expected Probability_High 0.180 got {p}"
        );
    }
}

#[test]
fn svm_xor() {
    let bytes = std::fs::read("bench/pmml/VectorInstanceTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/VectorInstanceTest.pmml"))
        .expect("svm pmml not found");
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load svm");
    // Test all 4 XOR cases
    let cases = vec![
        (0.0, 0.0, 0.1004236),
        (0.0, 1.0, 0.8995764),
        (1.0, 0.0, 0.8995764),
        (1.0, 1.0, 0.1004236),
    ];
    for (x1, x2, expected) in cases {
        let mut m = HashMap::new();
        m.insert("x1".to_string(), Value::Continuous(x1));
        m.insert("x2".to_string(), Value::Continuous(x2));
        let out = sess.run(&m as &dyn pmmlruntime::session::batch::Batch).unwrap().into_single().expect("run svm");
        println!("svm ({x1}, {x2}) out: {out:?}");
        let pred = out
            .get("predictedValue")
            .or_else(|| out.get("class"))
            .expect("predicted");
        if let Value::Continuous(p) = pred {
            assert!(
                (p - expected).abs() < 1e-5,
                "for ({x1}, {x2}) expected {expected} got {p}"
            );
        } else {
            panic!("expected continuous");
        }
    }
}
