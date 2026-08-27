use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

#[test]
fn association_load() {
    let bytes = std::fs::read("bench/pmml/AssociationOutputTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/AssociationOutputTest.pmml"))
        .expect("assoc pmml not found");
    let env = PmmlEnv::new();
    let res = Session::from_bytes(&env, &bytes, SessionOptions::default());
    match res {
        Ok(sess) => {
            println!("assoc load ok, active {}", sess.num_active_fields());
            let out = sess.run(&HashMap::default() as &dyn pmmlruntime::session::batch::Batch).unwrap().into_single().expect("run assoc");
            println!("assoc out: {out:?}");
            assert!(out.contains_key("predictedValue") || !out.is_empty());
        }
        Err(e) => {
            println!("assoc load error: {e}");
            panic!("should load");
        }
    }
}

#[test]
fn ruleset_load() {
    let bytes = std::fs::read("bench/pmml/SimpleRuleTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/SimpleRuleTest.pmml"))
        .expect("ruleset pmml not found");
    let env = PmmlEnv::new();
    let res = Session::from_bytes(&env, &bytes, SessionOptions::default());
    match res {
        Ok(sess) => {
            println!("ruleset load ok, active {}", sess.num_active_fields());
            let out = sess.run(&HashMap::default() as &dyn pmmlruntime::session::batch::Batch).unwrap().into_single().expect("run ruleset");
            println!("ruleset out: {out:?}");
            assert!(!out.is_empty());
        }
        Err(e) => {
            println!("ruleset load error: {e}");
            panic!("should load");
        }
    }
}
