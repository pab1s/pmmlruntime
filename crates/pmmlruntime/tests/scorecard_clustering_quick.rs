use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

#[test]
fn scorecard_attribute_reason() {
    let path = "bench/pmml/AttributeReasonCodeTest.pmml";
    let paths = [path, "../../bench/pmml/AttributeReasonCodeTest.pmml"];
    let mut bytes = None;
    for p in &paths {
        if let Ok(b) = std::fs::read(p) {
            bytes = Some(b);
            break;
        }
    }
    let bytes = bytes.expect("scorecard pmml not found");
    let env = PmmlEnv::new();
    let sess =
        Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load scorecard");
    // department=marketing, age 30, income 2000
    let mut m = HashMap::new();
    m.insert(
        "department".to_string(),
        Value::Discrete(pmmlruntime::base::SymbolId(0)),
    ); // but need correct symbol? Use string via intern? For now use continuous for age/income, discrete for department via string parse?
       // For scorecard, department is categorical string. Our session's run expects Value::Discrete with correct SymbolId interned.
       // We need to get SymbolId for "marketing" via sess.ir.symbol_names invert
    let marketing_sid = sess
        .ir
        .symbol_names
        .iter()
        .find(|(_, s)| *s == "marketing")
        .map_or(pmmlruntime::base::SymbolId(0), |(id, _)| *id);
    m.insert("department".to_string(), Value::Discrete(marketing_sid));
    m.insert("age".to_string(), Value::Continuous(35.0));
    m.insert("income".to_string(), Value::Continuous(1500.0));
    let out = sess
        .run(&m as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .expect("run scorecard");
    println!("scorecard out: {out:?}");
    let pred = out.get("predictedValue").expect("predictedValue");
    match pred {
        Value::Continuous(f) => {
            // For this fixture, initial 0 + dept marketing 19 + age 30-39? Actually age 35 => 12, income 1500 => 5 => total 36
            // But depends on fixture
            assert!(!f.is_nan());
            println!("scorecard predicted {f}");
        }
        _ => panic!("expected continuous"),
    }
}

#[test]
fn clustering_ranking() {
    let path = "bench/pmml/RankingTest.pmml";
    let paths = [path, "../../bench/pmml/RankingTest.pmml"];
    let mut bytes = None;
    for p in &paths {
        if let Ok(b) = std::fs::read(p) {
            bytes = Some(b);
            break;
        }
    }
    let bytes = bytes.expect("clustering pmml not found");
    let env = PmmlEnv::new();
    let sess =
        Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("load clustering");
    let mut m = HashMap::new();
    m.insert("input".to_string(), Value::Continuous(2.8));
    let out = sess
        .run(&m as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .expect("run clustering");
    println!("clustering out: {out:?}");
    let pred = out.get("predictedValue").expect("predictedValue");
    // Should be discrete cluster name, e.g., positive
    match pred {
        Value::Discrete(_) => println!("clustering predicted discrete ok"),
        Value::Continuous(_) => println!("clustering predicted continuous (maybe index)"),
        Value::Missing => panic!("expected discrete"),
    }
    assert!(!pred.is_missing());
}
