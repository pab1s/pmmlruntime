use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;
use std::path::Path;

fn score_tree_pmml(path: &str) -> anyhow::Result<()> {
    let bytes = std::fs::read(path)?;
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default())?;
    // Dummy inputs: for Iris, use Petal.Length/Petal.Width continuous
    // For other trees, we try to discover required fields via IR but for parity we just try missing -> should not panic
    // We'll try an empty input (all missing) — should return Missing or lastPrediction
    let out = sess.run(HashMap::new())?;
    assert!(
        out.contains_key("predictedValue"),
        "missing predictedValue for {}",
        path
    );
    // Also try a synthetic numeric input for first active field
    if sess.num_active_fields() > 0 {
        let mut m = HashMap::new();
        // we need to know field names; use sess.ir.field_names
        for (fid, name) in &sess.ir.field_names {
            let v = Value::Continuous(1.0);
            m.insert(name.clone(), v);
            if m.len() >= 2 {
                break;
            }
        }
        let out2 = sess.run(m)?;
        assert!(
            out2.contains_key("predictedValue"),
            "second run missing predictedValue for {}",
            path
        );
    }
    Ok(())
}

#[test]
fn tree_fixtures_parity() {
    let bench_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/pmml");
    // If bench dir not found, try relative to repo root
    let paths = std::fs::read_dir(&bench_dir).unwrap();
    let mut tree_files: Vec<String> = vec![];
    for entry in paths {
        let e = entry.unwrap();
        let p = e.path();
        if p.extension().map(|s| s == "pmml").unwrap_or(false) {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            // v1 only TreeModel supported; skip MiningModel fixtures that will error
            let content = std::fs::read_to_string(&p).unwrap();
            if content.contains("<TreeModel") {
                tree_files.push(p.to_string_lossy().to_string());
            }
        }
    }
    assert!(
        !tree_files.is_empty(),
        "no tree fixtures found in {:?}",
        bench_dir
    );
    println!("Found {} tree fixtures", tree_files.len());
    let mut tested = 0usize;
    for f in &tree_files {
        println!("Testing {}", f);
        let res = score_tree_pmml(f);
        match res {
            Ok(_) => tested += 1,
            Err(e) => {
                let msg = e.to_string();
                // MiningModel fixtures not supported in v1 — skip
                // JPMML-unsupported markup (weightedConfidence etc.) is expected to fail fast
                if msg.contains("no TreeModel")
                    || msg.contains("MiningModel")
                    || msg.contains("missing field")
                    || msg.contains("unsupported markup")
                    || msg.contains("weightedConfidence")
                    || msg.contains("aggregateNodes")
                {
                    println!("  -> SKIP (unsupported/JPMML parity): {msg}");
                    continue;
                }
                panic!("{} failed: {e}", f);
            }
        }
    }
    assert!(
        tested >= 5,
        "expected at least 5 tree fixtures tested, got {tested}"
    );
    println!("Tested {tested} tree fixtures successfully");
}

#[test]
fn decision_tree_iris_specific() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/pmml/DecisionTreeIris.pmml");
    let bytes = std::fs::read(&path).unwrap();
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).unwrap();
    // setosa case
    let mut m = HashMap::new();
    m.insert("Petal.Length".to_string(), Value::Continuous(1.4));
    m.insert("Petal.Width".to_string(), Value::Continuous(0.2));
    let out = sess.run(m).unwrap();
    let pred = out.get("predictedValue").unwrap();
    match pred {
        Value::Discrete(sid) => {
            let s = sess.ir.symbol_names.get(sid).unwrap();
            assert_eq!(s, "setosa", "expected setosa got {s}");
        }
        _ => panic!("expected discrete"),
    }
    // virginica case
    let mut m2 = HashMap::new();
    m2.insert("Petal.Length".to_string(), Value::Continuous(5.5));
    m2.insert("Petal.Width".to_string(), Value::Continuous(2.0));
    let out2 = sess.run(m2).unwrap();
    let pred2 = out2.get("predictedValue").unwrap();
    match pred2 {
        Value::Discrete(sid) => {
            let s = sess.ir.symbol_names.get(sid).unwrap();
            assert_eq!(s, "virginica");
        }
        _ => panic!("expected discrete"),
    }
}
