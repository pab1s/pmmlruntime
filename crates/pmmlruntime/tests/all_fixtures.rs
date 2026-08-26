use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::path::Path;

#[test]
fn all_fixtures_load() {
    let bench_dir = Path::new("bench/pmml");
    let bench_dir_alt = Path::new("../../bench/pmml");
    let dir = if bench_dir.exists() {
        bench_dir.to_path_buf()
    } else {
        bench_dir_alt.to_path_buf()
    };
    let mut total = 0;
    let mut ok = 0;
    let mut failed = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let e = entry.unwrap();
        let path = e.path();
        if path.extension().map(|s| s == "pmml").unwrap_or(false) {
            total += 1;
            let bytes = std::fs::read(&path).unwrap();
            let env = PmmlEnv::new();
            let res = Session::from_bytes(&env, &bytes, SessionOptions::default());
            match res {
                Ok(sess) => {
                    // Try to run with empty input (should not panic)
                    let out = sess.run(Default::default());
                    match out {
                        Ok(_) => {
                            println!("OK {}", path.display());
                            ok += 1;
                        }
                        Err(e) => {
                            println!("RUN FAIL {}: {}", path.display(), e);
                            failed.push(format!("{}: run {}", path.display(), e));
                            ok += 1; // still consider load ok
                        }
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    // JPMML-unsupported markup (weightedConfidence/aggregateNodes/distributionBased etc.)
                    // is expected to fail fast — treat as SKIP not FAIL (parity with UnsupportedMarkupInspector)
                    if msg.contains("unsupported markup") {
                        println!("SKIP (unsupported) {}: {}", path.display(), e);
                        ok += 1;
                    } else {
                        println!("LOAD FAIL {}: {}", path.display(), e);
                        failed.push(format!("{}: load {}", path.display(), e));
                    }
                }
            }
        }
    }
    println!("Total {}, ok {}, failed {}", total, ok, failed.len());
    for f in &failed {
        println!("FAILED: {}", f);
    }
    assert!(failed.is_empty(), "Some fixtures failed: {:?}", failed);
    assert!(total >= 44, "Expected at least 44 fixtures, got {}", total);
}
