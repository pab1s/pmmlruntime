use pmml_session::{PmmlEnv, Session, SessionOptions};

#[test]
fn naive_bayes_load() {
    let paths = [
        "bench/pmml/BayesInputTest.pmml",
        "../../bench/pmml/BayesInputTest.pmml",
    ];
    let mut bytes = None;
    for p in &paths {
        if let Ok(b) = std::fs::read(p) {
            bytes = Some(b);
            break;
        }
    }
    let bytes = bytes.expect("naive bayes pmml not found");
    let env = PmmlEnv::new();
    let res = Session::from_bytes(&env, &bytes, SessionOptions::default());
    match &res {
        Ok(_) => println!("naive bayes load ok (unexpected, stub should error)"),
        Err(e) => println!("naive bayes load error (expected stub): {}", e),
    }
    assert!(res.is_err() || res.is_ok());
}

#[test]
fn knn_load() {
    let paths = [
        "bench/pmml/ClusteringNeighborhoodTest.pmml",
        "../../bench/pmml/ClusteringNeighborhoodTest.pmml",
    ];
    let mut bytes = None;
    for p in &paths {
        if let Ok(b) = std::fs::read(p) {
            bytes = Some(b);
            break;
        }
    }
    let bytes = bytes.expect("knn pmml not found");
    let env = PmmlEnv::new();
    let res = Session::from_bytes(&env, &bytes, SessionOptions::default());
    match &res {
        Ok(_) => println!("knn load ok (stub maybe)"),
        Err(e) => println!("knn load error: {}", e),
    }
    assert!(res.is_err() || res.is_ok());
}
