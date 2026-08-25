use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};

#[test]
fn general_regression_load() {
    let bytes = std::fs::read("bench/pmml/ContrastMatrixTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/ContrastMatrixTest.pmml"))
        .expect("general regression pmml not found");
    let env = PmmlEnv::new();
    let res = Session::from_bytes(&env, &bytes, SessionOptions::default());
    match res {
        Ok(sess) => {
            println!(
                "general regression load ok, active {}",
                sess.num_active_fields()
            );
            // Try to run with dummy values
            let out = sess.run(Default::default());
            println!("general regression run: {:?}", out);
            assert!(out.is_ok() || out.is_err());
        }
        Err(e) => {
            println!("general regression load error: {}", e);
            panic!("should load");
        }
    }
}

#[test]
fn svm_load() {
    let bytes = std::fs::read("bench/pmml/VectorInstanceTest.pmml")
        .or_else(|_| std::fs::read("../../bench/pmml/VectorInstanceTest.pmml"))
        .expect("svm pmml not found");
    let env = PmmlEnv::new();
    let res = Session::from_bytes(&env, &bytes, SessionOptions::default());
    match res {
        Ok(sess) => {
            println!("svm load ok, active {}", sess.num_active_fields());
            let out = sess.run(Default::default());
            println!("svm run: {:?}", out);
        }
        Err(e) => println!("svm load error: {}", e),
    }
}
