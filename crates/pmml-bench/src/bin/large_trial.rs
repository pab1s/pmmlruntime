use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pmml_core::Value;
use pmml_session::{ExecutionProviderKind, PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn load_iris_session(kind: ExecutionProviderKind) -> Session {
    let xml = std::fs::read("/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/DecisionTreeIris.pmml").unwrap();
    let env = PmmlEnv::new();
    let opts = SessionOptions::default().execution_provider(kind);
    Session::from_bytes(&env, &xml, opts).unwrap()
}

fn load_regression_session(kind: ExecutionProviderKind) -> Option<Session> {
    // Try a few regression fixtures
    let candidates = [
        "/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/LinearRegression.pmml",
        "/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/Regression.pmml",
        "/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/AutoRegressive.pmml",
    ];
    for path in candidates {
        if let Ok(xml) = std::fs::read(path) {
            let env = PmmlEnv::new();
            let opts = SessionOptions::default().execution_provider(kind);
            if let Ok(sess) = Session::from_bytes(&env, &xml, opts) {
                return Some(sess);
            }
        }
    }
    // Fallback: use Iris as regression not available, return None and skip regression bench
    None
}

fn make_hash_batch(n: usize) -> Vec<HashMap<String, Value>> {
    (0..n)
        .map(|i| {
            let mut m = HashMap::new();
            let v = 1.0 + (i % 5) as f64;
            m.insert("Petal.Length".to_string(), Value::Continuous(v));
            m.insert("Petal.Width".to_string(), Value::Continuous(v * 0.5));
            m
        })
        .collect()
}

fn make_arrow_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("Petal.Length", DataType::Float64, true),
        Field::new("Petal.Width", DataType::Float64, true),
    ]));
    let mut len_vals = Vec::with_capacity(n);
    let mut wid_vals = Vec::with_capacity(n);
    for i in 0..n {
        let v = 1.0 + (i % 5) as f64;
        len_vals.push(v);
        wid_vals.push(v * 0.5);
    }
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(len_vals)) as _,
            Arc::new(Float64Array::from(wid_vals)) as _,
        ],
    )
    .unwrap()
}

fn time<F: FnOnce() -> T, T>(f: F) -> (T, Duration) {
    let start = Instant::now();
    let res = f();
    let dur = start.elapsed();
    (res, dur)
}

fn fmt_thr(rows: usize, dur: Duration) -> String {
    let secs = dur.as_secs_f64();
    let thr = rows as f64 / secs;
    let per_row_ns = secs * 1e9 / rows as f64;
    format!(
        "{:.2} ms total | {:.0} rows/sec | {:.0} ns/row",
        secs * 1000.0,
        thr,
        per_row_ns
    )
}

fn run_trial(size: usize) {
    println!("\n=== {} rows ===", size);
    // For n >= 1M, skip HashMap batch to avoid OOM (HashMap per row ~200B * 1M = 200MB + overhead)
    let use_hash = size <= 100_000;
    let arrow_batch = make_arrow_batch(size);
    println!(
        "Arrow batch created: {} rows, schema {:?}",
        arrow_batch.num_rows(),
        arrow_batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
    );

    // Test both providers
    for kind in [
        ExecutionProviderKind::CpuSerial,
        ExecutionProviderKind::CpuBatched,
    ] {
        let kind_str = match kind {
            ExecutionProviderKind::CpuSerial => "CpuSerial",
            ExecutionProviderKind::CpuBatched => "CpuBatched",
        };
        let sess = load_iris_session(kind);
        // Warmup single row
        let mut single = HashMap::new();
        single.insert("Petal.Length".to_string(), Value::Continuous(1.4));
        single.insert("Petal.Width".to_string(), Value::Continuous(0.2));
        let _ = sess.run(single.clone()).unwrap();

        // Single row loop (for reference)
        if size <= 10_000 {
            let (_, dur) = time(|| {
                for _ in 0..size {
                    let _ = sess.run(single.clone()).unwrap();
                }
            });
            println!(
                "  {} single-row loop ({}x run): {}",
                kind_str,
                size,
                fmt_thr(size, dur)
            );
        }

        if use_hash {
            let hash_batch = make_hash_batch(size);
            // Hash batch sequential (via run_batch with CpuSerial) and parallel (CpuBatched)
            let (res, dur) = time(|| sess.run_batch(hash_batch.clone()).unwrap());
            assert_eq!(res.len(), size);
            println!(
                "  {} HashMap run_batch (Vec<HashMap> {} rows): {}",
                kind_str,
                size,
                fmt_thr(size, dur)
            );
            // Also test run_batch_ref
            let hash_ref = make_hash_batch(size);
            let (res2, dur2) = time(|| sess.run_batch_ref(&hash_ref).unwrap());
            assert_eq!(res2.len(), size);
            println!(
                "  {} HashMap run_batch_ref (slice {} rows): {}",
                kind_str,
                size,
                fmt_thr(size, dur2)
            );
        } else {
            println!(
                "  {} HashMap run_batch: SKIPPED for {} rows (would OOM)",
                kind_str, size
            );
        }

        // Arrow batch - chunked for large sizes to avoid OOM (Vec<HashMap> of 10M would be >2GB)
        if size >= 1_000_000 {
            let chunk_size = 100_000;
            let mut total_dur = Duration::ZERO;
            for chunk_start in (0..size).step_by(chunk_size) {
                let len = (size - chunk_start).min(chunk_size);
                let chunk_batch = arrow_batch.slice(chunk_start, len);
                let (res, dur) = time(|| sess.run_batch_arrow(&chunk_batch).unwrap());
                assert_eq!(res.len(), len);
                total_dur += dur;
            }
            println!(
                "  {} Arrow run_batch_arrow chunked (RecordBatch {} rows, chunk 100k): {}",
                kind_str,
                size,
                fmt_thr(size, total_dur)
            );
            let mut total_dur2 = Duration::ZERO;
            for chunk_start in (0..size).step_by(chunk_size) {
                let len = (size - chunk_start).min(chunk_size);
                let chunk_batch = arrow_batch.slice(chunk_start, len);
                let (out_batch, dur) = time(|| sess.run_record_batch(&chunk_batch).unwrap());
                assert_eq!(out_batch.num_rows(), len);
                total_dur2 += dur;
            }
            println!(
                "  {} Arrow run_record_batch chunked ({} rows, chunk 100k): {}",
                kind_str,
                size,
                fmt_thr(size, total_dur2)
            );
        } else {
            let (res, dur) = time(|| sess.run_batch_arrow(&arrow_batch).unwrap());
            assert_eq!(res.len(), size);
            println!(
                "  {} Arrow run_batch_arrow (RecordBatch {} rows): {}",
                kind_str,
                size,
                fmt_thr(size, dur)
            );

            // Arrow to Arrow
            let (out_batch, dur2) = time(|| sess.run_record_batch(&arrow_batch).unwrap());
            assert_eq!(out_batch.num_rows(), size);
            println!(
                "  {} Arrow run_record_batch -> RecordBatch ({} rows, {} cols): {}",
                kind_str,
                size,
                out_batch.num_columns(),
                fmt_thr(size, dur2)
            );
        }
    }

    // Regression SIMD trial if available (only for small sizes to avoid confusion)
    if size <= 100_000 {
        if let Some(sess) = load_regression_session(ExecutionProviderKind::CpuBatched) {
            println!("  Regression fixture found for SIMD check ({} rows)", size);
            // For regression, we need a batch with appropriate fields. Try to use same Iris arrow batch but it will have missing fields for regression model
            // So we generate a generic batch with fields that match regression's expected input? Instead we just test with Iris-like batch but it will mostly be Missing
            // For proper SIMD we need a real regression fixture with Float64 fields. Let's just try a simple synthetic regression via pmml-evaluator simd direct test
            // For now, test the simd module directly with synthetic regression model
            {
                use pmml_core::FieldId;
                use pmml_ir::ir::*;
                let f0 = FieldId(0);
                let reg = pmml_ir::ir::RegressionIr {
                    function_name: "regression".into(),
                    mining_schema: MiningSchemaIr {
                        active_fields: vec![f0],
                        target_field: None,
                        field_metas: vec![],
                        missing_value_replacement: None,
                    },
                    regression_tables: vec![RegressionTableIr {
                        intercept: 1.0,
                        target_category: None,
                        numeric_predictors: vec![NumericPredictorIr {
                            field: f0,
                            coefficient: 2.0,
                            exponent: 1,
                        }],
                        categorical_predictors: vec![],
                    }],
                    normalization_method: RegressionNormalizationMethod::None,
                    targets: vec![],
                    output: vec![],
                };
                let n = size.min(10000); // cap for synthetic
                let rows: Vec<Vec<Value>> =
                    (0..n).map(|i| vec![Value::Continuous(i as f64)]).collect();
                let refs: Vec<&[Value]> = rows.iter().map(|r| r.as_slice()).collect();
                let start = Instant::now();
                let simd_out = pmml_evaluator::simd::evaluate_regression_batch_simd(&reg, &refs);
                let dur = start.elapsed();
                let scalar_start = Instant::now();
                let scalar_out =
                    pmml_evaluator::simd::evaluate_regression_batch_scalar(&reg, &refs);
                let scalar_dur = scalar_start.elapsed();
                assert_eq!(simd_out, scalar_out);
                let speedup = if dur.as_secs_f64() > 0.0 {
                    scalar_dur.as_secs_f64() / dur.as_secs_f64()
                } else {
                    1.0
                };
                println!(
                    "  SIMD regression synthetic {} rows: SIMD {} vs scalar {} (speedup {:.2}x)",
                    n,
                    fmt_thr(n, dur),
                    fmt_thr(n, scalar_dur),
                    speedup
                );
            }
            let _ = sess;
        } else {
            println!("  No regression fixture found, skipping regression SIMD trial");
        }
    }
}

fn main() {
    println!("PMML Large Batch Trial — Tree Iris (2 Float64 fields)");
    println!("Host: {} threads rayon", rayon::current_num_threads());
    let simd_enabled = {
        // Check if pmml-evaluator simd is available at runtime (wide always available, but feature gates the 4-wide path)
        // For benchmark, we just report that binary was built with simd feature if pmml-evaluator was built with it;
        // here we detect via cfg in pmml-evaluator crate by checking an env var is not reliable, so just print false/true based on whether the simd module's 4-wide path would be taken for regression batch >=4
        // We approximate by checking if the `wide` crate is linked (always) - so report true if we are running a release build with avx2 available
        #[cfg(target_feature = "avx")]
        {
            true
        }
        #[cfg(not(target_feature = "avx"))]
        {
            false
        }
    };
    println!("Host AVX: {}", simd_enabled);
    let sizes = [10_000, 100_000, 1_000_000, 10_000_000];
    for &size in &sizes {
        // For 10M, we need to be careful about memory and time. Do a single run, not multiple.
        // Also for 10M, HashMap path is skipped.
        run_trial(size);
        // Force GC between sizes
        println!("--- done {} ---\n", size);
    }
    println!("All trials done.");
}
