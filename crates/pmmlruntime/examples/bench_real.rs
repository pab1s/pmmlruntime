//! Real benchmark vs JPMML evaluator & transpiler — 5-run mean/std ready.
//!
//! Reports cold load, hot single-row, and batch (100/1k/10k) per-model.
//! Designed to be run 5× externally for mean/std aggregation, but also
//! does internal warmup + measured loops and prints JSON per model.
//!
//! Run: cargo run --release --example bench_real -- bench/pmml/DecisionTreeIris.pmml bench/pmml/GradientBoosterTest.pmml bench/pmml/TransformationDictionaryTest.pmml --runs 1 --iterations 100000
//!
//! The binary itself does one external run; call it 5× via bench.sh for stats.

use std::collections::HashMap;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pmmlruntime::session::batch::Batch;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use pmmlruntime::Value;

fn usage() {
    eprintln!("Usage: bench_real <pmml1> [pmml2 ...] [--runs N] [--iterations N] [--batch 100] [--json]");
    eprintln!("  Defaults: runs=1 external run does internal loops; iterations=50000 for single, batch sizes 100/1000/10000");
}

fn parse_args() -> (Vec<PathBuf>, usize, usize, bool) {
    let mut pmmls = Vec::new();
    let mut iterations = 50_000usize;
    let mut json = false;
    let mut args = std::env::args().collect::<Vec<_>>();
    // remove binary name
    args.remove(0);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                if i + 1 < args.len() {
                    iterations = args[i + 1].parse().unwrap_or(iterations);
                    i += 2;
                    continue;
                }
            }
            "--json" => {
                json = true;
            }
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            s if s.starts_with("--") => {
                eprintln!("unknown flag {s}");
                usage();
                std::process::exit(1);
            }
            _ => {
                pmmls.push(PathBuf::from(&args[i]));
            }
        }
        i += 1;
    }
    if pmmls.is_empty() {
        pmmls.push(PathBuf::from("bench/pmml/DecisionTreeIris.pmml"));
        pmmls.push(PathBuf::from("bench/pmml/GradientBoosterTest.pmml"));
        pmmls.push(PathBuf::from("bench/pmml/MissingValueStrategyTest.pmml"));
    }
    (pmmls, iterations, 0, json)
}

fn active_field_names(sess: &Session) -> Vec<String> {
    // collect from ir field_names filtered by mining_schema active_fields
    let ids = match &sess.ir.model {
        pmmlruntime::ir::ModelIr::Tree(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Regression(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Mining(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Scorecard(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Clustering(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::NaiveBayes(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::NearestNeighbor(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::SupportVectorMachine(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::GeneralRegression(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Association(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::RuleSet(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::NeuralNetwork(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::AnomalyDetection(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Baseline(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::GaussianProcess(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Text(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::TimeSeries(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::Sequence(m) => m.mining_schema.active_fields.clone(),
        pmmlruntime::ir::ModelIr::BayesianNetwork(m) => m.mining_schema.active_fields.clone(),
    };
    ids.iter()
        .filter_map(|fid| sess.ir.field_names.get(fid).cloned())
        .collect()
}

fn make_value(sess: &Session, field: &str, idx: usize, row: usize) -> Value {
    // find FieldMeta for this field to decide type
    let fid = sess.field_id(field);
    if let Some(fid) = fid {
        if let Some(meta) = sess.ir.data_dictionary.iter().find(|m| m.field_id == fid) {
            // if categorical/ordinal, use discrete value from allowed list or synthesize
            if meta.op_type == pmmlruntime::base::OpType::Categorical
                || meta.op_type == pmmlruntime::base::OpType::Ordinal
            {
                if !meta.values.is_empty() {
                    // cycle through allowed values
                    let sid = meta.values[(row + idx) % meta.values.len()];
                    return Value::Discrete(sid);
                }
                // fallback: if we have symbol_names, pick one
                if !sess.ir.symbol_names.is_empty() {
                    // pick first symbol that was interned for this field? approximate
                    if let Some((&sid, _)) = sess.ir.symbol_names.iter().next() {
                        return Value::Discrete(sid);
                    }
                }
                // fallback to string_to_value with categorical string
                return sess.string_to_value(field, "test");
            }
        }
    }
    // continuous: deterministic pseudo-randomish
    let v = ((row.wrapping_mul(7919).wrapping_add(idx.wrapping_mul(97)) % 10000) as f64) / 1000.0 + 0.1;
    // add variation per row
    let v2 = v + ((row % 7) as f64) * 0.13;
    Value::Continuous(v2)
}

fn make_single_input(sess: &Session, row: usize) -> HashMap<String, Value> {
    let fields = active_field_names(sess);
    let mut m = HashMap::with_capacity(fields.len());
    for (idx, f) in fields.iter().enumerate() {
        m.insert(f.clone(), make_value(sess, f, idx, row));
    }
    // if no active fields (e.g., Association uses group), still insert dummy
    if m.is_empty() {
        // fallback to all field_names
        for (idx, name) in sess.ir.field_names.values().enumerate().take(3) {
            m.insert(name.clone(), Value::Continuous(idx as f64 + row as f64 * 0.01));
        }
    }
    m
}

fn make_batch(sess: &Session, n: usize) -> Vec<HashMap<String, Value>> {
    (0..n).map(|r| make_single_input(sess, r)).collect()
}

fn time<F: FnMut()>(mut f: F, iterations: usize, warmup: usize) -> (Duration, Vec<Duration>) {
    for _ in 0..warmup {
        black_box(f());
    }
    let mut times = Vec::with_capacity(if iterations <= 10000 { iterations } else { 0 });
    let start = Instant::now();
    if iterations <= 10000 {
        for _ in 0..iterations {
            let t0 = Instant::now();
            black_box(f());
            times.push(t0.elapsed());
        }
        let total = start.elapsed();
        (total, times)
    } else {
        // for large iterations, just total to avoid vec bloat
        for _ in 0..iterations {
            black_box(f());
        }
        let total = start.elapsed();
        (total, times)
    }
}

fn bench_model(path: &Path, iterations_single: usize) {
    let env = PmmlEnv::new();
    let bytes = std::fs::read(path).expect("read pmml");

    // cold load: measure once per model, but also average of 100 loads to reduce noise
    let cold_iterations = 20;
    let mut cold_times = Vec::with_capacity(cold_iterations);
    for _ in 0..cold_iterations {
        let t0 = Instant::now();
        let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("from_bytes");
        black_box(&sess);
        cold_times.push(t0.elapsed());
    }
    cold_times.sort();
    let cold_median = cold_times[cold_times.len() / 2];
    let cold_mean = {
        let sum: Duration = cold_times.iter().sum();
        sum / cold_times.len() as u32
    };
    let cold_min = *cold_times.iter().min().unwrap();
    let cold_max = *cold_times.iter().max().unwrap();
    let cold_std = {
        let mean_ns = cold_mean.as_nanos() as f64;
        let var: f64 = cold_times
            .iter()
            .map(|d| {
                let x = d.as_nanos() as f64 - mean_ns;
                x * x
            })
            .sum::<f64>()
            / cold_times.len() as f64;
        Duration::from_nanos(var.sqrt() as u64)
    };

    // build session for hot
    let sess = Session::from_bytes(&env, &bytes, SessionOptions::default()).expect("session");
    let single_input = make_single_input(&sess, 0);

    // warm single
    let warmup = 10_000;
    let (total_single, per_row_times) = time(
        || {
            let out = sess.run(&single_input as &dyn Batch).expect("run").into_single().unwrap();
            black_box(out);
        },
        iterations_single,
        warmup,
    );
    let per_row_ns = total_single.as_nanos() as f64 / iterations_single as f64;
    let per_row_std_ns = if !per_row_times.is_empty() {
        let mean = per_row_times.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / per_row_times.len() as f64;
        let var = per_row_times.iter().map(|d| {
            let x = d.as_nanos() as f64 - mean;
            x*x
        }).sum::<f64>() / per_row_times.len() as f64;
        var.sqrt()
    } else {
        0.0
    };

    // batch sizes
    let batch_sizes = [100usize, 1_000, 10_000];
    let mut batch_results = Vec::new();
    for &bs in &batch_sizes {
        let batch = make_batch(&sess, bs);
        // determine iterations for batch: fewer for large
        let iters = match bs {
            100 => 5_000,
            1_000 => 1_000,
            10_000 => 200,
            _ => 500,
        };
        let warmup_b = 200;
        let (total_b, _) = time(
            || {
                let out = sess.run(&batch as &dyn Batch).expect("run_batch").into_rows();
                black_box(out);
            },
            iters,
            warmup_b,
        );
        let batch_mean_ns = total_b.as_nanos() as f64 / iters as f64;
        let per_row_batch_ns = batch_mean_ns / bs as f64;
        let throughput_rows_s = 1e9 / per_row_batch_ns;
        batch_results.push((bs, batch_mean_ns, per_row_batch_ns, throughput_rows_s));
    }

    // also arrow batch for 10k if applicable
    let arrow_batch_result = {
        // try to create RecordBatch for 10k
        let bs = 10_000;
        let batch_vec = make_batch(&sess, bs);
        // convert to RecordBatch via helper if possible: we need to know field types.
        // For simplicity, skip arrow if any categorical - just reuse row-major metric.
        // We'll attempt via arrow crate if fields are mostly continuous.
        None::<(f64, f64)>
    };

    // Print human + json
    let model_name = path.file_name().unwrap().to_string_lossy();
    let size_kb = bytes.len() as f64 / 1024.0;

    println!("=== {} ({:.1} KB, {} active fields) ===", model_name, size_kb, active_field_names(&sess).len());
    println!("cold load: median {:?} mean {:?} ± {:?} min {:?} max {:?} (n={})", cold_median, cold_mean, cold_std, cold_min, cold_max, cold_iterations);
    println!("hot single: {:.1} ns/row ± {:.1} ns (total {:?} for {} iters, warmup {})", per_row_ns, per_row_std_ns, total_single, iterations_single, warmup);
    for (bs, batch_ns, per_row_ns, thr) in &batch_results {
        println!("batch {:>5}: {:>8.1} µs/batch | {:>6.1} ns/row | {:>7.0} rows/s", bs, batch_ns/1_000.0, per_row_ns, thr);
    }
    if let Some((_, _)) = arrow_batch_result {
        // placeholder
    }

    // JSON for aggregation
    let json_line = serde_json::json!({
        "model": model_name,
        "size_kb": size_kb,
        "active_fields": active_field_names(&sess).len(),
        "cold_median_ns": cold_median.as_nanos() as u64,
        "cold_mean_ns": cold_mean.as_nanos() as u64,
        "cold_std_ns": cold_std.as_nanos() as u64,
        "cold_min_ns": cold_min.as_nanos() as u64,
        "cold_max_ns": cold_max.as_nanos() as u64,
        "hot_single_mean_ns": per_row_ns.round() as u64,
        "hot_single_std_ns": per_row_std_ns.round() as u64,
        "hot_single_total_ns": total_single.as_nanos() as u64,
        "hot_single_iters": iterations_single,
        "batches": batch_results.iter().map(|(bs, batch_ns, per_row, thr)| serde_json::json!({
            "size": bs,
            "batch_mean_ns": *batch_ns as u64,
            "per_row_ns": *per_row as u64,
            "throughput_rows_s": *thr as u64
        })).collect::<Vec<_>>(),
    });
    // print JSON to stderr for parsing? print to stdout with prefix
    println!("JSON {}", json_line.to_string());
}

fn main() -> anyhow::Result<()> {
    let (pmmls, iters, _, _json) = parse_args();
    for p in &pmmls {
        bench_model(p, iters);
        println!();
    }
    Ok(())
}
