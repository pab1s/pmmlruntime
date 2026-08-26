//! Score any PMML file (LightGBM, XGBoost, sklearn, R) — same API.
//!
//! After conversion there is no "LightGBM PMML" or "XGBoost PMML" —
//! `lightgbm2pmml`/`sklearn2pmml`/`r2pmml` all emit one PMML 4.4 `MiningModel`
//! (typically `Segmentation` with `multipleModelMethod="sum"` over `TreeModel`s).
//! The scoring engine never knows the original framework.
//!
//! This example mirrors JPMML-Evaluator's
//! [basic usage](https://github.com/jpmml/jpmml-evaluator#basic-usage)
//! and [advanced usage](https://github.com/jpmml/jpmml-evaluator#advanced-usage)
//! but in Rust (no JVM).
//!
//! Run:
//! ```sh
//! cargo run -p pmmlruntime --example score_file -- bench/pmml/GradientBoosterTest.pmml
//! cargo run -p pmmlruntime --example score_file -- bench/pmml/GradientBoosterTest.pmml input.csv --output out.csv
//! # input.csv is any CSV with a header row matching MiningSchema active fields, e.g.:
//! # x
//! # 0.5
//! # 1.0
//!```
//!
//! The `GradientBoosterTest.pmml` fixture is used here because it *is* a GBDT
//! ensemble (3 `RegressionModel` stumps summed → `modelChain` to probabilities)
//! — structurally identical to a small LightGBM PMML. Replace the path with
//! your own `lightgbm.pmml` / `xgboost.pmml`; the code does not change.

use std::collections::HashMap;
use std::path::PathBuf;

use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use pmmlruntime::Value;

fn print_output(out: &HashMap<String, Value>, ir: &pmmlruntime::ir::Ir) {
    // predictedValue is always present; other keys depend on Output/Targets
    for (k, v) in out {
        let display = match v {
            Value::Continuous(f) => format!("{f}"),
            Value::Discrete(sid) => ir
                .symbol_names
                .get(sid)
                .cloned()
                .unwrap_or_else(|| format!("Symbol({})", sid.0)),
            Value::Missing => "Missing".into(),
        };
        println!("  {k}: {display}");
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().collect::<Vec<_>>();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model.pmml> [input.csv] [--output out.csv]",
            args[0]
        );
        eprintln!("  model.pmml can be from LightGBM (lightgbm2pmml), XGBoost, sklearn2pmml, r2pmml — same code.");
        eprintln!("  Example: {} bench/pmml/GradientBoosterTest.pmml", args[0]);
        std::process::exit(1);
    }
    let model_path = PathBuf::from(&args[1]);
    let csv_path = args
        .get(2)
        .filter(|p| !p.starts_with("--"))
        .map(PathBuf::from);
    let out_path = {
        let idx = args.iter().position(|a| a == "--output");
        idx.and_then(|i| args.get(i + 1).map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("out.csv"))
    };

    // 1. Load PMML — same for _every_ framework. No LightGBM flag.
    let env = PmmlEnv::new();
    // from_file is shorthand for std::fs::read + from_bytes (cold: 68 µs for Iris 2.9 KB, quick-xml 0.37, 100 MB cap, XXE blocked)
    let sess = Session::from_file(
        &env,
        &model_path.to_string_lossy(),
        SessionOptions::default(),
    )?;
    println!(
        "Loaded {}: {} active field(s)",
        model_path.display(),
        sess.num_active_fields()
    );
    println!(
        "  DataDictionary: {} field(s)",
        sess.ir.data_dictionary.len()
    );
    for (fid, name) in &sess.ir.field_names {
        println!("    FieldId({}) = {name}", fid.0);
    }
    // Show what the model *is* in PMML terms (TreeModel vs MiningModel GBDT etc.)
    match &sess.ir.model {
        pmmlruntime::ir::ModelIr::Tree(_) => println!("  Model: TreeModel"),
        pmmlruntime::ir::ModelIr::Regression(_) => println!("  Model: RegressionModel"),
        pmmlruntime::ir::ModelIr::Mining(m) => {
            println!(
                "  Model: MiningModel (ensemble, {} segment(s), method {:?})",
                m.segmentation.segments.len(),
                m.segmentation.multiple_model_method
            );
            for seg in &m.segmentation.segments {
                let kind = match &*seg.model {
                    pmmlruntime::ir::ModelIr::Tree(_) => "TreeModel",
                    pmmlruntime::ir::ModelIr::Regression(_) => "RegressionModel",
                    pmmlruntime::ir::ModelIr::GeneralRegression(_) => "GeneralRegressionModel",
                    pmmlruntime::ir::ModelIr::Mining(_) => "MiningModel",
                    _ => "Other",
                };
                println!(
                    "    segment {} weight {}: {}",
                    seg.id.as_deref().unwrap_or("-"),
                    seg.weight,
                    kind
                );
            }
        }
        pmmlruntime::ir::ModelIr::GeneralRegression(_) => {
            println!("  Model: GeneralRegressionModel")
        }
        pmmlruntime::ir::ModelIr::Scorecard(_) => println!("  Model: Scorecard"),
        pmmlruntime::ir::ModelIr::Clustering(_) => println!("  Model: ClusteringModel"),
        pmmlruntime::ir::ModelIr::NaiveBayes(_) => println!("  Model: NaiveBayesModel"),
        pmmlruntime::ir::ModelIr::NearestNeighbor(_) => println!("  Model: NearestNeighborModel"),
        pmmlruntime::ir::ModelIr::SupportVectorMachine(_) => {
            println!("  Model: SupportVectorMachineModel")
        }
        pmmlruntime::ir::ModelIr::NeuralNetwork(_) => println!("  Model: NeuralNetwork"),
        pmmlruntime::ir::ModelIr::Association(_) => println!("  Model: AssociationModel"),
        pmmlruntime::ir::ModelIr::RuleSet(_) => println!("  Model: RuleSetModel"),
        pmmlruntime::ir::ModelIr::AnomalyDetection(_) => println!("  Model: AnomalyDetectionModel"),
        pmmlruntime::ir::ModelIr::Baseline(_) => println!("  Model: BaselineModel"),
        pmmlruntime::ir::ModelIr::GaussianProcess(_) => println!("  Model: GaussianProcessModel"),
        pmmlruntime::ir::ModelIr::Text(_) => println!("  Model: TextModel"),
        pmmlruntime::ir::ModelIr::TimeSeries(_) => println!("  Model: TimeSeriesModel"),
        pmmlruntime::ir::ModelIr::Sequence(_) => println!("  Model: SequenceModel"),
        pmmlruntime::ir::ModelIr::BayesianNetwork(_) => println!("  Model: BayesianNetworkModel"),
    }

    // 2a. Batch path — input data file (CSV). This is the "advanced" JPMML equivalent:
    //     evaluator.evaluate(batch) → PMML's `ModelVerification` style.
    if let Some(csv) = csv_path {
        let csv_str = std::fs::read_to_string(&csv)?;
        let batch = pmmlruntime::session::arrow::csv_str_to_record_batch(&csv_str, None, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "\nScoring batch {}: {} rows",
            csv.display(),
            batch.num_rows()
        );
        let outs = sess.run_batch_arrow(&batch)?;
        // outs is Vec<HashMap<String,Value>> — one map per row, same keys as single run
        for (i, out) in outs.iter().take(5).enumerate() {
            println!("row {i}:");
            print_output(out, &sess.ir);
        }
        if outs.len() > 5 {
            println!("... ({} more rows)", outs.len() - 5);
        }
        // Optionally write CSV (predictedValue + any OutputField)
        let header = {
            let mut keys: Vec<String> = outs
                .first()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            keys.sort();
            keys
        };
        // Write CSV manually (no csv crate needed)
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&out_path)?;
            writeln!(f, "{}", header.join(","))?;
            for out in &outs {
                let row: Vec<String> = header
                    .iter()
                    .map(|k| match out.get(k) {
                        Some(Value::Continuous(f)) => f.to_string(),
                        Some(Value::Discrete(sid)) => sess
                            .ir
                            .symbol_names
                            .get(sid)
                            .cloned()
                            .unwrap_or_else(|| format!("Symbol({})", sid.0)),
                        _ => String::new(),
                    })
                    .collect();
                writeln!(f, "{}", row.join(","))?;
            }
        }
        println!("\nWrote {} → {}", csv.display(), out_path.display());
        return Ok(());
    }

    // 2b. Single-row path — JPMML "basic usage" `evaluator.evaluate(arguments)` equivalent.
    //     For a LightGBM PMML, active fields are whatever you trained on, e.g. age,income,dept.
    //     Unknown fields are ignored; missing fields become Value::Missing (MiningSchema handles
    //     outlier/missing/invalid per PMML spec).
    println!("\nSingle-row example (replace field names/values with your model's active fields):");
    let mut input = HashMap::new();
    // Active fields = MiningSchema active inputs. GradientBoosterTest has one: "x".
    // A real LightGBM PMML will have e.g. "age"→Continuous(34.0), "dept"→Discrete(sid_for("sales")).
    let active_names: Vec<String> = match &sess.ir.model {
        pmmlruntime::ir::ModelIr::Tree(m) => m
            .mining_schema
            .active_fields
            .iter()
            .filter_map(|fid| sess.ir.field_names.get(fid).cloned())
            .collect(),
        pmmlruntime::ir::ModelIr::Regression(m) => m
            .mining_schema
            .active_fields
            .iter()
            .filter_map(|fid| sess.ir.field_names.get(fid).cloned())
            .collect(),
        pmmlruntime::ir::ModelIr::Mining(m) => m
            .mining_schema
            .active_fields
            .iter()
            .filter_map(|fid| sess.ir.field_names.get(fid).cloned())
            .collect(),
        _ => sess.ir.field_names.values().cloned().collect(),
    };
    if let Some(first_name) = active_names.first() {
        // For demo we just put 1.0; replace with your row
        input.insert(first_name.clone(), Value::Continuous(1.0));
        println!(
            "  input: {:?} (field {})",
            input.values().next().unwrap(),
            first_name
        );
        if active_names.len() > 1 {
            println!(
                "  (model has {} active fields: {:?})",
                active_names.len(),
                active_names
            );
        }
    } else {
        input.insert("x".to_string(), Value::Continuous(1.0));
    }

    // Helper for categorical: get SymbolId via sess.symbol_id("sales") or sess.string_to_value("dept","sales")
    // let sid = sess.symbol_id("sales").unwrap(); input.insert("dept".into(), Value::Discrete(sid));

    let out = sess.run(input)?;
    println!("output:");
    print_output(&out, &sess.ir);

    // 2c. Advanced: pre-resolved FieldId (avoid per-row HashMap<String,Value> hashing, ~402 ns vs ~1 µs)
    //     Mirrors JPMML `InputField`/`FieldValue` preparation.
    println!("\nAdvanced — FieldId batch (JPMML InputField/FieldValue equivalent, 402 ns single):");
    if let Some(fid) = sess.field_id("x") {
        let out2 = sess.run_with_ids(&[(fid, Value::Continuous(2.0))])?;
        print_output(&out2, &sess.ir);
    }

    println!("\nTip: LightGBM PMML is just a MiningModel. Score it with:");
    println!(
        "  cargo run -p pmml-cli -- run --model lightgbm.pmml --batch input.csv --output out.csv"
    );
    println!("See README.md \"Use it\" and docs/ARCHITECTURE.md for Batch/Arrow details.");

    // Touch unused args to avoid warning when --output not used in single-row mode
    let _ = &mut args;
    let _ = out_path;
    Ok(())
}
