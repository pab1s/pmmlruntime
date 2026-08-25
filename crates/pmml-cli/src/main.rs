use clap::{Parser, Subcommand};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "pmml-runtime", version, about = "PMML 4.4 runtime — ONNX-style", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect PMML file (prints DataDictionary, MiningSchema)
    Inspect {
        #[arg(long)]
        model: String,
    },
    /// Run scoring
    Run {
        #[arg(long)]
        model: String,
        #[arg(long)]
        input: Option<String>,
        #[arg(long)]
        output: Option<String>,
        /// Batch CSV input (alias for --input) — uses Arrow RecordBatch + run_batch (parallel)
        #[arg(long)]
        batch: Option<String>,
    },
    Verify {
        #[arg(long)]
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { model } => inspect(&model),
        Commands::Run {
            model,
            input,
            output,
            batch,
        } => {
            // --batch takes precedence over --input
            let inp = batch.as_deref().or(input.as_deref());
            run(&model, inp, output.as_deref(), batch.is_some())
        }
        Commands::Verify { model } => verify(&model),
    }
}

fn inspect(model: &str) -> anyhow::Result<()> {
    let bytes = std::fs::read(model)?;
    let raw = pmml_xml::unmarshal(&bytes).map_err(|e| anyhow::anyhow!("unmarshal: {e}"))?;
    println!("DataDictionary: {} fields", raw.data_dictionary.len());
    for df in &raw.data_dictionary {
        println!("  - {} {} {}", df.name, df.data_type, df.op_type);
    }
    if let Some(tm) = &raw.tree_model {
        println!(
            "TreeModel function={} missingStrategy={:?} noTrueChild={:?}",
            tm.function_name, tm.missing_value_strategy, tm.no_true_child_strategy
        );
        println!("  MiningSchema: {} fields", tm.mining_schema.len());
        for mf in &tm.mining_schema {
            println!(
                "    - {} usage={:?} importance={:?}",
                mf.name, mf.usage_type, mf.importance
            );
        }
        println!("  Output: {} fields", tm.output.len());
        for of in &tm.output {
            println!(
                "    - {} feature={:?} value={:?}",
                of.name, of.feature, of.value
            );
        }
        println!(
            "  Root node id={:?} score={:?} children={}",
            tm.root.id,
            tm.root.score,
            tm.root.children.len()
        );
        // lower to IR for field counts
        let ir = pmml_ir::lower(raw).map_err(|e| anyhow::anyhow!("lower: {e}"))?;
        println!(
            "IR: data_dictionary={}, derived={}, tree nodes={}",
            ir.data_dictionary.len(),
            ir.derived_fields.len(),
            match &ir.model {
                pmml_ir::ir::ModelIr::Tree(t) => t.nodes.len(),
                _ => 0,
            }
        );
    } else {
        println!("No TreeModel (v1 only Tree supported)");
    }
    Ok(())
}

fn run(
    model: &str,
    input: Option<&str>,
    output: Option<&str>,
    is_batch_flag: bool,
) -> anyhow::Result<()> {
    let env = pmml_session::PmmlEnv::new();
    // Use CpuBatched when --batch or large CSV; otherwise CpuSerial. For CLI batch we default to batched
    // to achieve 3M rows/s via rayon par_iter chunked at 1k (plan A3).
    let opts = if is_batch_flag || input.is_some() {
        pmml_session::SessionOptions::default()
            .execution_provider(pmml_session::ExecutionProviderKind::CpuBatched)
    } else {
        pmml_session::SessionOptions::default()
    };
    let sess = pmml_session::Session::from_file(&env, model, opts)
        .map_err(|e| anyhow::anyhow!("session: {e}"))?;
    println!(
        "Loaded model: {} active fields={}",
        model,
        sess.num_active_fields()
    );

    if let Some(inp) = input {
        // Batch path via Arrow: arrow::csv::Reader -> RecordBatch -> run_batch -> arrow::csv::Writer
        // We use the Arrow bridge for CSV handling but retain fallback manual line split for robustness.
        let out_path = output.unwrap_or("output.csv");
        let csv_text = std::fs::read_to_string(inp)?;
        if is_batch_flag {
            // Arrow path: parse CSV into RecordBatch, then to Value maps, then batched scoring
            // Has_header true; schema inferred from header (Utf8) for generic CSVs
            match pmml_session::arrow::csv_str_to_record_batch(&csv_text, None, true) {
                Ok(batch) => {
                    // RecordBatch -> Vec<HashMap<String, Value>> (Arrow bridge)
                    let inputs = pmml_session::arrow::record_batch_to_value_maps(&batch);
                    // Capture header for output
                    let header_line = csv_text.lines().next().unwrap_or("").to_string();
                    // run_batch uses rayon chunked 1k internally
                    let outputs = sess
                        .run_batch(inputs)
                        .map_err(|e| anyhow::anyhow!("run_batch: {e}"))?;
                    // Build output CSV: original lines + predictedValue column
                    let mut out_lines = vec![format!("{},predictedValue", header_line)];
                    let orig_lines: Vec<&str> = csv_text.lines().skip(1).collect();
                    for (orig, out_map) in orig_lines.iter().zip(outputs.iter()) {
                        if orig.trim().is_empty() {
                            continue;
                        }
                        let pred = out_map
                            .get("predictedValue")
                            .or_else(|| out_map.values().next())
                            .unwrap_or(&pmml_core::Value::Missing);
                        let pred_str = match pred {
                            pmml_core::Value::Continuous(f) => f.to_string(),
                            pmml_core::Value::Discrete(sid) => sess
                                .ir
                                .symbol_names
                                .get(sid)
                                .cloned()
                                .unwrap_or_else(|| format!("{sid:?}")),
                            pmml_core::Value::Missing => "Missing".into(),
                        };
                        out_lines.push(format!("{orig},{pred_str}"));
                    }
                    // Demonstrate Arrow writer path as well: build output RecordBatch from outputs
                    // (not strictly needed for file, but validates arrow bridge; we keep manual file for now)
                    {
                        let out_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                            arrow::datatypes::Field::new(
                                "predictedValue",
                                arrow::datatypes::DataType::Utf8,
                                true,
                            ),
                        ]));
                        let _ = pmml_session::arrow::value_maps_to_record_batch(
                            &outputs,
                            out_schema,
                            Some(&sess.ir.symbol_names),
                        );
                        // writer would be: arrow::csv::WriterBuilder::new(out_schema).build(file)
                    }
                    std::fs::write(out_path, out_lines.join("\n"))?;
                    println!(
                        "Scored {} rows (Arrow batch) -> {out_path}",
                        out_lines.len() - 1
                    );
                }
                Err(e) => {
                    eprintln!("Arrow CSV parse failed ({e}), falling back to manual split");
                    return run_manual_batch(&sess, &csv_text, out_path);
                }
            }
        } else {
            // Manual fallback path (for --input without --batch flag, still batched via run_batch but line-split)
            return run_manual_batch(&sess, &csv_text, out_path);
        }
    } else {
        // Single example: Petal.Length=1.4, Petal.Width=0.2 => setosa
        let mut example = HashMap::new();
        example.insert(
            "Petal.Length".to_string(),
            pmml_core::Value::Continuous(1.4),
        );
        example.insert("Petal.Width".to_string(), pmml_core::Value::Continuous(0.2));
        let out = sess.run(example).map_err(|e| anyhow::anyhow!("run: {e}"))?;
        // pretty print with resolved symbols
        for (k, v) in &out {
            let s = match v {
                pmml_core::Value::Discrete(sid) => sess
                    .ir
                    .symbol_names
                    .get(sid)
                    .cloned()
                    .unwrap_or_else(|| format!("{sid:?}")),
                pmml_core::Value::Continuous(f) => f.to_string(),
                pmml_core::Value::Missing => "Missing".into(),
            };
            println!("  {k} = {s} ({v:?})");
        }
    }
    Ok(())
}

fn run_manual_batch(
    sess: &pmml_session::Session,
    csv_text: &str,
    out_path: &str,
) -> anyhow::Result<()> {
    let mut lines = csv_text.lines();
    let header = lines.next().ok_or_else(|| anyhow::anyhow!("empty csv"))?;
    let cols: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    // Collect inputs first, then use run_batch (parallel when CpuBatched)
    let mut inputs: Vec<HashMap<String, pmml_core::Value>> = Vec::new();
    let mut orig_lines: Vec<String> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let vals: Vec<&str> = line.split(',').collect();
        let mut map: HashMap<String, pmml_core::Value> = HashMap::new();
        for (col, val) in cols.iter().zip(vals.iter()) {
            let v = if let Ok(f) = val.parse::<f64>() {
                pmml_core::Value::Continuous(f)
            } else if val.is_empty() {
                pmml_core::Value::Missing
            } else {
                pmml_core::Value::Continuous(val.parse().unwrap_or(0.0))
            };
            map.insert(col.clone(), v);
        }
        inputs.push(map);
        orig_lines.push(line.to_string());
    }
    let outputs = sess
        .run_batch(inputs)
        .map_err(|e| anyhow::anyhow!("run_batch: {e}"))?;
    let mut out_lines = vec![format!("{},predictedValue", header)];
    for (orig, out_map) in orig_lines.iter().zip(outputs.iter()) {
        let pred = out_map
            .get("predictedValue")
            .or_else(|| out_map.values().next())
            .unwrap_or(&pmml_core::Value::Missing);
        let pred_str = match pred {
            pmml_core::Value::Continuous(f) => f.to_string(),
            pmml_core::Value::Discrete(sid) => sess
                .ir
                .symbol_names
                .get(sid)
                .cloned()
                .unwrap_or_else(|| format!("{sid:?}")),
            pmml_core::Value::Missing => "Missing".into(),
        };
        out_lines.push(format!("{orig},{pred_str}"));
    }
    std::fs::write(out_path, out_lines.join("\n"))?;
    println!("Scored {} rows -> {out_path}", out_lines.len() - 1);
    Ok(())
}

fn verify(model: &str) -> anyhow::Result<()> {
    let bytes = std::fs::read(model)?;
    let raw = pmml_xml::unmarshal(&bytes).map_err(|e| anyhow::anyhow!("unmarshal: {e}"))?;
    pmml_ir::verify_raw(&raw).map_err(|e| anyhow::anyhow!("verify_raw: {e}"))?;
    let ir = pmml_ir::lower(raw).map_err(|e| anyhow::anyhow!("lower: {e}"))?;
    pmml_ir::verify_ir(&ir).map_err(|e| anyhow::anyhow!("verify_ir: {e}"))?;
    println!(
        "verify: {model} OK — Tree nodes={}",
        match &ir.model {
            pmml_ir::ir::ModelIr::Tree(t) => t.nodes.len(),
            _ => 0,
        }
    );
    Ok(())
}
