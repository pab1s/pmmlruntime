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
        Commands::Run { model, input, output } => run(&model, input.as_deref(), output.as_deref()),
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
        println!("TreeModel function={} missingStrategy={:?} noTrueChild={:?}", tm.function_name, tm.missing_value_strategy, tm.no_true_child_strategy);
        println!("  MiningSchema: {} fields", tm.mining_schema.len());
        for mf in &tm.mining_schema {
            println!("    - {} usage={:?} importance={:?}", mf.name, mf.usage_type, mf.importance);
        }
        println!("  Output: {} fields", tm.output.len());
        for of in &tm.output {
            println!("    - {} feature={:?} value={:?}", of.name, of.feature, of.value);
        }
        println!("  Root node id={:?} score={:?} children={}", tm.root.id, tm.root.score, tm.root.children.len());
        // lower to IR for field counts
        let ir = pmml_ir::lower(raw).map_err(|e| anyhow::anyhow!("lower: {e}"))?;
        println!("IR: data_dictionary={}, derived={}, tree nodes={}", ir.data_dictionary.len(), ir.derived_fields.len(), match &ir.model { pmml_ir::ir::ModelIr::Tree(t) => t.nodes.len(), _ => 0 });
    } else {
        println!("No TreeModel (v1 only Tree supported)");
    }
    Ok(())
}

fn run(model: &str, input: Option<&str>, output: Option<&str>) -> anyhow::Result<()> {
    let env = pmml_session::PmmlEnv::new();
    let opts = pmml_session::SessionOptions::default();
    let sess = pmml_session::Session::from_file(&env, model, opts).map_err(|e| anyhow::anyhow!("session: {e}"))?;
    println!("Loaded model: {} active fields={}", model, sess.num_active_fields());

    if let Some(inp) = input {
        // CSV batch: read header, score each row, write output
        let out_path = output.unwrap_or("output.csv");
        let csv_text = std::fs::read_to_string(inp)?;
        let mut lines = csv_text.lines();
        let header = lines.next().ok_or_else(|| anyhow::anyhow!("empty csv"))?;
        let cols: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
        let mut out_lines = vec![format!("{},predictedValue", header)];
        for line in lines {
            if line.trim().is_empty() { continue; }
            let vals: Vec<&str> = line.split(',').collect();
            let mut map: HashMap<String, pmml_core::Value> = HashMap::new();
            for (col, val) in cols.iter().zip(vals.iter()) {
                let v = if let Ok(f) = val.parse::<f64>() {
                    pmml_core::Value::Continuous(f)
                } else if val.is_empty() {
                    pmml_core::Value::Missing
                } else {
                    // For discrete string, we hash to SymbolId — but for Tree continuous inputs, not needed.
                    // Use Continuous fallback.
                    pmml_core::Value::Continuous(val.parse().unwrap_or(0.0))
                };
                map.insert(col.clone(), v);
            }
            let out = sess.run(map).map_err(|e| anyhow::anyhow!("run: {e}"))?;
            let pred = out.get("predictedValue").or_else(|| out.values().next()).unwrap_or(&pmml_core::Value::Missing);
            let pred_str = match pred {
                pmml_core::Value::Continuous(f) => f.to_string(),
                pmml_core::Value::Discrete(sid) => sess.ir.symbol_names.get(sid).cloned().unwrap_or_else(|| format!("{sid:?}")),
                pmml_core::Value::Missing => "Missing".into(),
            };
            out_lines.push(format!("{line},{pred_str}"));
        }
        std::fs::write(out_path, out_lines.join("\n"))?;
        println!("Scored {} rows -> {out_path}", out_lines.len()-1);
    } else {
        // Single example: Petal.Length=1.4, Petal.Width=0.2 => setosa
        let mut example = HashMap::new();
        example.insert("Petal.Length".to_string(), pmml_core::Value::Continuous(1.4));
        example.insert("Petal.Width".to_string(), pmml_core::Value::Continuous(0.2));
        let out = sess.run(example).map_err(|e| anyhow::anyhow!("run: {e}"))?;
        // pretty print with resolved symbols
        for (k, v) in &out {
            let s = match v {
                pmml_core::Value::Discrete(sid) => sess.ir.symbol_names.get(sid).cloned().unwrap_or_else(|| format!("{sid:?}")),
                pmml_core::Value::Continuous(f) => f.to_string(),
                pmml_core::Value::Missing => "Missing".into(),
            };
            println!("  {k} = {s} ({v:?})");
        }
    }
    Ok(())
}

fn verify(model: &str) -> anyhow::Result<()> {
    let bytes = std::fs::read(model)?;
    let raw = pmml_xml::unmarshal(&bytes).map_err(|e| anyhow::anyhow!("unmarshal: {e}"))?;
    pmml_ir::verify_raw(&raw).map_err(|e| anyhow::anyhow!("verify_raw: {e}"))?;
    let ir = pmml_ir::lower(raw).map_err(|e| anyhow::anyhow!("lower: {e}"))?;
    pmml_ir::verify_ir(&ir).map_err(|e| anyhow::anyhow!("verify_ir: {e}"))?;
    println!("verify: {model} OK — Tree nodes={}", match &ir.model { pmml_ir::ir::ModelIr::Tree(t) => t.nodes.len(), _ => 0 });
    Ok(())
}
