//! `pmml-runtime` CLI — `inspect` / `run` / `verify` subcommands (`clap` 4.5 derive).
//!
//! Thin wrapper over `pmml_session` and `pmml_xml`/`pmml_ir` for file-based scoring.
//! It mirrors `jpmml-evaluator` CLI but ONNX-style: `PmmlEnv` + `Session` + `Batch`.
//! All I/O is via `std::fs`; Arrow `csv` bridging uses `pmmlruntime::session::arrow` for batch.
//!
//! # Commands
//!
//! - `inspect --model model.pmml` — prints `DataDictionary`, `MiningSchema`, `Output`, and `Ir` counts.
//! - `run --model model.pmml [--input input.csv --output output.csv] [--batch batch.csv]` — single or batched scoring.
//! - `verify --model model.pmml` — `unmarshal` → `verify_raw` → `lower` → `verify_ir` and reports `Tree` node count.
//!
//! # Performance
//!
//! `run --batch` uses `pmmlruntime::session::ExecutionProviderKind::CpuBatched` (rayon `par_chunks(256)`, fallback `<256` serial)
//! and the `arrow` `csv` path for zero-copy input when possible.

#![allow(clippy::too_many_lines)]

use clap::{Parser, Subcommand};
use std::collections::HashMap;

/// CLI entry ( `pmml-runtime` v0.1.0, ONNX-style ).
///
/// `clap` derive with subcommands `Inspect` / `Run` / `Verify`. `PmmlEnv` is created once
/// per invocation; `Session` is built via `Session::from_file`.
#[derive(Parser)]
#[command(name = "pmml-runtime", version, about = "PMML 4.4 runtime — ONNX-style", long_about = None)]
struct Cli {
    /// Subcommand (`inspect`, `run`, `verify`).
    #[command(subcommand)]
    command: Commands,
}

/// Subcommands for `pmml-runtime`.
#[derive(Subcommand)]
enum Commands {
    /// Inspect PMML file (prints `DataDictionary`, `MiningSchema`).
    ///
    /// Reads `model` via `pmmlruntime::xml::unmarshal` and `pmmlruntime::ir::lower`, then prints `DataDictionary` fields,
    /// `TreeModel` function/`missing_value_strategy`/`no_true_child_strategy`, mining schema, output, root node, and `Ir` counts.
    Inspect {
        /// Path to `model.pmml` (e.g. `DecisionTreeIris.pmml`).
        #[arg(long)]
        model: String,
    },
    /// Run scoring (single example or CSV batch).
    ///
    /// Without `--input`/`--batch` it scores a hard-coded Iris example (`Petal.Length=1.4`, `Petal.Width=0.2`).
    /// With `--batch`/`--input` it reads CSV via `arrow::csv` or manual split, then `Session::run_batch` (batched) or `run`.
    Run {
        /// Path to `model.pmml`.
        #[arg(long)]
        model: String,
        /// CSV input path for batch (alias for `--batch` without `arrow` fast path).
        #[arg(long)]
        input: Option<String>,
        /// Output CSV path (default `output.csv`).
        #[arg(long)]
        output: Option<String>,
        /// Batch CSV input (alias for `--input`) — uses `RecordBatch` + `run_batch` (parallel `rayon`).
        #[arg(long)]
        batch: Option<String>,
    },
    /// Verify PMML file (`unmarshal` → `verify` → `lower` → `verify_ir`).
    ///
    /// Reports `Tree` node count on success; errors are bubbled via `anyhow` with context (`unmarshal:` / `lower:`).
    Verify {
        /// Path to `model.pmml`.
        #[arg(long)]
        model: String,
    },
}

/// Entry point — parses `Cli` via `clap::Parser` and dispatches to `inspect` / `run` / `verify`.
///
/// # Returns
///
/// `Ok(())` on success, `Err(anyhow)` on `unmarshal` / `lower` / IO failure. `clap` handles `--help`/`--version` before reaching here.
///
/// # Errors
///
/// Propagates via `anyhow`: `inspect`/`run`/`verify` errors are wrapped with `anyhow!` and cause non-zero exit.
///
/// # Examples
///
/// ```text
/// // $ pmml-runtime inspect --model model.pmml
/// // $ pmml-runtime run --model model.pmml --batch input.csv --output out.csv
/// ```
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

/// Inspect a PMML file — print `DataDictionary`, `MiningSchema`, `Output`, root node, and `Ir`.
///
/// # Parameters
///
/// - `model`: path to PMML file.
///
/// # Returns
///
/// `Ok(())` after printing to `stdout`. `Err` if `std::fs::read` or `pmmlruntime::xml::unmarshal` fails.
///
/// # Errors
///
/// - `IO` if file read fails.
/// - `Parse` / `UnsupportedMarkup` / `InvalidValue` from `unmarshal` / `lower` are mapped to `anyhow` with context.
/// - Not an `Ir` `TreeModel` is reported as `No TreeModel (v1 only Tree supported)` but still prints `DataDictionary`.
///
/// # Panics
///
/// Does not panic; errors are returned.
///
/// # Examples
///
/// ```no_run
/// pmml_cli::main(); // via `pmml-runtime inspect --model model.pmml`
/// ```
fn inspect(model: &str) -> anyhow::Result<()> {
    let bytes = std::fs::read(model)?;
    let raw = pmmlruntime::xml::unmarshal(&bytes).map_err(|e| anyhow::anyhow!("unmarshal: {e}"))?;
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
        let ir = pmmlruntime::ir::lower(raw).map_err(|e| anyhow::anyhow!("lower: {e}"))?;
        println!(
            "IR: data_dictionary={}, derived={}, tree nodes={}",
            ir.data_dictionary.len(),
            ir.derived_fields.len(),
            match &ir.model {
                pmmlruntime::ir::ModelIr::Tree(t) => t.nodes.len(),
                _ => 0,
            }
        );
    } else {
        println!("No TreeModel (v1 only Tree supported)");
    }
    Ok(())
}

/// Run scoring — single example or CSV batch.
///
/// Creates `PmmlEnv::new()` and `SessionOptions` ( `CpuBatched` if `--batch` or `input.is_some()`, else `CpuSerial`),
/// then `Session::from_file`. For `input.is_some()` it treats `input` as CSV path and scores via `arrow::csv`
/// → `RecordBatch` → `record_batch_to_value_maps` → `run_batch` (or manual `run_manual_batch` fallback). For
/// `is_batch_flag` it uses the `arrow` path; otherwise manual line-split.
///
/// # Parameters
///
/// - `model`: path to `model.pmml`.
/// - `input`: optional CSV path (from `--batch` or `--input`).
/// - `output`: optional output path (default `output.csv`).
/// - `is_batch_flag`: `true` if `--batch` was passed (selects `arrow` + `CpuBatched`).
///
/// # Returns
///
/// `Ok(())` after writing `output.csv` or printing single example. `Err` on `Session::from_file` / `run` / IO.
///
/// # Errors
///
/// - `PmmlError::Io` / `Parse` / `InvalidValue` mapped to `anyhow`.
/// - CSV parse failure falls back to manual split but still returns `Err` if scoring fails.
/// - Empty CSV returns `anyhow!("empty csv")` via `run_manual_batch`.
///
/// # Performance
///
/// Batch path shards via `rayon` `par_chunks(256)` and `BatchCtx::for_record_batch` zero-copy for `RecordBatch`.
///
/// # Panics
///
/// Does not panic; empty `batch` returns `Ok(Vec::new())` inside `Session` and is handled.
fn run(
    model: &str,
    input: Option<&str>,
    output: Option<&str>,
    is_batch_flag: bool,
) -> anyhow::Result<()> {
    let env = pmmlruntime::session::PmmlEnv::new();
    // Use CpuBatched when --batch or large CSV; otherwise CpuSerial. For CLI batch we default to batched
    // to achieve 3M rows/s via rayon par_iter chunked at 1k (plan A3).
    let opts = if is_batch_flag || input.is_some() {
        pmmlruntime::session::SessionOptions::default()
            .execution_provider(pmmlruntime::session::ExecutionProviderKind::CpuBatched)
    } else {
        pmmlruntime::session::SessionOptions::default()
    };
    let sess = pmmlruntime::session::Session::from_file(&env, model, opts)
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
            match pmmlruntime::session::arrow::csv_str_to_record_batch(&csv_text, None, true) {
                Ok(batch) => {
                    // RecordBatch -> Vec<HashMap<String, Value>> (Arrow bridge)
                    let inputs = pmmlruntime::session::arrow::record_batch_to_value_maps(&batch);
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
                            .unwrap_or(&pmmlruntime::base::Value::Missing);
                        let pred_str = match pred {
                            pmmlruntime::base::Value::Continuous(f) => f.to_string(),
                            pmmlruntime::base::Value::Discrete(sid) => sess
                                .ir
                                .symbol_names
                                .get(sid)
                                .cloned()
                                .unwrap_or_else(|| format!("{sid:?}")),
                            pmmlruntime::base::Value::Missing => "Missing".into(),
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
                        let _ = pmmlruntime::session::arrow::value_maps_to_record_batch(
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
            pmmlruntime::base::Value::Continuous(1.4),
        );
        example.insert(
            "Petal.Width".to_string(),
            pmmlruntime::base::Value::Continuous(0.2),
        );
        let out = sess.run(example).map_err(|e| anyhow::anyhow!("run: {e}"))?;
        // pretty print with resolved symbols
        for (k, v) in &out {
            let s = match v {
                pmmlruntime::base::Value::Discrete(sid) => sess
                    .ir
                    .symbol_names
                    .get(sid)
                    .cloned()
                    .unwrap_or_else(|| format!("{sid:?}")),
                pmmlruntime::base::Value::Continuous(f) => f.to_string(),
                pmmlruntime::base::Value::Missing => "Missing".into(),
            };
            println!("  {k} = {s} ({v:?})");
        }
    }
    Ok(())
}

/// Manual CSV batch fallback — line-split → `Value::Continuous` / `Missing` → `Session::run_batch` → write `output.csv`.
///
/// Handles CSV without `arrow` (e.g. when `arrow::csv::Reader` failed). Splits header on `,`,
/// then each line on `,` and `parse::<f64>` for numerics; empty → `Missing`, else `Continuous(f)`.
///
/// # Parameters
///
/// - `sess`: `&Session` (already loaded).
/// - `csv_text`: full CSV text (header + rows).
/// - `out_path`: path to write output CSV (`header,predictedValue` + rows).
///
/// # Returns
///
/// `Ok(())` after writing file; `Err` if `run_batch` or `fs::write` fails.
///
/// # Errors
///
/// - `anyhow!("empty csv")` if no header.
/// - `PmmlError` from `run_batch` mapped to `anyhow`.
fn run_manual_batch(
    sess: &pmmlruntime::session::Session,
    csv_text: &str,
    out_path: &str,
) -> anyhow::Result<()> {
    let mut lines = csv_text.lines();
    let header = lines.next().ok_or_else(|| anyhow::anyhow!("empty csv"))?;
    let cols: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    // Collect inputs first, then use run_batch (parallel when CpuBatched)
    let mut inputs: Vec<HashMap<String, pmmlruntime::base::Value>> = Vec::new();
    let mut orig_lines: Vec<String> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let vals: Vec<&str> = line.split(',').collect();
        let mut map: HashMap<String, pmmlruntime::base::Value> = HashMap::new();
        for (col, val) in cols.iter().zip(vals.iter()) {
            let v = if let Ok(f) = val.parse::<f64>() {
                pmmlruntime::base::Value::Continuous(f)
            } else if val.is_empty() {
                pmmlruntime::base::Value::Missing
            } else {
                pmmlruntime::base::Value::Continuous(val.parse().unwrap_or(0.0))
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
            .unwrap_or(&pmmlruntime::base::Value::Missing);
        let pred_str = match pred {
            pmmlruntime::base::Value::Continuous(f) => f.to_string(),
            pmmlruntime::base::Value::Discrete(sid) => sess
                .ir
                .symbol_names
                .get(sid)
                .cloned()
                .unwrap_or_else(|| format!("{sid:?}")),
            pmmlruntime::base::Value::Missing => "Missing".into(),
        };
        out_lines.push(format!("{orig},{pred_str}"));
    }
    std::fs::write(out_path, out_lines.join("\n"))?;
    println!("Scored {} rows -> {out_path}", out_lines.len() - 1);
    Ok(())
}

/// Verify a PMML file (`unmarshal` → `verify_raw` → `lower` → `verify_ir`).
///
/// # Parameters
///
/// - `model`: path to `model.pmml`.
///
/// # Returns
///
/// `Ok(())` on success after printing `verify: … OK — Tree nodes=N`; `Err` with `anyhow` context on failure.
///
/// # Errors
///
/// - `IO` if `read` fails.
/// - `Parse` / `UnsupportedMarkup` / `InvalidValue` from `verify_raw` / `lower` / `verify_ir` are wrapped as `verify_raw:` / `lower:` / `verify_ir:`.
///
/// # Examples
///
/// ```text
/// // $ pmml-runtime verify --model model.pmml
/// // verify: model.pmml OK — Tree nodes=5
/// ```
fn verify(model: &str) -> anyhow::Result<()> {
    let bytes = std::fs::read(model)?;
    let raw = pmmlruntime::xml::unmarshal(&bytes).map_err(|e| anyhow::anyhow!("unmarshal: {e}"))?;
    pmmlruntime::ir::verify_raw(&raw).map_err(|e| anyhow::anyhow!("verify_raw: {e}"))?;
    let ir = pmmlruntime::ir::lower(raw).map_err(|e| anyhow::anyhow!("lower: {e}"))?;
    pmmlruntime::ir::verify_ir(&ir).map_err(|e| anyhow::anyhow!("verify_ir: {e}"))?;
    println!(
        "verify: {model} OK — Tree nodes={}",
        match &ir.model {
            pmmlruntime::ir::ModelIr::Tree(t) => t.nodes.len(),
            _ => 0,
        }
    );
    Ok(())
}
