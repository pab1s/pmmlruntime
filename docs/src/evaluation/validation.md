# Validation & Threshold Gating

> **Info:** Looking for the fixture suite? See [Correctness & Fixture Parity](./correctness.md). Looking for the latency budget? See [Performance: Cold, Hot, and Batch](./performance.md).

Score held-out rows, compute the metric you care about, and fail the build when the metric misses its threshold. pmmlruntime gives you `Session::run`; you own the metric and the gate.

## Concepts

| Concept | Description |
| --- | --- |
| **ModelVerification** | A PMML block with expected output per row: `VerificationFields` plus an `InlineTable`. |
| **Holdout set** | A CSV whose header matches the active fields, plus a label column you keep out of scoring. |
| **Threshold** | The value that fails the build. Accuracy above 0.95 is a common starting point. |

## Get expected values

`ScalarVerificationTest.pmml` is a PMML 4.2 RPart tree whose `ModelVerification` block lists expected values for five rows, including `predicted species` and the three probabilities. pmmlruntime parses the model and leaves that block in the XML, so read the rows from the file, feed the input columns through `Session::run`, and compare. Alternatively, score a holdout CSV you trust and compare `predictedValue` against its label column.

## Score and gate

The example below scores two rows and fails when accuracy drops under 0.95:

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions, Value};
use pmmlruntime::session::batch::Batch;
use pmmlruntime::session::arrow::csv_str_to_record_batch;

let env = PmmlEnv::new();
let bytes = std::fs::read("bench/pmml/DecisionTreeIris.pmml")?;
let sess = Session::from_bytes(&env, &bytes, SessionOptions::default())?;

let csv = "Petal.Length,Petal.Width,species\n1.4,0.2,setosa\n6.0,2.0,virginica\n";
let batch = csv_str_to_record_batch(csv, None, true).map_err(|e| anyhow::anyhow!(e))?;
let outs = sess.run(&batch as &dyn Batch)?.into_rows();

let expected = ["setosa", "virginica"];
let correct = outs.iter().zip(expected.iter()).filter(|(out, exp)| {
    let pred = out.get("predictedValue").and_then(|v| match v {
        Value::Discrete(sid) => sess.ir.symbol_names.get(sid).map(String::as_str),
        _ => None,
    }).unwrap_or("");
    pred == *exp
}).count();
let accuracy = correct as f64 / expected.len() as f64;
assert!(accuracy >= 0.95, "accuracy {accuracy:.3} below threshold");
```

Compare labels through `sess.ir.symbol_names` rather than through the raw `SymbolId`, because the id is dense per `Ir` and only the symbol table decodes it back to a label.

> **Note:** Keep the label column out of the scoring input. Score the active fields from `MiningSchema`, then read `predictedValue` after `build_output`.

## Metrics worth gating

| Metric | Shape of the code | Direction that passes |
| --- | --- | --- |
| Accuracy | Count matching `Discrete` labels, divide by row count. | Higher. |
| RMSE | Diff `Continuous` predictions against labels, square, average, take the square root. | Lower. |

One `run` gives you every metric, so score once and compute all of them over the same `outs`.

## Check the probability output

Classification models declare `probability(...)` fields. Those fields sum to 1.0 on a row when the scored node carries `ScoreDistribution` values, which is how `lower` fills them. The Iris fixture shows the other case: its leaves declare no distribution, so every probability stays 0 while `predictedValue` still resolves. Treat a change in either value as a regression in `build_output` or in the node distributions, not as a model problem.

```rust
use pmmlruntime::Value;
use std::collections::HashMap;

fn check_probability_sum(outs: &[HashMap<String, Value>], classes: &[&str]) -> Result<(), String> {
    for row in outs {
        let sum: f64 = classes.iter()
            .map(|cls| match row.get(&format!("probability({cls})")) {
                Some(Value::Continuous(c)) => *c,
                _ => 0.0,
            })
            .sum();
        if (sum - 1.0).abs() > 0.01 {
            return Err(format!("probability sum {sum:.3} is not 1.0"));
        }
    }
    Ok(())
}
```

Four `ResultFeature` values never produce a number: `standardError`, `standardDeviation`, `confidenceIntervalLower`, and `confidenceIntervalUpper`. They resolve to `Value::Missing`. Use `build_output_strict` to fail instead of reading a missing column.

## Diff against a Java run

When you already produce expected scores with a JPMML harness, score the same CSV and diff the two files. `bench/BENCHMARK.md` holds that harness and its raw output. Column ordering differences are harmless, so sort the header before comparing. A value difference points at `lower` or `build_output`, and the fixture suite in [Correctness & Fixture Parity](./correctness.md) pinpoints which of the two moved.

## Wire it into CI

Drop the gate into `crates/pmmlruntime/tests/` so `cargo test` runs it on every push:

```rust
#[test]
fn validation_accuracy_gating() -> Result<(), Box<dyn std::error::Error>> {
    use pmmlruntime::{PmmlEnv, Session, SessionOptions};
    use pmmlruntime::session::batch::Batch;
    use pmmlruntime::session::arrow::csv_str_to_record_batch;

    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &std::fs::read("bench/pmml/DecisionTreeIris.pmml")?, SessionOptions::default())?;
    let csv = std::fs::read_to_string("bench/csv/holdout.csv")?;
    let batch = csv_str_to_record_batch(&csv, None, true).map_err(|e| anyhow::anyhow!(e))?;
    let rows = sess.run(&batch as &dyn Batch)?.into_rows();
    assert!(!rows.is_empty());
    Ok(())
}
```

Add the boundary checks to the same job, because a threshold gate is only as good as its input.

```bash
cargo test --test all_fixtures -- --nocapture
cargo test --test hardening -- --nocapture
cargo fuzz run fuzz_unmarshal -- -max_total_time=60
```

> **Warning:** A gate on synthetic rows proves the pipeline runs, not that the model is right. Gate model quality on rows you hold out from training, and keep the fixture suite for engine parity.

## Next Steps

* [Correctness & Fixture Parity](./correctness.md): reproduce the 52 fixture suite that guards the engine.
* [Performance: Cold, Hot, and Batch](./performance.md): set a latency budget beside your accuracy threshold.
* [MiningSchema, DataDictionary & Output](../concepts/schema.md): read `Output` features before computing a metric.
* [Quickstart: Score Iris in 5 Minutes](../getting-started/quickstart.md): wire `Session::from_bytes` and a batch in six steps.

*Next: [Security & Hardening](../production/security.md) → · Previous: [Performance: Cold, Hot, and Batch](./performance.md)*
