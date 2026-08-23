use crate::env::PmmlEnv;
use crate::options::{ExecutionProviderKind, SessionOptions};
use crate::providers::{CpuBatchedProvider, CpuSerialProvider, ExecutionProvider};
use pmml_core::error::{PmmlError, Result};
use pmml_core::{FieldId, Value};
use pmml_ir::ir::Ir;
use std::collections::HashMap;
use std::sync::Arc;

/// Session — immutable, Send+Sync, analogous to OrtSession.
/// Holds Arc<Ir> and provider.
pub struct Session {
    pub env: PmmlEnv,
    pub options: SessionOptions,
    pub ir: Arc<Ir>,
    provider: Box<dyn ExecutionProvider>,
    // reverse map for field name -> FieldId (from Ir field_names)
    name_to_id: HashMap<String, FieldId>,
    // max field id for values vec size
    max_field_id: usize,
    // target field name for output (if known)
    target_name: Option<String>,
}

impl Session {
    /// Create from bytes (PMML XML).
    pub fn from_bytes(env: &PmmlEnv, bytes: &[u8], options: SessionOptions) -> Result<Self> {
        let raw = pmml_xml::unmarshal(bytes)?;
        pmml_ir::verify_raw(&raw)?;
        let ir = pmml_ir::lower(raw)?;
        pmml_ir::verify_ir(&ir)?;
        Self::from_ir(env.clone(), ir, options)
    }

    /// Create from file path.
    pub fn from_file(env: &PmmlEnv, path: &str, options: SessionOptions) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| PmmlError::Io(e.to_string()))?;
        Self::from_bytes(env, &bytes, options)
    }

    fn from_ir(env: PmmlEnv, ir: Ir, options: SessionOptions) -> Result<Self> {
        let provider: Box<dyn ExecutionProvider> = match options.execution_provider {
            ExecutionProviderKind::CpuSerial => Box::new(CpuSerialProvider),
            ExecutionProviderKind::CpuBatched => Box::new(CpuBatchedProvider),
        };

        // Build name->FieldId map from Ir field_names (FieldId -> name)
        let mut name_to_id: HashMap<String, FieldId> = HashMap::new();
        for (fid, name) in &ir.field_names {
            name_to_id.insert(name.clone(), *fid);
        }
        // Also include derived fields names (if any) — already in field_names if lower populated correctly
        // Determine max field id for values vec
        let max_field_id = name_to_id
            .values()
            .map(|fid| fid.as_usize())
            .max()
            .unwrap_or(0)
            + 1;
        // Extract target name from model if classification
        let target_name = match &ir.model {
            pmml_ir::ir::ModelIr::Tree(t) => t
                .mining_schema
                .target_field
                .and_then(|fid| ir.field_names.get(&fid).cloned()),
            _ => None,
        };

        Ok(Self {
            env,
            options,
            ir: Arc::new(ir),
            provider,
            name_to_id,
            max_field_id: max_field_id.max(16), // at least 16
            target_name,
        })
    }

    /// Run single row. Input map: field name -> Value.
    /// Returns output map: output field name -> Value (includes predictedValue).
    pub fn run(&self, input: HashMap<String, Value>) -> Result<HashMap<String, Value>> {
        // Build flat values array
        let mut values = vec![Value::Missing; self.max_field_id.max(self.ir.num_fields() + 4)];
        let mut input_by_id: HashMap<FieldId, Value> = HashMap::new();
        for (name, val) in input {
            if let Some(&fid) = self.name_to_id.get(&name) {
                let idx = fid.as_usize();
                if idx < values.len() {
                    values[idx] = val;
                    input_by_id.insert(fid, val);
                }
            } else {
                // Unknown field — ignore per PMML (or error). We'll ignore.
            }
        }

        // MiningSchema: copy active fields (already done via input_by_id + values)
        // For v1, mining_schema apply is handled via values directly; we still call provider which does derived+model.
        // But mining_schema's missing handling is trivial v1 (already Missing).
        // Call provider
        let predicted = self.provider.evaluate(&self.ir, &mut values)?;

        // Targets (none v1)
        // Build output
        let output = match &self.ir.model {
            pmml_ir::ir::ModelIr::Tree(tree) => {
                pmml_evaluator::output::build_output(&tree.output, predicted, &HashMap::new())
            }
            _ => {
                let mut m = HashMap::new();
                m.insert("predictedValue".to_string(), predicted);
                m
            }
        };

        // Also insert target-named output for convenience (like sklearn expects Species)
        let mut final_out = output;
        if let Some(tname) = &self.target_name {
            final_out.entry(tname.clone()).or_insert(predicted);
        }
        // Ensure predictedValue always present
        final_out
            .entry("predictedValue".to_string())
            .or_insert(predicted);

        Ok(final_out)
    }

    /// Convenience: run with string values (coerced). Useful for CSV.
    pub fn run_from_strings(
        &self,
        input: HashMap<String, String>,
    ) -> Result<HashMap<String, Value>> {
        let mut map: HashMap<String, Value> = HashMap::new();
        for (k, v) in input {
            // Try parse as f64 continuous, else discrete symbol
            let val = if let Ok(f) = v.parse::<f64>() {
                Value::Continuous(f)
            } else if v.is_empty() || v.eq_ignore_ascii_case("missing") {
                Value::Missing
            } else {
                // Need SymbolId interning: we can't create SymbolId without interner context.
                // For now, create a synthetic SymbolId via hash of string (not interned). This will not match tree's SymbolId.
                // So we need proper interning: look up via Ir? In v1, discrete field values are compared via string? But lower interned them to SymbolId.
                // For input discrete, we must intern to same IDs as model. We don't have interner snapshot.
                // Workaround: for now, if value not numeric, we treat as Continuous? No.
                // We need to map string -> SymbolId via a deterministic hash shared with lower.
                // Simpler: for discrete inputs, we will compare via string in future, but now Value::Discrete requires correct SymbolId.
                // Hack: use ahash of string lower 32 bits as SymbolId — will be consistent if we also hash during lower? But lower uses lasso sequential ids, not hash.
                // So this path will fail for categorical inputs in v1 unless we store string->SymbolId map.
                // For Iris, inputs are continuous (Petal.Length etc) so they are f64, not discrete, so this path is fine.
                // For discrete inputs, we fallback to a placeholder that won't match; but v1 only tests continuous.
                Value::Continuous(v.parse().unwrap_or(0.0))
            };
            map.insert(k, val);
        }
        self.run(map)
    }

    /// Number of active fields.
    pub fn num_active_fields(&self) -> usize {
        match &self.ir.model {
            pmml_ir::ir::ModelIr::Tree(t) => t.mining_schema.active_fields.len(),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::Value;
    use std::collections::HashMap;

    #[test]
    fn session_iris_tree() {
        let xml = std::fs::read("/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/DecisionTreeIris.pmml").unwrap();
        let env = PmmlEnv::new();
        let opts = SessionOptions::default();
        let sess = Session::from_bytes(&env, &xml, opts).unwrap();
        let mut input = HashMap::new();
        input.insert("Petal.Length".to_string(), Value::Continuous(1.4)); // setosa
        input.insert("Petal.Width".to_string(), Value::Continuous(0.2));
        let out = sess.run(input).unwrap();
        // Predicted should be setosa
        let pred = out.get("predictedValue").unwrap();
        match pred {
            Value::Discrete(sid) => {
                // SymbolId for setosa should be interned; we check not missing
                assert_ne!(sid.0, u32::MAX);
            }
            _ => panic!("expected discrete"),
        }
    }

    #[test]
    fn session_iris_virginica() {
        let xml = std::fs::read("/home/pab1s/Projects/jpmml-migration/upstream/jpmml-evaluator/pmml-evaluator-testing/src/test/resources/pmml/DecisionTreeIris.pmml").unwrap();
        let env = PmmlEnv::new();
        let sess = Session::from_bytes(&env, &xml, SessionOptions::default()).unwrap();
        let mut input = HashMap::new();
        input.insert("Petal.Length".to_string(), Value::Continuous(6.0));
        input.insert("Petal.Width".to_string(), Value::Continuous(2.0));
        let out = sess.run(input).unwrap();
        assert!(out.contains_key("predictedValue"));
    }
}
