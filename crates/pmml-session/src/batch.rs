//! Batch abstraction — ONNX-style `Batch`/`BatchResult` for PMML scoring.
//!
//! Design mirrors ONNX Runtime `OrtValue` + `OrtIoBinding`:
//! - `Batch` is the single logical input type, with two physical layouts:
//!   * `RowMajor` (`Vec<HashMap<String,Value>>` / `&[HashMap]`) — JPMML compat, ergonomic for single row
//!   * `Columnar` (`RecordBatch`) — Arrow zero-copy, for >10k rows (16.5M rows/s)
//! - `BatchCtx` holds `Session`'s `name_to_id`/`symbol_str_to_id`/`Ir` refs to avoid per-row allocation
//! - `ExecutionProvider::eval_batch` shards via rayon; `Session` only materializes `Value[FieldId]`
//!
//! If Arrow is always better, why not only Arrow? See `docs/PORTING.md` E1 and `BENCHMARK.md` §5:
//! Arrow wins at 100k (61ns/row) but loses for single row (conversion >1µs) and needs schema agreement.
//! `Collection`/`List` (Association) and Python `dict` map naturally to `HashMap`. So keep both, provider picks.

use crate::arrow::value_maps_to_record_batch;
use ahash::AHashMap;
use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;
use pmml_core::error::{PmmlError, Result};
use pmml_core::{FieldId, SymbolId, Value};
use pmml_ir::ir::Ir;
use std::collections::HashMap;
use std::sync::Arc;

/// Hint for provider to choose SIMD/parallel strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchFormat {
    RowMajor,
    Columnar,
}

/// Context for materialization — refs to `Session`'s caches, no per-row alloc.
pub struct BatchCtx<'a> {
    pub name_to_id: &'a AHashMap<String, FieldId>,
    pub name_to_id_std: &'a HashMap<String, FieldId>,
    pub symbol_str_to_id: &'a HashMap<String, SymbolId>,
    pub symbol_names: &'a HashMap<SymbolId, String>,
    pub ir: &'a Ir,
    pub max_field_id: usize,
    // Precomputed for RecordBatch: (FieldId, column index)
    // For RowMajor batches this is empty.
    pub col_map: Vec<(FieldId, usize)>,
    // For output mapping
    pub output_fields: &'a [pmml_ir::ir::OutputFieldIr],
    pub target_name: Option<&'a String>,
    pub symbol_names_vec: &'a [String],
}

impl<'a> BatchCtx<'a> {
    /// Build ctx for a generic batch (no col_map).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name_to_id: &'a AHashMap<String, FieldId>,
        name_to_id_std: &'a HashMap<String, FieldId>,
        symbol_str_to_id: &'a HashMap<String, SymbolId>,
        symbol_names: &'a HashMap<SymbolId, String>,
        ir: &'a Ir,
        max_field_id: usize,
        output_fields: &'a [pmml_ir::ir::OutputFieldIr],
        target_name: Option<&'a String>,
        symbol_names_vec: &'a [String],
    ) -> Self {
        Self {
            name_to_id,
            name_to_id_std,
            symbol_str_to_id,
            symbol_names,
            ir,
            max_field_id,
            col_map: Vec::new(),
            output_fields,
            target_name,
            symbol_names_vec,
        }
    }

    /// Build ctx for RecordBatch — precompute col_map (FieldId → column index).
    #[allow(clippy::too_many_arguments)]
    pub fn for_record_batch(
        name_to_id: &'a AHashMap<String, FieldId>,
        name_to_id_std: &'a HashMap<String, FieldId>,
        symbol_str_to_id: &'a HashMap<String, SymbolId>,
        symbol_names: &'a HashMap<SymbolId, String>,
        ir: &'a Ir,
        max_field_id: usize,
        output_fields: &'a [pmml_ir::ir::OutputFieldIr],
        target_name: Option<&'a String>,
        symbol_names_vec: &'a [String],
        batch: &RecordBatch,
    ) -> Self {
        let mut col_map = Vec::new();
        for (col_idx, field) in batch.schema().fields().iter().enumerate() {
            if let Some(&fid) = name_to_id.get(field.name().as_str()) {
                col_map.push((fid, col_idx));
            }
        }
        Self {
            name_to_id,
            name_to_id_std,
            symbol_str_to_id,
            symbol_names,
            ir,
            max_field_id,
            col_map,
            output_fields,
            target_name,
            symbol_names_vec,
        }
    }
}

/// Logical batch — row-major or columnar. Object-safe for `&dyn Batch`.
pub trait Batch: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn format(&self) -> BatchFormat;
    /// Materialize `row` into `values[FieldId.as_usize()] = Value`.
    /// `values` is already zeroed with `Missing`; impl only overwrites known fields.
    fn materialize_row(&self, row: usize, values: &mut [Value], ctx: &BatchCtx) -> Result<()>;
}

/// Result of a batch execution.
pub enum BatchResult {
    /// Row-major `Vec<HashMap>` — used for `Vec<HashMap>` inputs and Arrow inputs that still output rows.
    Rows(Vec<HashMap<String, Value>>),
    /// Columnar `RecordBatch` — used when caller wants Arrow output (e.g. `run_record_batch`).
    Columnar(RecordBatch),
}

impl BatchResult {
    pub fn len(&self) -> usize {
        match self {
            BatchResult::Rows(v) => v.len(),
            BatchResult::Columnar(b) => b.num_rows(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Convert `Rows` to `RecordBatch` via `value_maps_to_record_batch`.
    pub fn into_record_batch(
        self,
        schema: Arc<arrow::datatypes::Schema>,
        symbol_names: Option<&HashMap<SymbolId, String>>,
    ) -> Result<RecordBatch> {
        match self {
            BatchResult::Rows(maps) => value_maps_to_record_batch(&maps, schema, symbol_names)
                .map_err(PmmlError::InvalidValue),
            BatchResult::Columnar(b) => Ok(b),
        }
    }
    /// Unwrap `Rows` for backward compat.
    pub fn into_rows(self) -> Vec<HashMap<String, Value>> {
        match self {
            BatchResult::Rows(v) => v,
            BatchResult::Columnar(b) => crate::arrow::record_batch_to_value_maps(&b),
        }
    }
}

// Row-major: Vec<HashMap<String, Value>>
impl Batch for Vec<HashMap<String, Value>> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn format(&self) -> BatchFormat {
        BatchFormat::RowMajor
    }
    fn materialize_row(&self, row: usize, values: &mut [Value], ctx: &BatchCtx) -> Result<()> {
        let map = &self[row];
        for (name, val) in map {
            if let Some(&fid) = ctx.name_to_id.get(name) {
                let idx = fid.as_usize();
                if idx < values.len() {
                    values[idx] = *val;
                }
            }
        }
        Ok(())
    }
}

// Row-major slice via &[HashMap] — also support Vec's slice for &dyn Batch
impl Batch for [HashMap<String, Value>] {
    fn len(&self) -> usize {
        <[HashMap<String, Value>]>::len(self)
    }
    fn format(&self) -> BatchFormat {
        BatchFormat::RowMajor
    }
    fn materialize_row(&self, row: usize, values: &mut [Value], ctx: &BatchCtx) -> Result<()> {
        let map = &self[row];
        for (name, val) in map {
            if let Some(&fid) = ctx.name_to_id.get(name) {
                let idx = fid.as_usize();
                if idx < values.len() {
                    values[idx] = *val;
                }
            }
        }
        Ok(())
    }
}

// Columnar: RecordBatch
impl Batch for RecordBatch {
    fn len(&self) -> usize {
        self.num_rows()
    }
    fn format(&self) -> BatchFormat {
        BatchFormat::Columnar
    }
    fn materialize_row(&self, row: usize, values: &mut [Value], ctx: &BatchCtx) -> Result<()> {
        for (fid, col_idx) in &ctx.col_map {
            let col = self.column(*col_idx);
            if col.is_null(row) {
                values[fid.as_usize()] = Value::Missing;
            } else {
                let val = match col.data_type() {
                    ArrowDataType::Float64 => {
                        let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                        Value::Continuous(arr.value(row))
                    }
                    ArrowDataType::Utf8 => {
                        let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                        let s = arr.value(row);
                        if let Some(sid) = ctx.symbol_str_to_id.get(s) {
                            Value::Discrete(*sid)
                        } else if let Ok(f) = s.parse::<f64>() {
                            // Numeric string where discrete not found — treat as Continuous
                            Value::Continuous(f)
                        } else {
                            Value::Missing
                        }
                    }
                    _ => Value::Missing,
                };
                values[fid.as_usize()] = val;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use pmml_core::{FieldId, SymbolId, Value};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn dummy_ctx<'a>(
        name_to_id: &'a AHashMap<String, FieldId>,
        name_to_id_std: &'a HashMap<String, FieldId>,
        symbol_str_to_id: &'a HashMap<String, SymbolId>,
        symbol_names: &'a HashMap<SymbolId, String>,
        ir: &'a Ir,
        max_field_id: usize,
        output_fields: &'a [pmml_ir::ir::OutputFieldIr],
        symbol_names_vec: &'a [String],
    ) -> BatchCtx<'a> {
        BatchCtx::new(
            name_to_id,
            name_to_id_std,
            symbol_str_to_id,
            symbol_names,
            ir,
            max_field_id,
            output_fields,
            None,
            symbol_names_vec,
        )
    }

    #[test]
    fn batch_row_major_materialize() {
        let mut name_to_id = AHashMap::new();
        name_to_id.insert("f0".to_string(), FieldId(0));
        name_to_id.insert("f1".to_string(), FieldId(1));
        let symbol_str_to_id = HashMap::new();
        let symbol_names = HashMap::new();
        // minimal Ir
        let ir = Ir {
            data_dictionary: vec![],
            derived_fields: vec![],
            model: pmml_ir::ir::ModelIr::Tree(pmml_ir::ir::TreeIr {
                function_name: "classification".into(),
                missing_value_strategy: pmml_ir::ir::MissingValueStrategy::None,
                no_true_child_strategy: pmml_ir::ir::NoTrueChildStrategy::ReturnNullPrediction,
                nodes: vec![],
                mining_schema: pmml_ir::ir::MiningSchemaIr {
                    active_fields: vec![FieldId(0), FieldId(1)],
                    target_field: None,
                    field_metas: vec![],
                    missing_value_replacement: None,
                },
                targets: vec![],
                output: vec![],
            }),
            field_names: HashMap::new(),
            symbol_names: HashMap::new(),
            extensions: vec![],
            element_coverage: 0,
        };
        let output_fields: Vec<pmml_ir::ir::OutputFieldIr> = vec![];
        let symbol_names_vec: Vec<String> = vec![];
        let name_to_id_std: HashMap<String, FieldId> = HashMap::new();
        let ctx = dummy_ctx(
            &name_to_id,
            &name_to_id_std,
            &symbol_str_to_id,
            &symbol_names,
            &ir,
            4,
            &output_fields,
            &symbol_names_vec,
        );
        let batch = vec![
            {
                let mut m = HashMap::new();
                m.insert("f0".to_string(), Value::Continuous(1.0));
                m.insert("f1".to_string(), Value::Continuous(2.0));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("f0".to_string(), Value::Continuous(3.0));
                m
            },
        ];
        let mut values = vec![Value::Missing; 4];
        batch.materialize_row(0, &mut values, &ctx).unwrap();
        assert_eq!(values[0], Value::Continuous(1.0));
        assert_eq!(values[1], Value::Continuous(2.0));
        values.fill(Value::Missing);
        batch.materialize_row(1, &mut values, &ctx).unwrap();
        assert_eq!(values[0], Value::Continuous(3.0));
        assert_eq!(values[1], Value::Missing);
    }

    #[test]
    fn batch_columnar_materialize() {
        let mut name_to_id = AHashMap::new();
        name_to_id.insert("x".to_string(), FieldId(0));
        name_to_id.insert("y".to_string(), FieldId(1));
        let symbol_str_to_id = HashMap::new();
        let symbol_names = HashMap::new();
        let ir = Ir {
            data_dictionary: vec![],
            derived_fields: vec![],
            model: pmml_ir::ir::ModelIr::Tree(pmml_ir::ir::TreeIr {
                function_name: "classification".into(),
                missing_value_strategy: pmml_ir::ir::MissingValueStrategy::None,
                no_true_child_strategy: pmml_ir::ir::NoTrueChildStrategy::ReturnNullPrediction,
                nodes: vec![],
                mining_schema: pmml_ir::ir::MiningSchemaIr {
                    active_fields: vec![FieldId(0), FieldId(1)],
                    target_field: None,
                    field_metas: vec![],
                    missing_value_replacement: None,
                },
                targets: vec![],
                output: vec![],
            }),
            field_names: HashMap::new(),
            symbol_names: HashMap::new(),
            extensions: vec![],
            element_coverage: 0,
        };
        let output_fields: Vec<pmml_ir::ir::OutputFieldIr> = vec![];
        let symbol_names_vec: Vec<String> = vec![];
        let name_to_id_std: HashMap<String, FieldId> = HashMap::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, true),
            Field::new("y", ArrowDataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![Some(1.0), Some(3.0)])) as _,
                Arc::new(Float64Array::from(vec![Some(2.0), None])) as _,
            ],
        )
        .unwrap();
        let ctx = BatchCtx::for_record_batch(
            &name_to_id,
            &name_to_id_std,
            &symbol_str_to_id,
            &symbol_names,
            &ir,
            4,
            &output_fields,
            None,
            &symbol_names_vec,
            &batch,
        );
        let mut values = vec![Value::Missing; 4];
        batch.materialize_row(0, &mut values, &ctx).unwrap();
        assert_eq!(values[0], Value::Continuous(1.0));
        assert_eq!(values[1], Value::Continuous(2.0));
        values.fill(Value::Missing);
        batch.materialize_row(1, &mut values, &ctx).unwrap();
        assert_eq!(values[0], Value::Continuous(3.0));
        assert_eq!(values[1], Value::Missing);
    }
}
