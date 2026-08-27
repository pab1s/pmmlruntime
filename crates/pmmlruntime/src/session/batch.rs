//! Batch abstraction — session-style `Batch`/`BatchResult` for PMML scoring.
//!
//! Design mirrors session runtime `Value` + `batch binding`:
//! - `Batch` is the single logical input type, with two physical layouts:
//!   * `RowMajor` (`Vec<HashMap<String,Value>>` / `&[HashMap]`) — PMML compat, ergonomic for single row
//!   * `Columnar` (`RecordBatch`) — Arrow zero-copy, for `>10k` rows (16.5M rows/s)
//! - `BatchCtx` holds `Session`'s `name_to_id`/`symbol_str_to_id`/`Ir` refs to avoid per-row allocation
//! - `ExecutionProvider::eval_batch` shards via `rayon`; `Session` only materializes `Value[FieldId]`
//!
//! If Arrow is always better, why not only Arrow? See `docs/PORTING.md` E1 and `BENCHMARK.md` §5:
//! Arrow wins at 100k (61ns/row) but loses for single row (conversion `>1µs`) and needs schema agreement.
//! `Collection`/`List` (Association) and Python `dict` map naturally to `HashMap`. So keep both, provider picks.
//!
//! # What belongs here
//!
//! - [`BatchFormat`] — hint for provider (`RowMajor` vs `Columnar`).
//! - [`BatchCtx`] — no-per-row-alloc context (refs to `Session` caches + `Ir` + `col_map` for `RecordBatch`).
//! - [`Batch`] trait — object-safe `Send+Sync` with `materialize_row` to `Value[FieldId]`.
//! - [`BatchResult`] — `Rows(Vec<HashMap>)` or `Columnar(RecordBatch)`, with helpers to convert.

use crate::base::error::{PmmlError, Result};
use crate::base::{FieldId, SymbolId, Value};
use crate::ir::Ir;
use crate::session::arrow::value_maps_to_record_batch;
use ahash::AHashMap;
use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

/// Hint for provider to choose SIMD/parallel strategy.
///
/// `RowMajor` is `Vec<HashMap<String, Value>>` (PMML compat, ergonomic). `Columnar` is `RecordBatch`
/// (Arrow zero-copy, best at `>10k` rows). Provider's [`preferred_format`](crate::session::providers::ExecutionProvider::preferred_format)
/// hints, but `Session` keeps both so callers aren't forced into Arrow for single rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchFormat {
    /// Row-major: `Vec<HashMap<String, Value>>` or `&[HashMap]`.
    RowMajor,
    /// Columnar: `RecordBatch`.
    Columnar,
}

/// Context for materialization — refs to `Session`'s caches, no per-row alloc.
///
/// Built by `Session::run_batch*` and passed to `ExecutionProvider::eval_batch` and `Batch::materialize_row`.
/// For `RecordBatch` it precomputes `col_map: Vec<(FieldId, column_index)>` so materialization is a direct
/// array lookup without `HashMap` per row.
///
/// All refs are borrowed from `Session` + `Ir`; `BatchCtx` itself is not `Send` due to lifetimes, but
/// `Batch` impls for `RecordBatch` read `col_map` without mutation, and `rayon` shards capture `&BatchCtx: Sync` via scoped closure.
///
/// # Examples
///
/// ```no_run
/// use pmmlruntime::session::batch::BatchCtx;
/// # let ctx: BatchCtx = unimplemented!();
/// let needed = ctx.max_field_id;
/// # let _ = needed;
/// ```
pub struct BatchCtx<'a> {
    /// `field name → FieldId` hot map (`AHashMap` for 3× vs `SipHash`).
    pub name_to_id: &'a AHashMap<String, FieldId>,
    /// `field name → FieldId` std map for `MiningModel`/`GeneralRegression` evaluator (cached).
    pub name_to_id_std: &'a HashMap<String, FieldId>,
    /// `String → SymbolId` forward map for categorical `Utf8` → `Discrete`.
    pub symbol_str_to_id: &'a HashMap<String, SymbolId>,
    /// `SymbolId → String` reverse map (dense `symbol_names` from `Ir`).
    pub symbol_names: &'a HashMap<SymbolId, String>,
    /// Immutable lowered model (for `MiningSchema` etc.).
    pub ir: &'a Ir,
    /// `max(FieldId)+1` clamped to at least 16, size for `values` slice.
    pub max_field_id: usize,
    // Precomputed for RecordBatch: (FieldId, column index)
    // For RowMajor batches this is empty.
    /// Precomputed `FieldId → column index` for `RecordBatch` (empty for `RowMajor`).
    pub col_map: Vec<(FieldId, usize)>,
    // For output mapping
    /// Cached `OutputFieldIr` slice (pre-resolved, avoids per-row `ModelIr` match).
    pub output_fields: &'a [crate::ir::OutputFieldIr],
    /// Target field name if known (inserted alongside `predictedValue`).
    pub target_name: Option<&'a String>,
    /// Dense `SymbolId.0 as usize → String` (cache-line friendly for probability output).
    pub symbol_names_vec: &'a [String],
}

impl<'a> BatchCtx<'a> {
    /// Build ctx for a generic batch (no `col_map`).
    ///
    /// Used for `Vec<HashMap>` / `&[HashMap]` where `materialize_row` looks up `name_to_id` per key.
    ///
    /// # Parameters
    ///
    /// All refs are from `Session` caches + `Ir`; `max_field_id` vs `num_fields()+4` is precomputed by `Session`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name_to_id: &'a AHashMap<String, FieldId>,
        name_to_id_std: &'a HashMap<String, FieldId>,
        symbol_str_to_id: &'a HashMap<String, SymbolId>,
        symbol_names: &'a HashMap<SymbolId, String>,
        ir: &'a Ir,
        max_field_id: usize,
        output_fields: &'a [crate::ir::OutputFieldIr],
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

    /// Build ctx for `RecordBatch` — precompute `col_map` (`FieldId → column index`).
    ///
    /// Iterates `batch.schema().fields()` and keeps only columns whose name is in `name_to_id`.
    /// This avoids `HashMap<String, Value>` per row; `materialize_row` then reads `Float64Array`/`StringArray`
    /// directly.
    ///
    /// # Parameters
    ///
    /// - `batch`: `RecordBatch` whose `Schema` field names are matched against `name_to_id`.
    #[allow(clippy::too_many_arguments)]
    pub fn for_record_batch(
        name_to_id: &'a AHashMap<String, FieldId>,
        name_to_id_std: &'a HashMap<String, FieldId>,
        symbol_str_to_id: &'a HashMap<String, SymbolId>,
        symbol_names: &'a HashMap<SymbolId, String>,
        ir: &'a Ir,
        max_field_id: usize,
        output_fields: &'a [crate::ir::OutputFieldIr],
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
///
/// Implementors are `Send + Sync` so `ExecutionProvider::eval_batch` can shard with `rayon`.
/// The hot method is `materialize_row`, which fills `values[FieldId]` for one row.
/// `BatchCtx` carries `name_to_id` / `col_map` so materialization needs no per-row `HashMap` clone for `RecordBatch`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::batch::{Batch, BatchCtx};
/// use pmmlruntime::base::Value;
/// use std::collections::HashMap;
/// // Vec<HashMap> batch
/// let batch = vec![{ let mut m=HashMap::new(); m.insert("x".into(), Value::Continuous(1.0)); m }];
/// assert_eq!(batch.len(), 1);
/// assert_eq!(batch.format(), pmmlruntime::session::batch::BatchFormat::RowMajor);
/// ```
pub trait Batch: Send + Sync {
    /// Number of rows in the batch.
    fn len(&self) -> usize;
    /// `true` if `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Physical layout hint (`RowMajor` vs `Columnar`).
    fn format(&self) -> BatchFormat;
    /// Materialize `row` into `values[FieldId.as_usize()] = Value`.
    /// `values` is already zeroed with `Missing`; impl only overwrites known fields.
    ///
    /// # Parameters
    ///
    /// - `row`: row index `0..len()`.
    /// - `values`: `&mut [Value]` sized to `ctx.max_field_id`, initialized to `Missing`.
    /// - `ctx`: refs to `name_to_id` / `col_map` / `symbol_str_to_id`.
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` always in current impls (missing columns become `Missing`), but
    /// future impls could return `PmmlError::InvalidValue` for type mismatches.
    fn materialize_row(&self, row: usize, values: &mut [Value], ctx: &BatchCtx) -> Result<()>;
}

/// Result of a batch execution.
///
/// `Rows` is `Vec<HashMap>` (row-major output), `Columnar` is `RecordBatch` (when caller wants Arrow output).
/// `Session::run_batch` always returns `Rows` for now; `run_record_batch` converts via `into_record_batch`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::batch::BatchResult;
/// use std::collections::HashMap;
/// let br = BatchResult::Rows(vec![]);
/// assert!(br.is_empty());
/// ```
pub enum BatchResult {
    /// Row-major `Vec<HashMap>` — used for `Vec<HashMap>` inputs and Arrow inputs that still output rows.
    Rows(Vec<HashMap<String, Value>>),
    /// Columnar `RecordBatch` — used when caller wants Arrow output (e.g. `run_record_batch`).
    Columnar(RecordBatch),
}

impl BatchResult {
    /// Number of rows in the result.
    pub fn len(&self) -> usize {
        match self {
            BatchResult::Rows(v) => v.len(),
            BatchResult::Columnar(b) => b.num_rows(),
        }
    }
    /// `true` if `len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Convert `Rows` to `RecordBatch` via `value_maps_to_record_batch`.
    ///
    /// # Parameters
    ///
    /// - `schema`: Arrow schema for output columns.
    /// - `symbol_names`: optional `SymbolId → String` for `Discrete` resolution.
    ///
    /// # Returns
    ///
    /// `Ok(RecordBatch)` or `Err(PmmlError::InvalidValue)` if `RecordBatch::try_new` fails.
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
    /// Unwrap `Rows` for backward compat; `Columnar` is converted via `record_batch_to_value_maps`.
    ///
    /// # Returns
    ///
    /// `Vec<HashMap<String, Value>>` one per row.
    pub fn into_rows(self) -> Vec<HashMap<String, Value>> {
        match self {
            BatchResult::Rows(v) => v,
            BatchResult::Columnar(b) => crate::session::arrow::record_batch_to_value_maps(&b),
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
    use crate::base::{FieldId, SymbolId, Value};
    use ahash::AHashMap;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[allow(clippy::too_many_arguments)]
    fn dummy_ctx<'a>(
        name_to_id: &'a AHashMap<String, FieldId>,
        name_to_id_std: &'a HashMap<String, FieldId>,
        symbol_str_to_id: &'a HashMap<String, SymbolId>,
        symbol_names: &'a HashMap<SymbolId, String>,
        ir: &'a Ir,
        max_field_id: usize,
        output_fields: &'a [crate::ir::OutputFieldIr],
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
            model: crate::ir::ModelIr::Tree(crate::ir::TreeIr {
                function_name: "classification".into(),
                missing_value_strategy: crate::ir::MissingValueStrategy::None,
                no_true_child_strategy: crate::ir::NoTrueChildStrategy::ReturnNullPrediction,
                nodes: vec![],
                mining_schema: crate::ir::MiningSchemaIr {
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
        let output_fields: Vec<crate::ir::OutputFieldIr> = vec![];
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
            model: crate::ir::ModelIr::Tree(crate::ir::TreeIr {
                function_name: "classification".into(),
                missing_value_strategy: crate::ir::MissingValueStrategy::None,
                no_true_child_strategy: crate::ir::NoTrueChildStrategy::ReturnNullPrediction,
                nodes: vec![],
                mining_schema: crate::ir::MiningSchemaIr {
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
        let output_fields: Vec<crate::ir::OutputFieldIr> = vec![];
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
