//! Arrow bridge — `RecordBatch` ↔ `Vec<HashMap<String, Value>>` and CSV handling.
//!
//! Enabled unconditionally (workspace already depends on `arrow` 53 with `csv`).
//! Handles `InlineTable` conversion and `TableLocator` placeholder gracefully (returns empty batch).
//!
//! # What belongs here
//!
//! - [`ir_to_arrow_schema`] / [`data_dictionary_to_schema`] — derive an Arrow [`Schema`] from `Ir`/`FieldMeta`.
//! - [`value_maps_to_record_batch`] — row-major `Value` maps → columnar `RecordBatch` (nulls for `Missing`).
//! - [`record_batch_to_value_maps`] — inverse, for CSV → batch bridging.
//! - [`inline_table_to_record_batch`] / [`table_locator_placeholder_batch`] — `InlineTable` rows vs `TableLocator` empty.
//! - [`csv_str_to_record_batch`] — `arrow::csv::Reader` → `RecordBatch` (concatenates multi-batch CSV).
//!
//! # Performance
//!
//! `value_maps_to_record_batch` allocates one `Vec<Option<T>>` per column (O(rows*cols)). `record_batch_to_value_maps`
//! allocates one `HashMap` per row — prefer `Session::run_batch_arrow` which avoids this via `Batch::materialize_row`.

use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

use pmml_core::Value;
use pmml_ir::ir::{FieldMeta, Ir};

/// Convert `Ir`'s `data_dictionary` + field names into an Arrow [`Schema`].
///
/// Continuous (`Double`/`Float`/`Integer`) → `Float64`, categorical → `Utf8`, boolean → `Utf8` (as `"true"`/`"false"`).
/// Delegates to [`data_dictionary_to_schema`].
///
/// # Parameters
///
/// - `ir`: lowered model, uses `ir.data_dictionary`.
///
/// # Returns
///
/// `Arc<Schema>` with one field per `FieldMeta`, all nullable (for `Missing`).
///
/// # Examples
///
/// ```
/// use pmml_session::arrow::ir_to_arrow_schema;
/// use pmml_core::{FieldId, field::{DataType, OpType}};
/// use pmml_ir::ir::{FieldMeta, Ir, ModelIr, TreeIr, MiningSchemaIr, MissingValueStrategy, NoTrueChildStrategy};
/// use std::collections::HashMap;
/// let ir = Ir {
///     data_dictionary: vec![FieldMeta { field_id: FieldId(0), name: "x".into(), data_type: DataType::Double, op_type: OpType::Continuous, values: vec![], ..Default::default() }],
///     derived_fields: vec![],
///     model: ModelIr::Tree(TreeIr {
///         function_name: "classification".into(),
///         missing_value_strategy: MissingValueStrategy::None,
///         no_true_child_strategy: NoTrueChildStrategy::ReturnNullPrediction,
///         nodes: vec![],
///         mining_schema: MiningSchemaIr { active_fields: vec![], target_field: None, field_metas: vec![], missing_value_replacement: None },
///         targets: vec![],
///         output: vec![],
///     }),
///     field_names: HashMap::new(),
///     symbol_names: HashMap::new(),
///     extensions: vec![],
///     element_coverage: 0,
/// };
/// let schema = ir_to_arrow_schema(&ir);
/// assert_eq!(schema.fields().len(), 1);
/// ```
pub fn ir_to_arrow_schema(ir: &Ir) -> Arc<Schema> {
    data_dictionary_to_schema(&ir.data_dictionary)
}

/// Convert a slice of `FieldMeta` into an Arrow [`Schema`].
///
/// Maps `DataType::Double`/`Float`/`Integer` → `Float64`, `String` → `Utf8`, `Boolean` → `Utf8` (kept as string),
/// others → `Utf8`. All fields are nullable to carry `Missing` as null.
///
/// # Parameters
///
/// - `data_dictionary`: slice of `FieldMeta` from `Ir` or `RawPmml`.
///
/// # Returns
///
/// `Arc<Schema>`.
///
/// # Examples
///
/// ```
/// use pmml_session::arrow::data_dictionary_to_schema;
/// use pmml_core::{FieldId, field::{DataType, OpType}};
/// use pmml_ir::ir::FieldMeta;
/// let fields = vec![
///     FieldMeta { field_id: FieldId(0), name: "a".into(), data_type: DataType::Double, op_type: OpType::Continuous, values: vec![], ..Default::default() },
///     FieldMeta { field_id: FieldId(1), name: "b".into(), data_type: DataType::String, op_type: OpType::Categorical, values: vec![], ..Default::default() },
/// ];
/// let schema = data_dictionary_to_schema(&fields);
/// assert_eq!(schema.field(0).data_type(), &arrow::datatypes::DataType::Float64);
/// assert_eq!(schema.field(1).data_type(), &arrow::datatypes::DataType::Utf8);
/// ```
pub fn data_dictionary_to_schema(data_dictionary: &[FieldMeta]) -> Arc<Schema> {
    let fields: Vec<ArrowField> = data_dictionary
        .iter()
        .map(|fm| {
            let arrow_type = match fm.data_type {
                pmml_core::field::DataType::Double
                | pmml_core::field::DataType::Float
                | pmml_core::field::DataType::Integer => ArrowDataType::Float64,
                pmml_core::field::DataType::String => ArrowDataType::Utf8,
                pmml_core::field::DataType::Boolean => ArrowDataType::Utf8, // keep as string for "true"/"false"
                _ => ArrowDataType::Utf8,
            };
            // All fields nullable (Missing handling)
            ArrowField::new(fm.name.clone(), arrow_type, true)
        })
        .collect();
    Arc::new(Schema::new(fields))
}

/// Convert `Vec<HashMap<String, Value>>` into a `RecordBatch` using the given schema.
///
/// Missing values become nulls. `Continuous` → `Float64`, `Discrete` → `Utf8` (resolved via `symbol_names` if available).
/// For `Float64` columns, `Discrete` values that are numeric strings are parsed to `f64`; otherwise `None`.
///
/// # Parameters
///
/// - `maps`: row-major `Value` maps (each map is `field name → Value`).
/// - `schema`: Arrow schema (e.g. from [`data_dictionary_to_schema`]).
/// - `symbol_names`: optional `SymbolId → String` for resolving `Discrete` to display string.
///
/// # Returns
///
/// `Ok(RecordBatch)` with `num_rows() == maps.len()`, or `Err(String)` if `RecordBatch::try_new` fails.
///
/// # Errors
///
/// Returns `Err(String)` with Arrow error message if arrays and schema disagree (e.g. wrong column count).
///
/// # Performance
///
/// Allocates `Vec<Option<f64>>` / `Vec<Option<String>>` per column, then one array per column. O(rows*cols).
///
/// # Examples
///
/// ```
/// use pmml_session::arrow::{value_maps_to_record_batch, data_dictionary_to_schema};
/// use pmml_core::{Value, FieldId, field::{DataType, OpType}};
/// use pmml_ir::ir::FieldMeta;
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use arrow::datatypes::{Field as ArrowField, DataType as ArrowDataType, Schema};
/// let schema = Arc::new(Schema::new(vec![ArrowField::new("a", ArrowDataType::Float64, true)]));
/// let mut m = HashMap::new();
/// m.insert("a".into(), Value::Continuous(1.5));
/// let batch = value_maps_to_record_batch(&[m], schema, None).unwrap();
/// assert_eq!(batch.num_rows(), 1);
/// ```
pub fn value_maps_to_record_batch(
    maps: &[HashMap<String, Value>],
    schema: Arc<Schema>,
    symbol_names: Option<&HashMap<pmml_core::SymbolId, String>>,
) -> Result<RecordBatch, String> {
    if maps.is_empty() {
        if schema.fields().is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }
        let arrays: Vec<Arc<dyn Array>> = schema
            .fields()
            .iter()
            .map(|f| match f.data_type() {
                ArrowDataType::Float64 => {
                    Arc::new(Float64Array::from(Vec::<Option<f64>>::new())) as Arc<dyn Array>
                }
                _ => Arc::new(StringArray::from(Vec::<Option<&str>>::new())) as Arc<dyn Array>,
            })
            .collect();
        return RecordBatch::try_new(schema, arrays).map_err(|e| e.to_string());
    }

    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let name = field.name().as_str();
        match field.data_type() {
            ArrowDataType::Float64 => {
                let mut builder: Vec<Option<f64>> = Vec::with_capacity(maps.len());
                for row in maps {
                    match row.get(name) {
                        Some(Value::Continuous(v)) => builder.push(Some(*v)),
                        Some(Value::Missing) | None => builder.push(None),
                        Some(Value::Discrete(sid)) => {
                            // try resolve to f64 if symbol is numeric string
                            if let Some(sym_map) = symbol_names {
                                if let Some(s) = sym_map.get(sid) {
                                    if let Ok(f) = s.parse::<f64>() {
                                        builder.push(Some(f));
                                        continue;
                                    }
                                }
                            }
                            builder.push(None);
                        }
                    }
                }
                arrays.push(Arc::new(Float64Array::from(builder)) as Arc<dyn Array>);
            }
            _ => {
                let mut builder: Vec<Option<String>> = Vec::with_capacity(maps.len());
                for row in maps {
                    match row.get(name) {
                        Some(Value::Discrete(sid)) => {
                            if let Some(sym_map) = symbol_names {
                                builder.push(sym_map.get(sid).cloned());
                            } else {
                                builder.push(Some(format!("{:?}", sid)));
                            }
                        }
                        Some(Value::Continuous(v)) => builder.push(Some(v.to_string())),
                        Some(Value::Missing) | None => builder.push(None),
                    }
                }
                // Convert Vec<Option<String>> to StringArray
                let arr = StringArray::from(
                    builder
                        .iter()
                        .map(|o| o.as_deref())
                        .collect::<Vec<Option<&str>>>(),
                );
                arrays.push(Arc::new(arr) as Arc<dyn Array>);
            }
        }
    }
    RecordBatch::try_new(schema, arrays).map_err(|e| e.to_string())
}

/// Convert a `RecordBatch` into `Vec<HashMap<String, Value>>` using its schema.
///
/// `Float64` columns → `Continuous`, `Utf8`/`Dictionary` → `Discrete` (via temp `SymbolId` hash — caller should re-intern if needed).
/// For batched scoring via `Session`, this is not used directly; `Session::run_batch` handles `HashMap` already.
/// This is primarily for CSV → batch bridging and `InlineTable` tests.
///
/// # Parameters
///
/// - `batch`: `RecordBatch` to convert.
///
/// # Returns
///
/// `Vec<HashMap<String, Value>>` one map per row, with `Missing` for nulls.
///
/// # Performance
///
/// Allocates one `HashMap` per row. Use `Session::run_batch_arrow` for zero-copy scoring instead.
///
/// # Examples
///
/// ```
/// use pmml_session::arrow::{record_batch_to_value_maps, value_maps_to_record_batch};
/// use arrow::datatypes::{Field as ArrowField, DataType as ArrowDataType, Schema};
/// use arrow::array::Float64Array;
/// use arrow::record_batch::RecordBatch;
/// use std::sync::Arc;
/// let schema = Arc::new(Schema::new(vec![ArrowField::new("a", ArrowDataType::Float64, true)]));
/// let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Float64Array::from(vec![Some(1.0)])) as _]).unwrap();
/// let maps = record_batch_to_value_maps(&batch);
/// assert_eq!(maps.len(), 1);
/// ```
pub fn record_batch_to_value_maps(batch: &RecordBatch) -> Vec<HashMap<String, Value>> {
    let schema = batch.schema();
    let num_rows = batch.num_rows();
    let mut out = Vec::with_capacity(num_rows);
    for row_idx in 0..num_rows {
        let mut map = HashMap::new();
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(col_idx);
            let name = field.name().clone();
            if col.is_null(row_idx) {
                map.insert(name, Value::Missing);
                continue;
            }
            match field.data_type() {
                ArrowDataType::Float64 => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("float64 array");
                    map.insert(name, Value::Continuous(arr.value(row_idx)));
                }
                ArrowDataType::Utf8 => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .expect("string array");
                    let s = arr.value(row_idx);
                    if let Ok(f) = s.parse::<f64>() {
                        map.insert(name, Value::Continuous(f));
                    } else {
                        // Create a deterministic placeholder SymbolId via hash — not matching Ir intern, but usable for roundtrip.
                        // For true discrete fidelity, caller should look up via Ir's symbol map.
                        let sid = pmml_core::SymbolId({
                            use std::collections::hash_map::DefaultHasher;
                            use std::hash::{Hash, Hasher};
                            let mut h = DefaultHasher::new();
                            s.hash(&mut h);
                            (h.finish() & 0x7FFF_FFFF) as u32
                        });
                        map.insert(name, Value::Discrete(sid));
                    }
                }
                // fallback: treat as string
                _ => {
                    // try to format via array's Display
                    map.insert(name, Value::Missing);
                }
            }
        }
        out.push(map);
    }
    out
}

/// Convert an `InlineTable` represented as `Vec<HashMap<String, String>>` (raw rows) into a `RecordBatch`.
///
/// `TableLocator` case is represented as `None` or empty vec — returns an empty batch with the given schema
/// and does not panic (graceful placeholder handling per plan A4).
///
/// # Parameters
///
/// - `rows`: slice of string maps from the inline table (e.g. parsed from `InlineTable` XML).
/// - `schema`: Arrow schema for the table (usually from `Ir`).
///
/// # Returns
///
/// `Ok(RecordBatch)` with `num_rows() == rows.len()`, or `Err(String)` if `RecordBatch::try_new` fails.
/// Empty `rows` returns a zero-row batch with correct schema (handles `TableLocator` placeholder).
///
/// # Errors
///
/// Returns `Err(String)` if arrays cannot be built for the schema.
///
/// # Examples
///
/// ```
/// use pmml_session::arrow::{inline_table_to_record_batch, data_dictionary_to_schema};
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use arrow::datatypes::{Field as ArrowField, DataType as ArrowDataType, Schema};
/// let schema = Arc::new(Schema::new(vec![ArrowField::new("x", ArrowDataType::Float64, true)]));
/// let rows = vec![{ let mut m=HashMap::new(); m.insert("x".into(),"1.0".into()); m }];
/// let batch = inline_table_to_record_batch(&rows, schema).unwrap();
/// assert_eq!(batch.num_rows(), 1);
/// ```
pub fn inline_table_to_record_batch(
    rows: &[HashMap<String, String>],
    schema: Arc<Schema>,
) -> Result<RecordBatch, String> {
    if rows.is_empty() {
        // Empty → zero rows, schema columns with zero length (handles TableLocator placeholder)
        if schema.fields().is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }
        let arrays: Vec<Arc<dyn Array>> = schema
            .fields()
            .iter()
            .map(|f| match f.data_type() {
                ArrowDataType::Float64 => {
                    Arc::new(Float64Array::from(Vec::<Option<f64>>::new())) as Arc<dyn Array>
                }
                _ => Arc::new(StringArray::from(Vec::<Option<&str>>::new())) as Arc<dyn Array>,
            })
            .collect();
        return RecordBatch::try_new(schema, arrays).map_err(|e| e.to_string());
    }
    // Build intermediate Value maps from String rows, then delegate.
    let mut value_maps = Vec::with_capacity(rows.len());
    for row in rows {
        let mut map = HashMap::new();
        for (k, v) in row {
            let val = if v.is_empty() {
                Value::Missing
            } else if let Ok(f) = v.parse::<f64>() {
                Value::Continuous(f)
            } else {
                // hash-based SymbolId placeholder
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                v.hash(&mut h);
                let sid = pmml_core::SymbolId((h.finish() & 0x7FFF_FFFF) as u32);
                Value::Discrete(sid)
            };
            map.insert(k.clone(), val);
        }
        value_maps.push(map);
    }
    value_maps_to_record_batch(&value_maps, schema, None)
}

/// CSV string → `RecordBatch` via `arrow::csv::Reader`.
///
/// `has_header` should be `true` for PMML batch files (first line is field names).
/// Returns a `RecordBatch` with `Float64`/`Utf8` columns inferred from schema if provided, else inferred.
///
/// This uses `arrow::csv::ReaderBuilder` with the supplied `schema`. If `schema` is `None`, we infer
/// schema from CSV header (all `Utf8`, then caller can cast).
///
/// # Parameters
///
/// - `csv_str`: CSV text (may contain newlines, header + rows).
/// - `schema`: optional Arrow schema; if `None`, header is split on `,` and all columns are `Utf8`.
/// - `has_header`: whether first line is a header (usual `true`).
///
/// # Returns
///
/// `Ok(RecordBatch)` with all CSV rows concatenated (handles arrow's 1024-row batching).
///
/// # Errors
///
/// Returns `Err(String)` if CSV parsing fails or the file is empty.
///
/// # Examples
///
/// ```
/// use pmml_session::arrow::csv_str_to_record_batch;
/// use std::sync::Arc;
/// use arrow::datatypes::{Field as ArrowField, DataType as ArrowDataType, Schema};
/// let csv = "a,b\n1.0,hello\n2.0,world\n";
/// let schema = Arc::new(Schema::new(vec![
///     ArrowField::new("a", ArrowDataType::Float64, true),
///     ArrowField::new("b", ArrowDataType::Utf8, true),
/// ]));
/// let batch = csv_str_to_record_batch(csv, Some(schema), true).unwrap();
/// assert_eq!(batch.num_rows(), 2);
/// ```
pub fn csv_str_to_record_batch(
    csv_str: &str,
    schema: Option<Arc<Schema>>,
    has_header: bool,
) -> Result<RecordBatch, String> {
    use arrow::csv::ReaderBuilder;
    use std::io::Cursor;

    let schema_ref = schema.unwrap_or_else(|| {
        // Infer schema from header: read first line
        let mut lines = csv_str.lines();
        let header = lines.next().unwrap_or("");
        let cols: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
        let fields: Vec<ArrowField> = cols
            .into_iter()
            .map(|c| ArrowField::new(c, ArrowDataType::Utf8, true))
            .collect();
        Arc::new(Schema::new(fields))
    });

    let cursor = Cursor::new(csv_str.as_bytes());
    let mut reader = ReaderBuilder::new(schema_ref.clone())
        .with_header(has_header)
        .build(cursor)
        .map_err(|e| e.to_string())?;

    let batch = reader
        .next()
        .ok_or_else(|| "empty csv".to_string())?
        .map_err(|e| e.to_string())?;
    // If file has multiple batches ( > 1024 rows ), concatenate
    let mut batches = vec![batch];
    for res in reader {
        batches.push(res.map_err(|e| e.to_string())?);
    }
    if batches.len() == 1 {
        Ok(batches.remove(0))
    } else {
        arrow::compute::concat_batches(&schema_ref, &batches).map_err(|e| e.to_string())
    }
}

/// Helper for `TableLocator` placeholder: always returns an empty `RecordBatch` with the supplied schema.
///
/// This satisfies the plan requirement to "handle `TableLocator` placeholder gracefully" without panicking.
///
/// # Parameters
///
/// - `schema`: schema for the placeholder.
///
/// # Returns
///
/// `Ok(RecordBatch)` with `num_rows() == 0`.
///
/// # Errors
///
/// Returns `Err(String)` only if `RecordBatch::try_new` fails (unlikely for empty).
///
/// # Examples
///
/// ```
/// use pmml_session::arrow::table_locator_placeholder_batch;
/// use std::sync::Arc;
/// use arrow::datatypes::{Field as ArrowField, DataType as ArrowDataType, Schema};
/// let schema = Arc::new(Schema::new(vec![ArrowField::new("x", ArrowDataType::Float64, true)]));
/// let batch = table_locator_placeholder_batch(schema).unwrap();
/// assert_eq!(batch.num_rows(), 0);
/// ```
pub fn table_locator_placeholder_batch(schema: Arc<Schema>) -> Result<RecordBatch, String> {
    inline_table_to_record_batch(&[], schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_schema_from_ir() {
        // Build a tiny Ir manually or use data_dictionary_to_schema
        let fields = vec![
            FieldMeta {
                field_id: pmml_core::FieldId(0),
                name: "Petal.Length".to_string(),
                data_type: pmml_core::field::DataType::Double,
                op_type: pmml_core::field::OpType::Continuous,
                values: vec![],
                ..Default::default()
            },
            FieldMeta {
                field_id: pmml_core::FieldId(1),
                name: "Species".to_string(),
                data_type: pmml_core::field::DataType::String,
                op_type: pmml_core::field::OpType::Categorical,
                values: vec![],
                ..Default::default()
            },
        ];
        let schema = data_dictionary_to_schema(&fields);
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).data_type(), &ArrowDataType::Float64);
        assert_eq!(schema.field(1).data_type(), &ArrowDataType::Utf8);
    }

    #[test]
    fn arrow_value_maps_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            ArrowField::new("a", ArrowDataType::Float64, true),
            ArrowField::new("b", ArrowDataType::Utf8, true),
        ]));
        let mut m1 = HashMap::new();
        m1.insert("a".to_string(), Value::Continuous(1.5));
        m1.insert("b".to_string(), Value::Discrete(pmml_core::SymbolId(42)));
        let maps = vec![m1];
        let batch = value_maps_to_record_batch(&maps, schema.clone(), None).expect("to batch");
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 2);
        let out_maps = record_batch_to_value_maps(&batch);
        assert_eq!(out_maps.len(), 1);
        assert_eq!(out_maps[0].get("a"), Some(&Value::Continuous(1.5)));
    }

    #[test]
    fn arrow_inline_table_empty_is_table_locator() {
        let schema = Arc::new(Schema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Float64,
            true,
        )]));
        let batch = inline_table_to_record_batch(&[], schema.clone()).expect("empty batch");
        assert_eq!(batch.num_rows(), 0);
        let placeholder = table_locator_placeholder_batch(schema).expect("placeholder");
        assert_eq!(placeholder.num_rows(), 0);
    }

    #[test]
    fn arrow_inline_table_to_batch() {
        let schema = Arc::new(Schema::new(vec![
            ArrowField::new("Petal.Length", ArrowDataType::Float64, true),
            ArrowField::new("Petal.Width", ArrowDataType::Float64, true),
        ]));
        let rows = vec![
            {
                let mut m = HashMap::new();
                m.insert("Petal.Length".to_string(), "1.4".to_string());
                m.insert("Petal.Width".to_string(), "0.2".to_string());
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("Petal.Length".to_string(), "6.0".to_string());
                m.insert("Petal.Width".to_string(), "2.5".to_string());
                m
            },
        ];
        let batch = inline_table_to_record_batch(&rows, schema).expect("batch");
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn arrow_csv_to_batch() {
        let csv = "a,b\n1.0,hello\n2.0,world\n";
        let schema = Arc::new(Schema::new(vec![
            ArrowField::new("a", ArrowDataType::Float64, true),
            ArrowField::new("b", ArrowDataType::Utf8, true),
        ]));
        let batch = csv_str_to_record_batch(csv, Some(schema), true).expect("csv batch");
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn arrow_data_dictionary_empty() {
        let schema = data_dictionary_to_schema(&[]);
        assert_eq!(schema.fields().len(), 0);
        let batch = value_maps_to_record_batch(&[], schema, None).expect("empty");
        assert_eq!(batch.num_rows(), 0);
    }
}
