//! Execution providers — ONNX `IExecutionProvider` analogy for PMML scoring.
//!
//! Providers own batch sharding and per-row evaluation. `Session` only materializes
//! `Value[FieldId]` via `with_value_buffer` and maps outputs; providers do `DerivedFields`
//! + model dispatch.
//!
//! This separation lets `CpuSerial` (simple, debuggable) and `CpuBatched` (`rayon` sharding)
//! share logic without duplicating `Session` output mapping.

pub mod cpu_batched;
pub mod cpu_serial;

use crate::batch::{Batch, BatchCtx, BatchResult};
use pmml_core::{Result, Value};
use pmml_ir::ir::Ir;

/// Execution provider trait — mirrors `IExecutionProvider` in ONNX Runtime.
///
/// Providers are `Send + Sync` so a single `Session` can be shared across threads.
/// `Session` builds a [`BatchCtx`] (no per-row allocation) and delegates to `eval_batch`;
/// for tiny batches (`<256` rows) even `CpuBatched` falls back to serial to avoid `rayon` overhead.
///
/// # Contract for implementors
///
/// - `eval_row` must handle `DerivedFields` and model dispatch for a single `&mut [Value]`.
/// - `eval_batch` must shard by `Batch::len()` and return `BatchResult::Rows` preserving order.
/// - `preferred_format` is a hint (`Columnar` for Arrow, `RowMajor` for `HashMap`). `Session` may still
///   pass either format; provider must handle both via `Batch::materialize_row`.
/// - `evaluate` is a backward-compat alias for `eval_row`.
///
/// # Concurrency
///
/// `eval_row` is called with a thread-local `&mut [Value]` slice, so it must not retain
/// references beyond the call. `eval_batch` may use `rayon` internally; it must not hold
/// `&Ir` mutably and must respect `BatchCtx` being `Send+Sync`.
///
/// # Examples
///
/// ```
/// use pmml_session::providers::{ExecutionProvider, CpuSerialProvider};
/// use pmml_core::Value;
/// use pmml_ir::ir::{Ir, ModelIr, TreeIr, MiningSchemaIr, MissingValueStrategy, NoTrueChildStrategy};
/// use std::collections::HashMap;
///
/// let provider = CpuSerialProvider::new();
/// assert_eq!(provider.name(), "CPU");
/// ```
pub trait ExecutionProvider: Send + Sync {
    /// Provider name (e.g. `"CPU"` or `"CPU_BATCHED"`), for diagnostics / telemetry.
    fn name(&self) -> &str;
    /// Evaluate a single row's `values[FieldId]` → predicted [`Value`].
    ///
    /// Handles `DerivedFields` + model dispatch (`Tree`, `Regression`, `Mining`, etc.).
    ///
    /// # Parameters
    ///
    /// - `ir`: immutable model reference (provider must not mutate it).
    /// - `values`: `&mut [Value]` indexed by `FieldId.as_usize()`, already materialized from `BatchCtx`.
    ///
    /// # Returns
    ///
    /// Predicted [`Value`] (`Continuous`, `Discrete(SymbolId)` or `Missing`).
    ///
    /// # Errors
    ///
    /// Returns [`pmml_core::PmmlError::InvalidValue`] if `eval_derived_fields` fails.
    fn eval_row(&self, ir: &Ir, values: &mut [Value]) -> Result<Value>;
    /// Evaluate a full `Batch` (row-major or columnar) → `BatchResult::Rows`.
    ///
    /// Default impl loops over `batch` via `eval_row`; `CpuBatched` overrides with `rayon`.
    ///
    /// # Parameters
    ///
    /// - `ir`: model reference.
    /// - `batch`: `&dyn Batch` (object-safe, `Send+Sync`), could be `Vec<HashMap>` or `RecordBatch`.
    /// - `ctx`: no-per-row-alloc context (refs to `name_to_id`, `Ir`, `output_fields` etc.).
    ///
    /// # Returns
    ///
    /// `BatchResult::Rows` with one output map per input row, preserving order.
    ///
    /// # Errors
    ///
    /// Propagates `PmmlError` from `materialize_row` or `eval_row`.
    fn eval_batch(&self, ir: &Ir, batch: &dyn Batch, ctx: &BatchCtx) -> Result<BatchResult>;
    /// Preferred batch layout for this provider (hint for `Session` to avoid conversion).
    ///
    /// `CpuSerial` and `CpuBatched` both return `Columnar` because Arrow zero-copy wins at large `n`,
    /// but they accept `RowMajor` as well. `Session` keeps `Batch` trait so callers aren't forced into Arrow.
    fn preferred_format(&self) -> crate::batch::BatchFormat {
        crate::batch::BatchFormat::Columnar
    }
    /// Backward compat: old `evaluate` name → `eval_row`.
    ///
    /// New code should call [`eval_row`](Self::eval_row). Kept for `Session::run` path.
    fn evaluate(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
        self.eval_row(ir, values)
    }
}

pub use cpu_batched::CpuBatchedProvider;
pub use cpu_serial::CpuSerialProvider;
