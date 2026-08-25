pub mod cpu_batched;
pub mod cpu_serial;

use crate::batch::{Batch, BatchCtx, BatchResult};
use pmml_core::{Result, Value};
use pmml_ir::ir::Ir;

/// ExecutionProvider trait — ONNX EP analogy.
///
/// Mirrors `IExecutionProvider` in ORT:
/// - `eval_row` is the per-row hot path (like `Execute` on a single node)
/// - `eval_batch` shards via `intra_op_threads` (rayon for `CpuBatched`, sequential for `CpuSerial`)
/// - `preferred_format` hints whether provider prefers `RowMajor` vs `Columnar`
pub trait ExecutionProvider: Send + Sync {
    fn name(&self) -> &str;
    /// Evaluate a single row's `values[FieldId]` → predicted `Value`.
    /// Handles `DerivedFields` + model dispatch (Tree, Regression, Mining, etc.).
    fn eval_row(&self, ir: &Ir, values: &mut [Value]) -> Result<Value>;
    /// Evaluate a full `Batch` (row-major or columnar) → `BatchResult::Rows`.
    /// Default impl loops over `batch` via `eval_row`; `CpuBatched` overrides with rayon.
    fn eval_batch(&self, ir: &Ir, batch: &dyn Batch, ctx: &BatchCtx) -> Result<BatchResult>;
    /// Preferred batch layout for this provider (hint for `Session` to avoid conversion).
    fn preferred_format(&self) -> crate::batch::BatchFormat {
        crate::batch::BatchFormat::Columnar
    }
    /// Backward compat: old `evaluate` name → `eval_row`.
    fn evaluate(&self, ir: &Ir, values: &mut [Value]) -> Result<Value> {
        self.eval_row(ir, values)
    }
}

pub use cpu_batched::CpuBatchedProvider;
pub use cpu_serial::CpuSerialProvider;
