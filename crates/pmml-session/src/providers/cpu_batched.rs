use super::ExecutionProvider;
use pmml_core::{Result, Value};
use pmml_ir::ir::Ir;

/// CpuBatched — stub for v1, prepared for Rayon + Arrow columnar.
// In v2 this will implement `evaluate_batch(RecordBatch) -> RecordBatch` via `rayon::par_iter` chunks.
pub struct CpuBatchedProvider;

impl ExecutionProvider for CpuBatchedProvider {
    fn name(&self) -> &str {
        "CPU_BATCHED"
    }

    fn evaluate(&self, _ir: &Ir, _values: &mut [Value]) -> Result<Value> {
        Err(pmml_core::error::PmmlError::UnsupportedMarkup(
            "CpuBatchedProvider not yet implemented — use CpuSerial in v1".into(),
        ))
    }
}
