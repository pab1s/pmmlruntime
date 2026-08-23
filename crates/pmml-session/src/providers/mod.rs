pub mod cpu_batched;
pub mod cpu_serial;

use pmml_core::{Result, Value};
use pmml_ir::ir::Ir;

/// ExecutionProvider trait — ONNX EP analogy.
pub trait ExecutionProvider: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, ir: &Ir, values: &mut [Value]) -> Result<Value>;
}

pub use cpu_batched::CpuBatchedProvider;
pub use cpu_serial::CpuSerialProvider;
