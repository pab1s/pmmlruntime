//! SessionOptions — like OrtSessionOptions (graph opt level, threads, EP).

/// ONNX-style graph optimization level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GraphOptimizationLevel {
    /// Disable all (interpreter, cold fastest)
    DisableAll = 0,
    /// Basic (bytecode) — default v1
    #[default]
    EnableBasic = 1,
    /// Extended (SIMD, batch) — stub v1, active v2
    EnableExtended = 2,
    /// All (includes JIT) — stub
    EnableAll = 3,
}

/// Execution provider kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExecutionProviderKind {
    #[default]
    CpuSerial,
    CpuBatched, // stub v1, ready for Rayon
}

/// Session options builder.
#[derive(Clone, Debug)]
pub struct SessionOptions {
    pub graph_optimization_level: GraphOptimizationLevel,
    pub intra_op_threads: usize,
    pub execution_provider: ExecutionProviderKind,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            graph_optimization_level: GraphOptimizationLevel::EnableBasic,
            intra_op_threads: 1,
            execution_provider: ExecutionProviderKind::CpuSerial,
        }
    }
}

impl SessionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn graph_optimization_level(mut self, lvl: GraphOptimizationLevel) -> Self {
        self.graph_optimization_level = lvl;
        self
    }

    pub fn intra_threads(mut self, n: usize) -> Self {
        self.intra_op_threads = n;
        self
    }

    pub fn execution_provider(mut self, ep: ExecutionProviderKind) -> Self {
        self.execution_provider = ep;
        self
    }
}
