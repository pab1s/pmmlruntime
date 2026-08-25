//! `SessionOptions` — like `OrtSessionOptions` (graph opt level, threads, EP).
//!
//! Builder for `Session::from_bytes` / `from_file`. All fields have defaults so `SessionOptions::default()`
//! is the common entry point. The builder consumes `self` and returns `Self` (fluent style).

/// ONNX-style graph optimization level.
///
/// Mirrors `GraphOptimizationLevel` in ORT. In v1 only `DisableAll` vs `EnableBasic` differ
/// (bytecode vs interpreter); `EnableExtended`/`EnableAll` are stubs for SIMD/JIT in v2.
///
/// # Variants
///
/// - `DisableAll` — disable all graph opts (interpreter, cold path fastest for tiny models).
/// - `EnableBasic` — basic opts (bytecode) — default in v1.
/// - `EnableExtended` — extended (SIMD, batch) — stub in v1, active in v2.
/// - `EnableAll` — all opts (includes JIT) — stub.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GraphOptimizationLevel {
    /// Disable all (interpreter, cold fastest).
    DisableAll = 0,
    /// Basic (bytecode) — default v1.
    #[default]
    EnableBasic = 1,
    /// Extended (SIMD, batch) — stub v1, active v2.
    EnableExtended = 2,
    /// All (includes JIT) — stub.
    EnableAll = 3,
}

/// Execution provider kind.
///
/// Mirrors `OrtExecutionProvider` selection. `Session::from_ir` matches on this to
/// box the concrete provider.
///
/// # Variants
///
/// - `CpuSerial` — single-threaded (`CpuSerialProvider`), no `rayon`, best for single rows.
/// - `CpuBatched` — parallel (`CpuBatchedProvider`, `rayon` `par_chunks(256)`, fallback ` <256` serial).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExecutionProviderKind {
    /// Single-threaded provider (default).
    #[default]
    CpuSerial,
    /// Parallel provider (`rayon`) for batch `>256` rows.
    CpuBatched, // stub v1, ready for Rayon
}

/// Session options builder.
///
/// Use `SessionOptions::default().execution_provider(kind).intra_threads(n)` etc.
/// All fields are `Copy` so the builder consumes `self` and returns `Self` without cloning.
///
/// # Examples
///
/// ```
/// use pmml_session::{SessionOptions, ExecutionProviderKind, GraphOptimizationLevel};
/// let opts = SessionOptions::default()
///     .graph_optimization_level(GraphOptimizationLevel::EnableBasic)
///     .intra_threads(4)
///     .execution_provider(ExecutionProviderKind::CpuBatched);
/// assert_eq!(opts.execution_provider, ExecutionProviderKind::CpuBatched);
/// ```
#[derive(Clone, Debug)]
pub struct SessionOptions {
    /// Graph optimization level (default `EnableBasic`).
    pub graph_optimization_level: GraphOptimizationLevel,
    /// Intra-op thread count for `ExecutionProvider::eval_batch` (default `1`).
    pub intra_op_threads: usize,
    /// Chosen execution provider (default `CpuSerial`).
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
    /// Create default options (`EnableBasic`, `1` thread, `CpuSerial`).
    ///
    /// Equivalent to `SessionOptions::default()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmml_session::SessionOptions;
    /// let opts = SessionOptions::new();
    /// assert_eq!(opts.intra_op_threads, 1);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set graph optimization level.
    ///
    /// # Parameters
    ///
    /// - `lvl`: `GraphOptimizationLevel` variant.
    ///
    /// # Returns
    ///
    /// `Self` with updated `graph_optimization_level`.
    pub fn graph_optimization_level(mut self, lvl: GraphOptimizationLevel) -> Self {
        self.graph_optimization_level = lvl;
        self
    }

    /// Set intra-op thread count.
    ///
    /// # Parameters
    ///
    /// - `n`: number of threads for provider sharding (e.g. `rayon` pool). `0` is treated as `1` by `rayon`.
    ///
    /// # Returns
    ///
    /// `Self` with updated `intra_op_threads`.
    pub fn intra_threads(mut self, n: usize) -> Self {
        self.intra_op_threads = n;
        self
    }

    /// Set execution provider kind.
    ///
    /// # Parameters
    ///
    /// - `ep`: `CpuSerial` or `CpuBatched`.
    ///
    /// # Returns
    ///
    /// `Self` with updated `execution_provider`.
    pub fn execution_provider(mut self, ep: ExecutionProviderKind) -> Self {
        self.execution_provider = ep;
        self
    }
}
