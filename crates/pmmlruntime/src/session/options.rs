//! `SessionOptions` — configuration for `Session` creation (graph optimization).

/// Graph optimization level.
///
/// Controls how much the IR is optimized before execution.
///
/// # Variants
///
/// - `DisableAll` — disable all graph opts (interpreter path, minimal cold overhead).
/// - `EnableBasic` — basic opts (bytecode) — default.
/// - `EnableExtended` — extended opts (SIMD, batch) — currently no-op, reserved for future use.
/// - `EnableAll` — all opts (includes future JIT) — currently no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GraphOptimizationLevel {
    /// Disable all (interpreter, cold fastest).
    DisableAll = 0,
    /// Basic (bytecode) — default.
    #[default]
    EnableBasic = 1,
    /// Extended (SIMD, batch) — currently no-op.
    EnableExtended = 2,
    /// All (includes future JIT) — currently no-op.
    EnableAll = 3,
}

/// Session options builder.
///
/// All fields have defaults so `SessionOptions::default()` is the common entry point.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::{SessionOptions, GraphOptimizationLevel};
/// let opts = SessionOptions::default()
///     .graph_optimization_level(GraphOptimizationLevel::EnableBasic);
/// assert_eq!(opts.graph_optimization_level, GraphOptimizationLevel::EnableBasic);
/// ```
#[derive(Clone, Debug)]
pub struct SessionOptions {
    /// Graph optimization level (default `EnableBasic`).
    pub graph_optimization_level: GraphOptimizationLevel,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            graph_optimization_level: GraphOptimizationLevel::EnableBasic,
        }
    }
}

impl SessionOptions {
    /// Create default options (`EnableBasic`).
    ///
    /// Equivalent to `SessionOptions::default()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::SessionOptions;
    /// let opts = SessionOptions::new();
    /// assert_eq!(opts.graph_optimization_level, pmmlruntime::session::GraphOptimizationLevel::EnableBasic);
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
}
