//! `PmmlEnv` — global environment, like `OrtEnv` (thread pool, logger).
//!
//! `PmmlEnv` owns the `Arc<EnvInner>` so it is cheap to `Clone` and `Send+Sync`.
//! In v1 it only carries a name for diagnostics; in v2 it will hold the `rayon::ThreadPool`
//! and logger/telemetry handles. `Session` keeps an `Arc` clone of the env so dropping
//! the caller's `PmmlEnv` does not invalidate existing sessions.

use std::sync::Arc;

/// Global environment. Cheap to clone (`Arc` inner).
///
/// Mirrors `OrtEnv` / `OrtApi::CreateEnv`. It is `Send` + `Sync` and `Clone` is a single
/// atomic increment. `Session::from_bytes` takes `&PmmlEnv` and clones it into the session.
///
/// # Examples
///
/// ```
/// use pmmlruntime::session::PmmlEnv;
/// let env = PmmlEnv::new();
/// let env2 = env.clone(); // cheap
/// assert_eq!(env.name(), env2.name());
/// ```
#[derive(Clone, Debug)]
pub struct PmmlEnv {
    inner: Arc<EnvInner>,
}

#[derive(Debug)]
struct EnvInner {
    // In v2 this will hold rayon::ThreadPool, logger, telemetry.
    // For v1, just a name (kept for OrtEnv parity; read via `name()` to avoid dead_code warnings).
    name: String,
}

impl PmmlEnv {
    /// Create a new environment with the default name `"pmml-runtime"`.
    ///
    /// # Returns
    ///
    /// `PmmlEnv` with `name() == "pmml-runtime"` and a fresh `Arc` inner.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::PmmlEnv;
    /// let env = PmmlEnv::new();
    /// assert_eq!(env.name(), "pmml-runtime");
    /// ```
    pub fn new() -> Self {
        Self {
            inner: Arc::new(EnvInner {
                name: "pmml-runtime".into(),
            }),
        }
    }

    /// Create an environment with a custom name (for diagnostics / telemetry).
    ///
    /// # Parameters
    ///
    /// - `name`: environment name (e.g. `"my-service"`). Converted via `Into<String>`.
    ///
    /// # Returns
    ///
    /// `PmmlEnv` with `name() == name`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::PmmlEnv;
    /// let env = PmmlEnv::with_name("test-env");
    /// assert_eq!(env.name(), "test-env");
    /// ```
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(EnvInner { name: name.into() }),
        }
    }

    /// Name of the environment (for diagnostics).
    ///
    /// Reads `inner.name` to keep the field live. Used by CLI / logs.
    ///
    /// # Returns
    ///
    /// `&str` slice of the inner name.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::session::PmmlEnv;
    /// let env = PmmlEnv::with_name("demo");
    /// assert_eq!(env.name(), "demo");
    /// ```
    pub fn name(&self) -> &str {
        &self.inner.name
    }
}

impl Default for PmmlEnv {
    /// `PmmlEnv::new()` — `name() == "pmml-runtime"`.
    fn default() -> Self {
        Self::new()
    }
}
