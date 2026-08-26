//! `PmmlEnv` — global environment for runtime coordination (thread pool, logger).
//!
//! `PmmlEnv` owns the `Arc<EnvInner>` so it is cheap to `Clone` and `Send+Sync`.
//! Currently it carries a name for diagnostics; the same handle is cloned into each `Session`
//! so dropping the caller's `PmmlEnv` does not invalidate existing sessions.

use std::sync::Arc;

/// Global environment. Cheap to clone (`Arc` inner).
///
/// It is `Send` + `Sync` and `Clone` is a single atomic increment.
/// `Session::from_bytes` takes `&PmmlEnv` and clones it into the session.
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
    // Reserved for thread pool, logger, telemetry handles.
    // Currently just a name (kept live via `name()` to avoid dead_code warnings).
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
