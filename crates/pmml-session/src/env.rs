//! PmmlEnv — global environment, like OrtEnv (thread pool, logger).

use std::sync::Arc;

/// Global environment. Cheap to clone (Arc inner).
#[derive(Clone, Debug)]
pub struct PmmlEnv {
    inner: Arc<EnvInner>,
}

#[derive(Debug)]
struct EnvInner {
    // In v2 this will hold rayon::ThreadPool, logger, telemetry.
    // For v1, just a name.
    name: String,
}

impl PmmlEnv {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(EnvInner {
                name: "pmml-runtime".into(),
            }),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(EnvInner { name: name.into() }),
        }
    }
}

impl Default for PmmlEnv {
    fn default() -> Self {
        Self::new()
    }
}
