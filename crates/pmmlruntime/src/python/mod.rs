//! Python bindings — `PyO3` 0.22 `extension-module`, like `ort.py`.
//!
//! This crate is the Python extension for `pmmlruntime` (`import pmml_runtime`).
//! In v1 it is a stub that reserves the crate name and exposes a trivial `hello()` for smoke tests.
//! In v2 it will expose `#[pyclass] InferenceSession` wrapping `crate::session::Session` with
//! `new(path)`, `run(dict)`, and `run_batch(list[dict])` plus `__repr__`.
//!
//! # Feature flags
//!
//! - `python` — enables `pyo3` 0.22 with `extension-module`. Without the feature, `pyo3` is not linked
//!   and the crate is a no-op (so `cargo test --workspace` doesn't need `libpython`).
//!
//! # What belongs here
//!
//! - `python_impl` module (`#[cfg(feature = "python")]`) with `#[pyfunction] hello` and `#[pymodule] pmml_runtime`.
//! - Future `InferenceSession` pyclass (v2) that owns `Session` and `PmmlEnv`.
//!
//! # Thread safety
//!
//! `PyO3` requires `Send` for `#[pyclass]`. `crate::session::Session` is already `Send+Sync`, so `InferenceSession` will be too.
//! GIL handling will use `py.allow_threads(|| sess.run(...))` to avoid blocking Python.
//!
//! # Examples
//!
//! ```python
//! # Python (not Rust):
//! # import pmml_runtime
//! # print(pmml_runtime.hello())  # -> "pmml-runtime v1"
//! ```
//!
//! # Rust example (placeholder)
//!
//! ```
//! // This crate currently has only placeholder()
//! assert_eq!(pmmlruntime::python::placeholder(), ());
//! ```

// In v2 this will be:
// use pyo3::prelude::*;
// #[pyclass] struct InferenceSession { inner: crate::session::Session }
// #[pymethods] impl InferenceSession { fn new(path: &str) -> ...; fn run(...) }

/// Placeholder to keep crate non-empty when `python` feature is off.
///
/// Always available so `cargo doc` / `cargo test` without `python` feature still link.
pub fn placeholder() {}

/// Python module implementation gated on `python` feature.
///
/// Exposes `pyo3` 0.22 `extension-module` with `#[pymodule] pmml_runtime`.
/// When the feature is off this module is not compiled, so `libpython` is not required.
#[cfg(feature = "python")]
mod python_impl {
    use pyo3::prelude::*;
    /// Simple hello function for smoke tests (`pmml_runtime.hello() -> str`).
    ///
    /// # Returns
    ///
    /// `"pmml-runtime v1"` string.
    ///
    /// # Examples
    ///
    /// ```python
    /// # import pmml_runtime
    /// # assert pmml_runtime.hello() == "pmml-runtime v1"
    /// ```
    #[pyfunction]
    fn hello() -> PyResult<String> {
        Ok("pmml-runtime v1".into())
    }
    /// Python module `pmml_runtime` (extension-module).
    ///
    /// Registers `hello` and, in v2, `InferenceSession`. The `m` `Bound<'_, PyModule>` is where
    /// functions/classes are added. Returns `Ok(())` on success, `PyErr` on registration failure.
    #[pymodule]
    fn pmml_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(hello, m)?)?;
        Ok(())
    }
}
