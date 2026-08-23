//! Python bindings — PyO3, like ort.py
//! v1 stub: not yet exposing InferenceSession. Crate exists to reserve name and header.

// In v2 this will be:
// use pyo3::prelude::*;
// #[pyclass] struct InferenceSession { inner: pmml_session::Session }
// #[pymethods] impl InferenceSession { fn new(path: &str) -> ...; fn run(...) }

pub fn placeholder() {}

#[cfg(feature = "python")]
mod python_impl {
    use pyo3::prelude::*;
    #[pyfunction]
    fn hello() -> PyResult<String> {
        Ok("pmml-runtime v1".into())
    }
    #[pymodule]
    fn pmml_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(hello, m)?)?;
        Ok(())
    }
}
