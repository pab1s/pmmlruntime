//! _native — pyo3 shim over include/pmml_runtime.h PmmlApi (ORT python pattern)
//!
//! Thin wrapper: Python holds PmmlSession* handle (long) obtained via PmmlGetApi().
//! Every run() does py.allow_threads(|| api->Run/RunArrow).
//! Arrow: pyarrow.Table._export_to_c -> ArrowArray/Schema -> PmmlApi.RunArrow.
//! This file is the scaffold — real PmmlGetApi linking happens after feat/c-binding's
//! Rust ffi implements the versioned table. Until then, hello() smoke test only.

use pyo3::prelude::*;

#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("pmml-runtime".into())
}

/// Placeholder InferenceSession — will hold PmmlSession* handle like OrtSession.
// #[pyclass]
// struct InferenceSession { handle: usize }  // *mut PmmlSession as usize

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    // m.add_class::<InferenceSession>()?;
    Ok(())
}
