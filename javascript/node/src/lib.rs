//! pmmlruntime-node — NAPI shim over include/pmml_runtime.h PmmlApi
//! Holds PmmlSession* handle like onnxruntime-node's native addon.
//! All calls go through PmmlGetApi() table.

use napi_derive::napi;

#[napi]
pub fn hello() -> String {
    "pmml-runtime".to_string()
}

// #[napi] struct InferenceSession { handle: usize }  // PmmlSession* as usize
