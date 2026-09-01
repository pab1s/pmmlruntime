//! pmmlruntime-web — WASM shim over PmmlApi (serial fallback, no rayon)
//! wasm-pack build --target web|bundler

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn hello() -> String {
    "pmml-runtime".to_string()
}

// #[wasm_bindgen] pub struct InferenceSession { handle: usize }
