//! pmmlruntime-web — WASM InferenceSession over `pmmlruntime::Session`.
//!
//! `wasm-pack build --target web|bundler`
//!
//! v1 scope: `Continuous` doubles only (same as `javascript/node`). `Discrete`
//! inputs via `Session::string_to_value` are a follow-up. WASM has no
//! filesystem, so construction is from in-memory PMML bytes (`Uint8Array`);
//! `run` takes a JS `Object` of `name -> number` and returns an `Object` of
//! `name -> string` (Discrete labels resolved via `ir.symbol_names`).

use std::collections::HashMap;

use pmmlruntime::base::{SymbolId, Value};
use pmmlruntime::session::batch::Batch;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn hello() -> String {
    "pmml-runtime".to_string()
}

#[wasm_bindgen]
pub struct InferenceSession {
    sess: Session,
    _env: PmmlEnv,
}

#[wasm_bindgen]
impl InferenceSession {
    /// Create a session from PMML XML bytes (`Uint8Array`).
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<InferenceSession, JsValue> {
        let env = PmmlEnv::new();
        let sess = Session::from_bytes(&env, bytes, SessionOptions::default())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { sess, _env: env })
    }

    /// Score one row: `input` is `{ "Petal.Length": 1.4, ... }` (numbers only).
    ///
    /// Non-numeric entries are skipped (treated as `Missing`). Returns
    /// `{ predictedValue: "setosa", ... }` with Discrete labels resolved.
    pub fn run(&self, input: &js_sys::Object) -> Result<js_sys::Object, JsValue> {
        let mut map: HashMap<String, Value> = HashMap::new();
        for key in js_sys::Object::keys(input).iter() {
            let name = key.as_string().ok_or_else(|| JsValue::from_str("run: non-string key"))?;
            let v = js_sys::Reflect::get(input, &key).map_err(|_| JsValue::from_str("run: Reflect::get failed"))?;
            if let Some(f) = v.as_f64() {
                map.insert(name, Value::Continuous(f));
            }
            // else: skip — Missing (v1 Continuous-only, no string_to_value yet)
        }
        let row = self
            .sess
            .run(&map as &dyn Batch)
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .into_single()
            .ok_or_else(|| JsValue::from_str("run: empty result"))?;
        let out = js_sys::Object::new();
        for (k, v) in row {
            let s = match v {
                Value::Continuous(f) => {
                    if f.fract() == 0.0 {
                        format!("{}", f as i64)
                    } else {
                        format!("{f}")
                    }
                }
                Value::Discrete(SymbolId(id)) => self
                    .sess
                    .ir
                    .symbol_names
                    .get(&SymbolId(id))
                    .cloned()
                    .unwrap_or_else(|| format!("Symbol({id})")),
                Value::Missing => "Missing".to_string(),
            };
            js_sys::Reflect::set(&out, &JsValue::from_str(&k), &JsValue::from_str(&s))
                .map_err(|_| JsValue::from_str("run: Reflect::set failed"))?;
        }
        Ok(out)
    }
}
