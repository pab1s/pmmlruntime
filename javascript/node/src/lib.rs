//! pmmlruntime-node — NAPI InferenceSession over `pmmlruntime::Session`.
//!
//! v1 scope: `Continuous` doubles only (`Petal.Length` / `Petal.Width` for
//! `bench/pmml/DecisionTreeIris.pmml`). `Discrete` inputs via
//! `Session::string_to_value` are a follow-up.
//!
//! Ownership: `PmmlEnv` is kept inside the struct (`_env`) so the `Session`'s
//! `Arc<Ir>` stays alive for the addon's lifetime (same pattern as
//! `python/_native` and `java/native`).

use napi_derive::napi;
use std::collections::HashMap;

use pmmlruntime::base::{SymbolId, Value};
use pmmlruntime::session::batch::Batch;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};

#[napi]
pub fn hello() -> String {
    "pmml-runtime".to_string()
}

#[napi]
pub struct InferenceSession {
    sess: Session,
    _env: PmmlEnv,
}

#[napi]
impl InferenceSession {
    #[napi(constructor)]
    pub fn new(path: String) -> napi::Result<Self> {
        let env = PmmlEnv::new();
        let sess = Session::from_file(&env, &path, SessionOptions::default())
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { sess, _env: env })
    }

    #[napi]
    pub fn run(&self, input: HashMap<String, f64>) -> napi::Result<HashMap<String, String>> {
        let map: HashMap<String, Value> = input
            .into_iter()
            .map(|(k, v)| (k, Value::Continuous(v)))
            .collect();
        let row = self
            .sess
            .run(&map as &dyn Batch)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?
            .into_single()
            .ok_or_else(|| napi::Error::from_reason("run: empty result"))?;
        Ok(row
            .into_iter()
            .map(|(k, v)| {
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
                (k, s)
            })
            .collect())
    }
}
