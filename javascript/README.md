# javascript — Node (NAPI) + Web (WASM) bindings

Like `onnxruntime/js/` — both over `include/pmml_runtime.h PmmlApi`. Now scaffolded.

* `node/` NAPI `optionalDependencies` per arch, `web/` wasm-pack, both thin shims (no scoring logic)
* API: `await InferenceSession.create("model.pmml")` -> `session.inputNames` -> `await session.run({...})` (ORT parity)
