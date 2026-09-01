# javascript — Node (NAPI) + Web (WASM) bindings

Like `onnxruntime/js/` (`js/node` + `js/web`) — both over `include/pmml_runtime.h`.

```
javascript/
  node/                           # napi-rs (like js/node)
    Cargo.toml                    # cdylib calling PmmlGetApi
    package.json                  # pmmlruntime-node, optionalDependencies per arch
  web/                            # wasm-pack (like js/web)
    Cargo.toml                    # wasm32-unknown-unknown cdylib
    package.json                  # pmmlruntime-web (pmml_bg.wasm)
  package.json                    # workspace root (optional)
```

```ts
import * as pm from "pmmlruntime-node";
const sess = await pm.InferenceSession.create("model.pmml");
sess.inputNames; // ["Petal.Length"]
await sess.run({ "Petal.Length": new pm.Tensor("float64", Float64Array.of(1.4), [1]) });
await sess.run({ "Petal.Length": 1.4 }); // plain dict sugar
```

* Node: `napi-rs` prebuilds per `linux-x64-gnu`/`darwin-arm64`/`win32-x64-msvc` like `onnxruntime-node`.
* Web: `wasm-pack` `wasm32-unknown-unknown`; `rayon` requires `SharedArrayBuffer` headers, fallback to serial otherwise.
* Stubbed on `feat/c-binding`; real impl on `feat/javascript-binding`.
