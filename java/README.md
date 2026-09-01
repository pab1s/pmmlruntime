# java — Java binding (JNI over `include/pmml_runtime.h`)

Like `onnxruntime/java/` — thin JNI shim over `PmmlApi`, not a reimplementation. Now scaffolded.

* Build: `cargo build -p pmmlruntime --features capi --release` → `libpmmlruntime.so`; `mvn package` bundles per arch.
* Run: `mvn test` (skips native until PmmlApi linked)
* Arrow: `arrow-vector` → C Data Interface → `PmmlApi.BindInputArrow` (planned)
