# java — Java binding (JNI over `include/pmml_runtime.h`)

Like `onnxruntime/java/` — thin JNI shim over the C `PmmlApi` table, not a reimplementation.

```
java/
  pom.xml                         # com.pmmlruntime:pmmlruntime:0.1.0 -> Maven Central
  src/main/java/com/pmmlruntime/
    PmmlEnv.java                  # wraps PmmlEnv* (AutoCloseable, long handle)
    PmmlSession.java              # wraps PmmlSession* (AutoCloseable)
    NodeInfo.java                 # input/output metadata (name, DataType, OpType)
    PmmlException.java
    NativeLoader.java             # extracts libpmmlruntime.so from jar resources
  src/main/resources/native/<os>-<arch>/libpmmlruntime.so
  src/main/rust/                  # optional jni crate calling PmmlGetApi (or pure JNI C)
  src/test/java/com/pmmlruntime/InferenceTest.java
```

* Build: `cargo build -p pmmlruntime --features capi --release` produces `libpmmlruntime.so`; `mvn package` bundles it per arch inside the jar.
* API: `OrtEnvironment.getEnvironment().createSession("model.pmml", opts)` → `session.run(Map<String,PmmlValue>)` → `Result implements AutoCloseable` (like `ai.onnxruntime.OrtSession`).
* Arrow: `org.apache.arrow:arrow-vector` `VectorSchemaRoot` → C Data Interface → `PmmlApi.BindInputArrow`.

See `docs/BINDINGS.md` for Java quickstart. This folder is stubbed on `feat/c-binding`; real impl lands on `feat/java-binding`.
