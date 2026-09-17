# java — Java binding

JNI shim over the `pmmlruntime` crate. The Java classes call a small native layer that holds
`PmmlEnv` and `PmmlSession` handles; scoring runs in the same Rust engine the crate exposes.

* Build the native library: `cargo build --manifest-path java/native/Cargo.toml --release`
* Run the tests: `mvn test` (NativeLoader finds the cargo output, no extra flags)
* Package: `mvn package` (adds the JNI library from `src/main/resources/native/<os>-<arch>/`
  when the release workflow has placed it there)
* API: `PmmlEnv.create()`, `PmmlSession.fromFile(env, path)`, `session.run(Map)`,
  `PmmlEnv.createSession(path)`
* Not yet implemented: `PmmlSession.fromBytes`, `getInputNames`, `getOutputNames`, and
  `NodeInfo`. They throw `UnsupportedOperationException` in this version.
* Arrow input is not wired yet; `arrow-vector` is declared optional in `pom.xml` for the
  follow-up that binds through the Arrow C Data Interface.

See `docs/src/deployment/java.md` for the loading order, the input mapping, and the error
mapping.
