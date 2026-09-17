# Setup

> This guide covers setup for Rust, Python, and C with pmmlruntime. For scoring your first model in five minutes, see [Quickstart: Score Iris in 5 Minutes](./quickstart.md).

Install pmmlruntime without a JVM and keep the same **Session** and **Batch** types across bindings. You load `DecisionTreeIris.pmml` once and score it from any of the three surfaces.

## Prerequisites

Install one toolchain per binding and check the version before you continue.

| Language | Version | Verify |
| --- | --- | --- |
| **Rust** | `1.78+` stable, edition `2021` | `rustc --version` |
| **Python** | `3.8+`, `maturin >=1.7`, `pyo3 0.22` | `python --version && maturin --version` |
| **C** | `cbindgen 0.26+`, `cargo`, C11 compiler | `cbindgen --version && cc --version` |
| **Docs** | `mdbook` with `mdbook-mermaid` | `mdbook --version` |

> **Note:** No JVM is required at build time or at runtime. Parsing runs through `quick-xml 0.37` with the limits in [Security & Hardening](../production/security.md).

## Install rust

Rust 1.78 or later, edition 2021. The `simd` feature pulls `wide 0.7` for vectorized evaluation, and `python` pulls `pyo3 0.22` for the binding.

```bash
# from your crate directory
cargo add pmmlruntime --features simd
rustc --version  # must print 1.78 or later
cargo build
```

Pin the version in `Cargo.toml` when a build must be reproducible.

```toml
[dependencies]
pmmlruntime = { version = "0.1", features = ["simd"] }

[workspace.package]
edition = "2021"
rust-version = "1.78"
```

> **Attention:** `simd` requires SSE2, AVX2, or NEON. Build with `--features simd` on `x86_64` or `aarch64` and keep the default feature set everywhere else.

Create one environment per process, then load models against it. Keep `env` alive while any session lives.

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions};
let env = PmmlEnv::new();
let sess = Session::from_bytes(&env, &std::fs::read("bench/pmml/DecisionTreeIris.pmml")?, SessionOptions::default())?;
```

### Feature flags

| Feature | Default | Enables | Turn it on when |
| --- | --- | --- | --- |
| **`default`** | `[]` | No optional crate | Minimal build |
| **`simd`** | Off | `wide 0.7` `f64x4` | Batches above 256 rows on `x86_64` or `aarch64` |
| **`python`** | Off | `pyo3 0.22` | `maturin develop` or a wheel |

Scalar and SIMD builds produce bit-identical scores.

```toml
[dependencies]
pmmlruntime = { version = "0.1", features = [] }       # scalar
pmmlruntime = { version = "0.1", features = ["simd"] } # vectorized
```

```bash
cargo add pmmlruntime --precise 0.1.0
grep pmmlruntime Cargo.lock
```

> **Warning:** Keep `python` in `python/_native/Cargo.toml` with `crate-type = ["cdylib"]`.

## Install python

Install the wheel, or build the extension from source with `maturin`.

```bash
pip install pmmlruntime  # prebuilt wheel (when published)
# or build from source:
pip install maturin>=1.7 numpy>=1.21 pyarrow>=12
maturin develop --features python --manifest-path python/_native/Cargo.toml
python -c "import pmmlruntime; print(pmmlruntime.__all__)"
```

The build compiles `python/_native` into `pmmlruntime._native`. Run `maturin develop` again after you change Rust code.

```bash
python -c "import pmmlruntime; s=pmmlruntime.InferenceSession('bench/pmml/DecisionTreeIris.pmml'); print(s.get_inputs())"
```

The command above printed the input fields, which come from the same **Ir** the Rust binding builds. See [Python Bindings](../deployment/python.md).

## Install C

Build the shared library and generate the header with `cbindgen`. The `cbindgen.toml` maps Rust FFI types to `include/pmml_runtime.h`. Link the library from your own C source; the crate ships no C example program.

```bash
cargo build --release --lib
cbindgen --config cbindgen.toml --crate pmmlruntime --output include/pmml_runtime.h
cc -Iinclude -Ltarget/release -lpmmlruntime -o my_scorer my_scorer.c
./my_scorer bench/pmml/DecisionTreeIris.pmml
```

Include the header and call the versioned `PmmlApi` from `PmmlGetApi(PMML_API_VERSION)`. Opaque `PmmlEnv*` and `PmmlSession*` handles come in `Create` and `Release` pairs, and `PmmlStatus*` reports errors, where `NULL` means `PMML_OK`.

```c
#include "pmml_runtime.h"
const PmmlApi* api = PmmlGetApi(PMML_API_VERSION);
PmmlEnv* env = NULL;
api->CreateEnv(PMML_LOG_WARNING, "setup-check", &env);
PmmlSession* sess = NULL;
api->CreateSessionFromFile(env, "bench/pmml/DecisionTreeIris.pmml", NULL, &sess);
api->ReleaseSession(sess);
api->ReleaseEnv(env);
```

See [C ABI & FFI](../deployment/c.md).

## Setup commands

| Task | Rust | Python | C | Artifact |
| --- | --- | --- | --- | --- |
| **Add dependency** | `cargo add pmmlruntime --features simd` | `pip install pmmlruntime` | `cargo build --release --lib` | lock file, wheel, or shared object |
| **Pin version** | `Cargo.toml` `version = "=0.1.0"` | `pyproject.toml` `requires-python` | `cbindgen.toml` `include_version` | CI input |
| **Build binding** | `cargo build` | `maturin develop --features python --manifest-path python/_native/Cargo.toml` | `cbindgen --output include/pmml_runtime.h` | `_native` module or header |
| **Verify** | `cargo test` | `import pmmlruntime` | `cc -Iinclude -Ltarget/release -lpmmlruntime` | Working API |

Both `cargo add` and a hand-edited `Cargo.toml` give you the same dependency entry.

## Binding API matrix

One engine exposes three bindings over the same **Ir**.

| Capability | **Rust** | **Python** | **C** |
| --- | --- | --- | --- |
| **Load** | `Session::from_bytes` / `from_file` | `InferenceSession(path_or_bytes)` | `CreateSessionFromArray` |
| **Single row** | `HashMap<String, Value>` | `dict` | `PmmlValue[]` + `PmmlValueTag` |
| **Batch** | `Vec<HashMap>` | `list[dict]` | `RunBatch` |
| **Columnar** | `RecordBatch` `arrow 53` | `pyarrow.RecordBatch` | `RunArrow` |
| **Threads** | `Send + Sync` `Arc<Ir>` | `py.allow_threads` | `PmmlEnv*` / `PmmlSession*` |

Rust `Continuous(f64)`, Python `float`, and C `PmmlValue.continuous` all reach the same cold path.

## Per-target features

Keep `simd` on the platforms that support it and drop it elsewhere. Cargo resolves the dependency set per target.

```toml
[target.'cfg(any(target_arch = "x86_64", target_arch = "aarch64"))'.dependencies]
pmmlruntime = { version = "0.1", features = ["simd"] }
[target.'cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))'.dependencies]
pmmlruntime = { version = "0.1", features = [] }
```

Name the environment when you run several models in one process, so logs tell them apart.

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions};
let env = PmmlEnv::with_name("billing-scorer");
let sess = Session::from_file(&env, "bench/pmml/GradientBoosterTest.pmml", SessionOptions::default())?;
println!("active: {}", sess.num_active_fields());
# Ok::<(), Box<dyn std::error::Error>>(())
```

See [Session, Env & Lifecycle](../concepts/session.md).

> **Note:** `PmmlEnv::with_name` sets metadata only. The limits stay at 100 MB and depth 512.

## Remove a feature

| Binding | Add it back | Remove it | Effect |
| --- | --- | --- | --- |
| **Rust `simd`** | `cargo add pmmlruntime --features simd` | Delete `simd` from `Cargo.toml` | Scalar evaluation |
| **Python** | `maturin develop --features python --manifest-path python/_native/Cargo.toml` | `pip uninstall pmmlruntime` | No Python extension |
| **C** | `cargo build --release --lib && cbindgen --output include/pmml_runtime.h` | `rm -f include/pmml_runtime.h` | No header or shared library |
| **Docs** | `cargo install mdbook-mermaid` | `cargo uninstall mdbook-mermaid` | Diagrams stop rendering |

```bash
cargo add pmmlruntime --features ""
maturin develop --features python --manifest-path python/_native/Cargo.toml
```

## Verify

### Step 1: Run the fixtures

```bash
cargo test
# expected: 52/52 Pass
```

### Step 2: Check SIMD parity

```bash
cargo test --features simd
# expected: same 52/52 Pass
```

### Step 3: Build the book

```bash
mdbook serve docs --open
# expected: book at http://localhost:3000
```

### Step 4: Score a file cold and in batch

```bash
cargo run -p pmmlruntime --example score_file -- bench/pmml/DecisionTreeIris.pmml
cargo run -p pmmlruntime --example score_file -- bench/pmml/DecisionTreeIris.pmml input.csv --output out.csv
cat out.csv
```

If `cargo test` fails while parsing, check that the file is UTF-8 and under 100 MB. See [Validation & ModelVerification](../evaluation/validation.md).

## Troubleshooting

| Symptom | Cause | Fix | Verify |
| --- | --- | --- | --- |
| `rustc 1.78+ required` | Toolchain too old | `rustup update && rustc --version` | `1.78+` |
| `maturin develop` fails | Missing `--features python` | `maturin develop --features python --manifest-path python/_native/Cargo.toml` | import works |
| `PmmlGetApi` returns `NULL` | Wrong `PMML_API_VERSION` | Use `PMML_API_VERSION` from the header | header loads |
| `mdbook serve` drops diagrams | `mdbook-mermaid` missing | `cargo install mdbook-mermaid && mdbook serve docs --open` | diagrams render |
| `cargo test` parse failure | PMML is not UTF-8 or exceeds 100 MB | `file model.pmml && wc -c model.pmml` | UTF-8 under 100 MB |
| Stale wheel | Rust code changed | `maturin develop --features python --manifest-path python/_native/Cargo.toml` | `import _native` works |

See [Observability & Troubleshooting](../production/troubleshooting.md).

## Next Steps

- [Quickstart: Score Iris in 5 Minutes](./quickstart.md): score `DecisionTreeIris.pmml` end to end with `HashMap` and `RecordBatch`.
- [One File, Any Framework](./one-file-any-framework.md): swap `sklearn2pmml`, `jpmml-xgboost`, and `r2pmml` artifacts without code changes.
- [Session, Env & Lifecycle](../concepts/session.md): decide when to share `PmmlEnv` and how `Session` becomes `Send + Sync`.
- [C ABI & FFI](../deployment/c.md): call `PmmlGetApi`, `RunBatch`, and Arrow through `include/pmml_runtime.h`.

*Next: [Quickstart: Score Iris in 5 Minutes](./quickstart.md) → · Previous: [Introduction](../README.md)*
