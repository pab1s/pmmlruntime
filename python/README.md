# python — Python binding (over `include/pmml_runtime.h`)

Like `onnxruntime` at repo root (`pyproject.toml` + C extension) — thin shim over `PmmlApi`, not a direct `use pmmlruntime::Session`.

```
python/
  pyproject.toml                  # maturin + pyo3, manylinux/macos/win wheels
  pmmlruntime/__init__.py         # InferenceSession wrapper
  _native/Cargo.toml              # pyo3 cdylib calling PmmlGetApi (NOT in workspace members)
  tests/test_inference.py
```

```python
import pmmlruntime as pm
env = pm.Env()
opts = pm.SessionOptions()
sess = pm.InferenceSession("model.pmml", sess_options=opts)
sess.get_inputs()                  # [NodeArg(name="Petal.Length", type="tensor(float)")]
sess.run(None, {"Petal.Length": 1.4})
sess.run(None, [{"x":1.0},{"x":2.0}])
import pyarrow as pa; sess.run(None, pa.table({"x":[1.0,2.0]}))  # Arrow via C Data Interface
b = sess.io_binding(); b.bind_input("x", 1.0); b.bind_output("predictedValue"); sess.run_with_iobinding(b)
```

* Build: `cargo build -p pmmlruntime --features capi` → `libpmmlruntime.so`; `python/_native` links it via `PmmlGetApi` and `py.allow_threads` around every `Run*`.
* Arrow: `pyarrow.Table._export_to_c` / `_import_from_c` → `ArrowArray`/`ArrowSchema` → `PmmlApi.RunArrow`.
* Install: `pip install -e python/` (dev) or `maturin develop` inside `python/`.

See `docs/BINDINGS.md`.
