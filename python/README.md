# python — Python binding

`InferenceSession` over the `pmmlruntime` Rust crate, built with maturin and pyo3.

```
python/
  pyproject.toml                  # maturin + pyo3, manylinux / macOS / Windows wheels
  pmmlruntime/__init__.py         # re-exports InferenceSession, SessionOptions, GraphOptimizationLevel
  _native/Cargo.toml              # pyo3 cdylib depending on the crate by path
  tests/test_inference.py
  tests/test_arrow.py
```

```python
import pmmlruntime as pm

sess = pm.InferenceSession("model.pmml")          # or pm.InferenceSession(bytes)
sess.get_inputs()                                 # [{"name": ..., "type": ...}]
sess.get_outputs()
sess.get_modelmeta()
sess.run(None, {"Petal.Length": 1.4, "Petal.Width": 0.2})
sess.run(None, [{"Petal.Length": 1.4}, {"Petal.Length": 6.0}])
```

* Build for development: `maturin develop` inside `python/`, or `pip install -e python/`.
  On a Python newer than pyo3 0.22 knows, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
* Build a wheel: `maturin build --release --out dist` inside `python/`.
* Arrow: `run` takes a `dict` or `list[dict]`. A pyarrow Table raises `TypeError` with a
  pointer to `to_pylist()`; `io_binding` and `run_with_iobinding` raise `NotImplementedError`.
* `None` becomes `Value::Missing`; `str` goes through `Session::string_to_value`; `bool`
  becomes 0 or 1; every `run` releases the GIL.
* Runtime dependencies: `numpy>=1.21`, `pyarrow>=12`.

See `docs/src/deployment/python.md` for the value mapping and the release commands.
