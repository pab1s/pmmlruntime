"""pmmlruntime — Python package over PmmlApi C ABI (like onnxruntime).

Public API mirrors onnxruntime.InferenceSession so sklearn2pmml/lightgbm2pmml
users have zero friction:

    import pmmlruntime as pm
    sess = pm.InferenceSession("model.pmml")
    sess.get_inputs()
    sess.run(None, {"Petal.Length": 1.4})
    import pyarrow as pa
    sess.run(None, pa.table({"x": [1.0, 2.0]}))  # columnar zero-copy
"""

from . import _native

def hello() -> str:
    return _native.hello()

class GraphOptimizationLevel:
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_ALL = 3

class SessionOptions:
    def __init__(self):
        self.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.providers = ["CPUExecutionProvider"]

    def append_execution_provider(self, name: str, **kwargs):
        self.providers.append((name, kwargs))

class InferenceSession:
    """Thin wrapper — holds PmmlSession* handle via PmmlApi, like onnxruntime.InferenceSession."""

    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None):
        # TODO: call _native.create_session(path_or_bytes, sess_options)
        self._handle = None
        self.path_or_bytes = path_or_bytes
        self.sess_options = sess_options or SessionOptions()
        raise NotImplementedError("InferenceSession over PmmlApi not yet linked — stub on feat/c-binding")

    def get_inputs(self):
        raise NotImplementedError

    def get_outputs(self):
        raise NotImplementedError

    def get_modelmeta(self):
        raise NotImplementedError

    def run(self, output_names, input_feed):
        raise NotImplementedError

    def run_with_iobinding(self, binding):
        raise NotImplementedError

    def io_binding(self):
        raise NotImplementedError

__all__ = ["InferenceSession", "SessionOptions", "GraphOptimizationLevel", "hello"]
