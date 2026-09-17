"""pmmlruntime — Python package (ORT-style InferenceSession over pmmlruntime::Session)."""

from ._native import InferenceSession as _NativeSession, SessionOptions as _NativeSessionOptions, GraphOptimizationLevelCls, hello
# Re-export native InferenceSession directly — like onnxruntime.InferenceSession
InferenceSession = _NativeSession
SessionOptions = _NativeSessionOptions
GraphOptimizationLevel = GraphOptimizationLevelCls

__all__ = ["InferenceSession", "SessionOptions", "GraphOptimizationLevel", "hello"]
