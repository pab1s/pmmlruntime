package com.pmmlruntime;

/** Input/output metadata — like Ort NodeInfo / onnxruntime NodeArg. */
public record NodeInfo(String name, String dataType, String opType) {}
