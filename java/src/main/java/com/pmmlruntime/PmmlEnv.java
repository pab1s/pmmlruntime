package com.pmmlruntime;

import java.io.Closeable;

/**
 * PmmlEnv — like OrtEnvironment (holds global threadpool + logger, shared across Sessions).
 * Wraps PmmlEnv* (opaque handle) via PmmlGetApi(). AutoCloseable.
 */
public final class PmmlEnv implements Closeable {
    private long handle; // PmmlEnv* as long, like ai.onnxruntime.OrtEnvironment nativeHandle

    private PmmlEnv(long handle) { this.handle = handle; }

    public static PmmlEnv create() {
        // TODO: call PmmlGetApi().CreateEnv and return PmmlEnv
        throw new UnsupportedOperationException("PmmlApi not yet linked — stub on feat/java-binding");
    }

    public PmmlSession createSession(String path) {
        return PmmlSession.fromFile(this, path);
    }

    long getHandle() { return handle; }

    @Override public void close() {
        if (handle != 0) {
            // NativeLoader.api.ReleaseEnv(handle)
            handle = 0;
        }
    }
}
