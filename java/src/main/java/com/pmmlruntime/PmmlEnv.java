package com.pmmlruntime;

import java.io.Closeable;

/**
 * PmmlEnv — like OrtEnvironment (holds global threadpool + logger, shared across Sessions).
 * Wraps PmmlEnv* (opaque handle) via PmmlGetApi(). AutoCloseable.
 *
 * <p>v1: the native env is owned by the session box in Rust
 * ({@code Box<(REnv, RSession)>}); {@code create()} returns a placeholder
 * handle {@code 1L}. v2 will thread a real {@code PmmlEnv*} handle.
 */
public final class PmmlEnv implements Closeable {
    private long handle; // PmmlEnv* as long, like ai.onnxruntime.OrtEnvironment nativeHandle

    private PmmlEnv(long handle) { this.handle = handle; }

    public static PmmlEnv create() {
        NativeLoader.load();
        return new PmmlEnv(1L);
    }

    public PmmlSession createSession(String path) {
        return PmmlSession.fromFile(this, path);
    }

    long getHandle() { return handle; }

    @Override public void close() {
        if (handle != 0) {
            // v1: nothing to release (env owned by session box); v2: api->ReleaseEnv(handle)
            handle = 0;
        }
    }
}
