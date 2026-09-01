package com.pmmlruntime;

import java.io.Closeable;
import java.util.List;
import java.util.Map;

/**
 * PmmlSession — like OrtSession, wraps PmmlSession* handle (long) via PmmlApi.
 * AutoCloseable; holds jlong handle, not direct Rust Session.
 */
public final class PmmlSession implements Closeable {
    private long handle;

    private PmmlSession(long handle) { this.handle = handle; }

    public static PmmlSession fromFile(PmmlEnv env, String path) {
        // TODO: nCreateSession(env.getHandle(), path) -> handle
        throw new UnsupportedOperationException("stub");
    }

    public static PmmlSession fromBytes(PmmlEnv env, byte[] pmml) {
        throw new UnsupportedOperationException("stub");
    }

    public List<String> getInputNames() { throw new UnsupportedOperationException("stub"); }
    public List<String> getOutputNames() { throw new UnsupportedOperationException("stub"); }

    public Map<String, Object> run(Map<String, Object> inputs) {
        throw new UnsupportedOperationException("stub — will call api->Run / RunArrow");
    }

    // JNI natives
    private static native long nCreateSession(long envHandle, String path);
    private static native void nRelease(long handle);

    @Override public void close() {
        if (handle != 0) { nRelease(handle); handle = 0; }
    }

    static { NativeLoader.load(); }
}
