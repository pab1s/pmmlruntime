package com.pmmlruntime;

import java.io.Closeable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * PmmlSession — like OrtSession, wraps native session handle (long).
 * AutoCloseable; holds jlong handle, not direct Rust Session.
 *
 * <p>v1 scope: {@code Continuous} doubles only. {@link #run(Map)} reads
 * {@code Petal.Length}/{@code Petal.Width} as doubles and returns
 * {@code predictedValue} as String. {@code Discrete} inputs via
 * {@code Session::string_to_value} are follow-up work.
 */
public final class PmmlSession implements Closeable {
    private long handle;

    private PmmlSession(long handle) { this.handle = handle; }

    public static PmmlSession fromFile(PmmlEnv env, String path) {
        long h = nCreateSession(env.getHandle(), path);
        if (h == 0) {
            throw new PmmlException(-1, "failed to create session for " + path);
        }
        return new PmmlSession(h);
    }

    public static PmmlSession fromBytes(PmmlEnv env, byte[] pmml) {
        throw new UnsupportedOperationException("fromBytes lands with CreateSessionFromArray (v2)");
    }

    public List<String> getInputNames() { throw new UnsupportedOperationException("getInputNames (v2)"); }
    public List<String> getOutputNames() { throw new UnsupportedOperationException("getOutputNames (v2)"); }

    public Map<String, Object> run(Map<String, Object> inputs) {
        Object l = inputs.get("Petal.Length");
        Object w = inputs.get("Petal.Width");
        if (!(l instanceof Number) || !(w instanceof Number)) {
            throw new IllegalArgumentException(
                "v1 supports Continuous doubles only: Petal.Length/Petal.Width must be Numbers (Discrete via string_to_value is follow-up)");
        }
        String label = nRun(handle, ((Number) l).doubleValue(), ((Number) w).doubleValue());
        Map<String, Object> out = new HashMap<>();
        out.put("predictedValue", label);
        return out;
    }

    // JNI natives
    private static native long nCreateSession(long envHandle, String path);
    private static native String nRun(long handle, double petalLength, double petalWidth);
    private static native void nRelease(long handle);

    @Override public void close() {
        if (handle != 0) { nRelease(handle); handle = 0; }
    }

    static { NativeLoader.load(); }
}
