package com.pmmlruntime;

import java.io.File;

/**
 * Loads libpmmlruntime_jni, like ai.onnxruntime.platform.NativeLibraryLoader.
 *
 * <p>Resolution order: {@code System.loadLibrary("pmmlruntime_jni")} first
 * (works when {@code java.library.path} points at the built .so), then
 * absolute-path fallback to the cargo build outputs so {@code mvn test}
 * works without extra flags:
 * {@code java/native/target/debug|release/libpmmlruntime_jni.so}
 * (also {@code .dylib} on macOS, {@code .dll} on Windows).
 */
public final class NativeLoader {
    private static boolean loaded;
    public static synchronized void load() {
        if (loaded) return;
        try {
            System.loadLibrary("pmmlruntime_jni");
            loaded = true;
            return;
        } catch (UnsatisfiedLinkError ignored) {
            // fall through to absolute-path candidates
        }
        String os = System.getProperty("os.name", "").toLowerCase();
        String libName = os.contains("mac") ? "libpmmlruntime_jni.dylib"
            : os.contains("win") ? "pmmlruntime_jni.dll"
            : "libpmmlruntime_jni.so";
        String userDir = System.getProperty("user.dir", ".");
        String[] candidates = {
            userDir + "/native/target/debug/" + libName,
            userDir + "/native/target/release/" + libName,
            userDir + "/java/native/target/debug/" + libName,
            userDir + "/java/native/target/release/" + libName,
            "java/native/target/debug/" + libName,
            "java/native/target/release/" + libName,
        };
        for (String c : candidates) {
            File f = new File(c);
            if (f.exists()) {
                System.load(f.getAbsolutePath());
                loaded = true;
                return;
            }
        }
        // Last resort: rethrow with actionable message.
        throw new UnsatisfiedLinkError(
            "libpmmlruntime_jni not found: build with "
            + "`cargo build --manifest-path java/native/Cargo.toml` "
            + "or set -Djava.library.path to its target dir");
    }
    private NativeLoader() {}
}
