package com.pmmlruntime;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Loads libpmmlruntime_jni, like ai.onnxruntime.platform.NativeLibraryLoader.
 *
 * <p>Resolution order:
 * <ol>
 *   <li>{@code System.loadLibrary("pmmlruntime_jni")}, for callers that point
 *       {@code java.library.path} at the built shared library.</li>
 *   <li>The copy shipped inside this jar under
 *       {@code native/<os>-<arch>/libpmmlruntime_jni.*}, extracted to a temp file.</li>
 *   <li>Cargo build directories, so {@code mvn test} works without extra flags:
 *       {@code java/native/target/debug|release/libpmmlruntime_jni.so}
 *       (also {@code .dylib} on macOS, {@code .dll} on Windows).</li>
 * </ol>
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
            // fall through to the shipped copy
        }
        String libName = libName();
        if (loadBundled(libName)) {
            loaded = true;
            return;
        }
        String[] candidates = cargoCandidates(libName);
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

    private static String libName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("mac") ? "libpmmlruntime_jni.dylib"
            : os.contains("win") ? "pmmlruntime_jni.dll"
            : "libpmmlruntime_jni.so";
    }

    /** Directory name used by release builds for the bundled copy. */
    private static String platformDir() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String name = os.contains("mac") ? "macos"
            : os.contains("win") ? "windows" : "linux";
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.contains("aarch64") || arch.contains("arm64")) {
            arch = "aarch64";
        } else if (arch.contains("64")) {
            arch = "x86_64";
        }
        return name + "-" + arch;
    }

    private static boolean loadBundled(String libName) {
        String resource = "/native/" + platformDir() + "/" + libName;
        try (InputStream in = NativeLoader.class.getResourceAsStream(resource)) {
            if (in == null) return false;
            Path dir = Files.createTempDirectory("pmmlruntime-native");
            Path target = dir.resolve(libName);
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            target.toFile().deleteOnExit();
            dir.toFile().deleteOnExit();
            System.load(target.toAbsolutePath().toString());
            return true;
        } catch (IOException | UnsatisfiedLinkError e) {
            return false;
        }
    }

    private static String[] cargoCandidates(String libName) {
        String userDir = System.getProperty("user.dir", ".");
        return new String[] {
            userDir + "/native/target/debug/" + libName,
            userDir + "/native/target/release/" + libName,
            userDir + "/java/native/target/debug/" + libName,
            userDir + "/java/native/target/release/" + libName,
            "java/native/target/debug/" + libName,
            "java/native/target/release/" + libName,
        };
    }
    private NativeLoader() {}
}
