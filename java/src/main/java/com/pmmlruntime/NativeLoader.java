package com.pmmlruntime;

import java.io.*;
import java.nio.file.*;

/**
 * Extracts libpmmlruntime.so/.dylib/.dll from src/main/resources/native/<os>-<arch>/ inside the jar,
 * like ai.onnxruntime.platform.NativeLibraryLoader.
 */
public final class NativeLoader {
    private static boolean loaded;
    public static synchronized void load() {
        if (loaded) return;
        // TODO: detect os.arch, extract resource to temp file, System.load(temp)
        loaded = true;
    }
    private NativeLoader() {}
}
