package com.pmmlruntime;

/** Maps PmmlErrorCode + message from PmmlStatus (like OrtException). */
public class PmmlException extends RuntimeException {
    public final int code;
    public PmmlException(int code, String message) { super(message); this.code = code; }
}
