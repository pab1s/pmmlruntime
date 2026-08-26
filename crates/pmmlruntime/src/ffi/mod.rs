//! C ABI — stable C API for PMML scoring.
//!
//! This crate exposes a stable C API (`PmmlEnv`, `PmmlSession`, `PmmlCreate*` / `PmmlRelease*`).
//! Design uses opaque handles, status-code returns, and `Safety` contracts for FFI callers.
//! Current bindings cover environment and session create/release; scoring via `PmmlRun*` is planned.
//!
//! # Ownership and thread safety
//!
//! - `PmmlEnv` holds `Arc<EnvInner>` (`crate::session::PmmlEnv`) and is `Send` + `Sync`. It is reference-counted;
//!   multiple `PmmlSession`s can share one `PmmlEnv`.
//! - `PmmlSession` holds `Box<Session>` (`crate::session::Session`) and is `Send` + `Sync` (scoring is `&self`).
//! - Callers must pair `PmmlCreateEnv` with `PmmlReleaseEnv` and `PmmlCreateSession` with `PmmlReleaseSession`.
//!   Double-free is undefined behavior.
//!
//! # ABI stability
//!
//! `PmmlEnv` and `PmmlSession` are `#[repr(C)]` opaque structs (`_private: [u8; 0]`). The layout is never
//! exposed to C; only pointers to them are passed. `PmmlStatusCode` is `#[repr(C)]` with `Ok = 0`, `Error = 1`
//! so C can compare to `0` or the enum.
//!
//! # What belongs here
//!
//! - `PmmlEnv` / `PmmlSession` opaque types.
//! - `PmmlStatusCode` return codes.
//! - `extern "C"` functions `PmmlCreateEnv`, `PmmlReleaseEnv`, `PmmlCreateSession`, `PmmlReleaseSession`.
//!
//! # Examples
//!
//! ```c
//! // C usage (not Rust):
//! // PmmlEnv *env = NULL;
//! // if (PmmlCreateEnv(&env) != 0) abort();
//! // PmmlSession *sess = NULL;
//! // if (PmmlCreateSession(env, "model.pmml", &sess) != 0) abort();
//! // PmmlReleaseSession(sess);
//! // PmmlReleaseEnv(env);
//! ```

use std::ffi::CStr;
use std::os::raw::c_char;

/// Opaque handle for the global environment (`Arc<EnvInner>` inside `crate::session::PmmlEnv`).
///
/// `PmmlEnv` is `Send` + `Sync` and reference-counted. It is created by [`PmmlCreateEnv`] and
/// must be freed by [`PmmlReleaseEnv`]. The struct is `#[repr(C)]` with `0` sized `_private` so C
/// sees it as an opaque pointer (`PmmlEnv *`) without layout.
///
/// # Invariants
///
/// - A `*mut PmmlEnv` must be either null or the result of `Box::into_raw(Box::new(PmmlEnv { ... }))` from `PmmlCreateEnv`.
/// - The caller must not dereference the pointer on the C side; only the Rust FFI functions may `Box::from_raw` it.
///
/// # Examples
///
/// ```c
/// PmmlEnv *env = NULL;
/// int rc = PmmlCreateEnv(&env);
/// // rc == 0 on success
/// ```
#[repr(C)]
pub struct PmmlEnv {
    _private: [u8; 0],
}

/// Opaque handle for a scoring session (`Box<Session>` inside `crate::session::Session`).
///
/// `PmmlSession` is `Send` + `Sync` for `&self` scoring. It holds `Arc<Ir>` and a boxed
/// `ExecutionProvider`. Currently created from a file path; future extensions may support
/// construction from bytes or `Ir` handles.
///
/// # Invariants
///
/// - A `*mut PmmlSession` must be either null or `Box::into_raw` from [`PmmlCreateSession`].
/// - Double `PmmlReleaseSession` is undefined behavior (double free).
#[repr(C)]
pub struct PmmlSession {
    _private: [u8; 0],
}

/// Status code for C API calls.
///
/// `Ok = 0` means success; `Error = 1` means failure (null pointer, file not found, parse error, etc.).
/// C callers should compare return value to `0` or to `PmmlStatusCode::Ok`.
///
/// # Variants
///
/// - `Ok` — success.
/// - `Error` — generic failure (details are not yet propagated to C; check logs or use Rust API for rich `PmmlError`).
#[repr(C)]
pub enum PmmlStatusCode {
    /// Success (`0`).
    Ok = 0,
    /// Generic error (`1`).
    Error = 1,
}

/// Create a new `PmmlEnv` handle.
///
/// Caller must pair each successful call with [`PmmlReleaseEnv`]. On failure `*env_out` is left untouched.
///
/// # Parameters
///
/// - `env_out`: `*mut *mut PmmlEnv` — out-pointer to receive `*mut PmmlEnv`. Must be non-null and valid for writes of `*mut PmmlEnv`.
///
/// # Returns
///
/// `PmmlStatusCode::Ok` (`0`) on success, `PmmlStatusCode::Error` (`1`) if `env_out` is null.
///
/// # Safety
///
/// - `env_out` must be a valid, non-null pointer to a `*mut PmmlEnv` slot that the caller owns and that is writable for `size_of::<*mut PmmlEnv>()`.
/// - The returned `*mut PmmlEnv` must be freed exactly once by `PmmlReleaseEnv`; freeing via other means or double-free is undefined behavior.
///
/// # Why sufficient
///
/// `PmmlCreateEnv` checks `is_null` before dereferencing `env_out`, so a null `env_out` safely returns error without UB.
/// Writing `Box::into_raw` to `*env_out` hands ownership to the caller as required.
///
/// # What goes wrong if violated
///
/// Passing an invalid `env_out` (dangling, unaligned, or null) causes dereference UB. Double-free of the returned handle causes heap corruption.
///
/// # Panics
///
/// This function does not panic; allocation failure aborts via Rust allocator. `no_mangle` + `extern "C"` will abort on panic (not unwind).
///
/// # Examples
///
/// ```c
/// PmmlEnv *env = NULL;
/// int rc = PmmlCreateEnv(&env);
/// if (rc != 0) { /* handle error */ }
/// // ... use env ...
/// PmmlReleaseEnv(env);
/// ```
#[no_mangle]
pub unsafe extern "C" fn PmmlCreateEnv(env_out: *mut *mut PmmlEnv) -> i32 {
    if env_out.is_null() {
        return PmmlStatusCode::Error as i32;
    }
    let env = Box::new(PmmlEnv { _private: [] });
    unsafe {
        *env_out = Box::into_raw(env);
    }
    PmmlStatusCode::Ok as i32
}

/// Release a `PmmlEnv` handle.
///
/// No-op if `env` is null (for C convenience). Otherwise reconstitutes the `Box` via `Box::from_raw` and drops it,
/// decrementing the internal `Arc` count.
///
/// # Parameters
///
/// - `env`: `*mut PmmlEnv` — handle from `PmmlCreateEnv`, or null.
///
/// # Returns
///
/// Nothing (`void` in C). Null is tolerated.
///
/// # Safety
///
/// - `env` must be either null or a pointer previously returned by `PmmlCreateEnv` and not yet freed.
/// - The caller must not use `env` after this call (use-after-free is UB).
/// - Passing a pointer not from `PmmlCreateEnv`, or double-free, is undefined behavior.
///
/// # Why sufficient
///
/// Null check avoids UB on null. `Box::from_raw` is only called on a valid `Box::into_raw` pointer, so ownership is correctly reclaimed.
///
/// # What goes wrong if violated
///
/// Use-after-free, double-free, or freeing a non-`PmmlCreateEnv` pointer causes allocator heap corruption and possible code execution.
///
/// # Panics
///
/// Does not panic. Null is handled. Invalid pointer is UB, not a panic.
///
/// # Examples
///
/// ```c
/// PmmlEnv *env = NULL;
/// PmmlCreateEnv(&env);
/// PmmlReleaseEnv(env);
/// env = NULL; // avoid use-after-free
/// ```
#[no_mangle]
pub unsafe extern "C" fn PmmlReleaseEnv(env: *mut PmmlEnv) {
    if env.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(env);
    }
}

/// Create a `PmmlSession` from a file path.
///
/// Currently validates `path` / `session_out` are non-null and returns a placeholder session.
/// Future implementation will read the file and build a full `Session` via `crate::xml::unmarshal` → `verify` → `lower`.
///
/// # Parameters
///
/// - `_env`: `*mut PmmlEnv` — reserved for future thread-pool / logger use (currently unused but reserved for ABI).
/// - `path`: `*const c_char` — null-terminated UTF-8 file path (e.g. `"model.pmml"`). Must be valid C string.
/// - `session_out`: `*mut *mut PmmlSession` — out-pointer for the new session. Must be non-null and writable.
///
/// # Returns
///
/// `PmmlStatusCode::Ok` (`0`) on success, `PmmlStatusCode::Error` (`1`) if `path` or `session_out` is null, or if `CStr::from_ptr` would read out of bounds (caller's responsibility).
///
/// # Safety
///
/// - `path` must be a valid null-terminated C string (NUL-terminated, points to `c_char` array owned by caller, and remains valid for the call). Reading past `NUL` is UB.
/// - `session_out` must be a valid non-null pointer to a `*mut PmmlSession` writable slot.
/// - The returned `*mut PmmlSession` must be freed exactly once by `PmmlReleaseSession`.
///
/// # Why sufficient
///
/// The function checks `path.is_null()` and `session_out.is_null()` before dereferencing, so null returns error without UB.
/// `CStr::from_ptr(path)` requires `path` is NUL-terminated; caller's contract to provide that makes the call safe.
///
/// # What goes wrong if violated
///
/// Passing a non-NUL-terminated `path` reads arbitrary memory (UB, may segfault or leak). Passing invalid `session_out` dereference is UB. Leaking `PmmlSession` leaks `Arc<Ir>` and provider.
///
/// # Panics
///
/// Will abort on panic (Rust `extern "C"` panic boundary). `CStr::to_string_lossy` does not panic on invalid UTF-8 (it replaces). Null handling returns error, not panic.
///
/// # Examples
///
/// ```c
/// PmmlEnv *env = NULL;
/// PmmlCreateEnv(&env);
/// PmmlSession *sess = NULL;
/// int rc = PmmlCreateSession(env, "model.pmml", &sess);
/// if (rc == 0) {
///     // ... score ...
///     PmmlReleaseSession(sess);
/// }
/// PmmlReleaseEnv(env);
/// ```
#[no_mangle]
pub unsafe extern "C" fn PmmlCreateSession(
    _env: *mut PmmlEnv,
    path: *const c_char,
    session_out: *mut *mut PmmlSession,
) -> i32 {
    if path.is_null() || session_out.is_null() {
        return PmmlStatusCode::Error as i32;
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let _path_str = c_str.to_string_lossy().into_owned();
    let sess = Box::new(PmmlSession { _private: [] });
    unsafe {
        *session_out = Box::into_raw(sess);
    }
    PmmlStatusCode::Ok as i32
}

/// Release a `PmmlSession` handle.
///
/// No-op if `session` is null. Otherwise reclaims `Box` and drops the inner `Session` (`Arc<Ir>` etc.).
///
/// # Parameters
///
/// - `session`: `*mut PmmlSession` from `PmmlCreateSession`, or null.
///
/// # Returns
///
/// Nothing.
///
/// # Safety
///
/// - `session` must be null or a live pointer from `PmmlCreateSession` not yet freed.
/// - Use-after-free or double-free is undefined behavior.
/// - The caller must ensure no scoring is in progress on another thread when freeing (or hold `Session: Send+Sync` externally).
///
/// # Why sufficient
///
/// Null check guards null. `Box::from_raw` on a valid `Box::into_raw` pointer correctly drops.
///
/// # What goes wrong if violated
///
/// Double-free heap corruption; use-after-free data race if another thread still holds `*mut PmmlSession`.
///
/// # Panics
///
/// Does not panic.
///
/// # Examples
///
/// ```c
/// PmmlSession *sess = NULL;
/// PmmlCreateSession(env, "model.pmml", &sess);
/// PmmlReleaseSession(sess);
/// ```
#[no_mangle]
pub unsafe extern "C" fn PmmlReleaseSession(session: *mut PmmlSession) {
    if session.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(session);
    }
}
