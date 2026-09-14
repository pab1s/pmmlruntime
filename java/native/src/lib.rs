//! pmmlruntime-jni — JNI shim over `pmmlruntime::Session` (same process, no dlopen).
//!
//! v1 scope: `Continuous` doubles only (`Petal.Length` / `Petal.Width` for
//! `bench/pmml/DecisionTreeIris.pmml`). `Discrete` inputs via
//! `Session::string_to_value` are a follow-up — see `nRun` docs below.
//!
//! Ownership: `nCreateSession` boxes `(REnv, RSession)` and returns it as
//! `jlong`; `nRelease` frees it. `PmmlEnv` is owned by the session box for v1
//! (Java `PmmlEnv.create()` returns a placeholder handle `1L`).

use jni::objects::{JClass, JString};
use jni::sys::{jdouble, jlong, jstring};
use jni::JNIEnv;

use std::collections::HashMap;

use pmmlruntime::base::{SymbolId, Value};
use pmmlruntime::session::batch::Batch;
use pmmlruntime::session::{PmmlEnv as REnv, Session as RSession, SessionOptions};

/// Boxed native state held as `jlong` on the Java side.
type Handle = (REnv, RSession);

fn throw_runtime(env: &mut JNIEnv, msg: String) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

/// Create a session from a PMML file path.
///
/// Java: `private static native long nCreateSession(long envHandle, String path);`
/// `_env_handle` is reserved for v2 (explicit `PmmlEnv*` threading); v1 owns
/// the env inside the session box.
#[no_mangle]
pub extern "system" fn Java_com_pmmlruntime_PmmlSession_nCreateSession(
    mut env: JNIEnv,
    _class: JClass,
    _env_handle: jlong,
    path: JString,
) -> jlong {
    let path: String = match env.get_string(&path) {
        Ok(s) => s.into(),
        Err(e) => {
            throw_runtime(&mut env, format!("nCreateSession: bad path string: {e}"));
            return 0;
        }
    };
    let renv = REnv::new();
    let sess = match RSession::from_file(&renv, &path, SessionOptions::default()) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, format!("nCreateSession: {e}"));
            return 0;
        }
    };
    Box::into_raw(Box::new((renv, sess))) as jlong
}

/// Score one Iris row.
///
/// Java: `private static native String nRun(long handle, double petalLength, double petalWidth);`
///
/// v1 takes the two `Continuous` doubles directly to avoid fragile
/// `java.util.Map` ↔ Rust conversion. `Discrete` (string/categorical) inputs
/// should go through `Session::string_to_value` — follow-up work.
///
/// Returns the `predictedValue` label as a Java `String` (`Discrete` resolved
/// via `Session.ir.symbol_names`, `Continuous` via `to_string`).
#[no_mangle]
pub extern "system" fn Java_com_pmmlruntime_PmmlSession_nRun(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    petal_length: jdouble,
    petal_width: jdouble,
) -> jstring {
    if handle == 0 {
        throw_runtime(&mut env, "nRun: null session handle".to_string());
        return std::ptr::null_mut();
    }
    // SAFETY: handle came from `Box::into_raw` in `nCreateSession` and is only
    // freed in `nRelease` after `close()` zeroes the Java handle.
    let (_, sess) = unsafe { &*(handle as *const Handle) };
    let mut input: HashMap<String, Value> = HashMap::with_capacity(2);
    input.insert("Petal.Length".to_string(), Value::Continuous(petal_length));
    input.insert("Petal.Width".to_string(), Value::Continuous(petal_width));
    let row = match sess.run(&input as &dyn Batch) {
        Ok(r) => r,
        Err(e) => {
            throw_runtime(&mut env, format!("nRun: {e}"));
            return std::ptr::null_mut();
        }
    }
    .into_single();
    let row = match row {
        Some(m) => m,
        None => {
            throw_runtime(&mut env, "nRun: empty result".to_string());
            return std::ptr::null_mut();
        }
    };
    let label = match row.get("predictedValue") {
        Some(Value::Discrete(SymbolId(id))) => sess
            .ir
            .symbol_names
            .get(&SymbolId(*id))
            .cloned()
            .unwrap_or_else(|| format!("Symbol({id})")),
        Some(Value::Continuous(f)) => {
            // Classification models return Discrete; keep Continuous readable.
            if f.fract() == 0.0 {
                format!("{}", *f as i64)
            } else {
                format!("{f}")
            }
        }
        Some(Value::Missing) => "Missing".to_string(),
        None => {
            throw_runtime(&mut env, "nRun: missing predictedValue".to_string());
            return std::ptr::null_mut();
        }
    };
    match env.new_string(label) {
        Ok(s) => s.into_raw(),
        Err(e) => {
            throw_runtime(&mut env, format!("nRun: new_string failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free the boxed `(REnv, RSession)`.
///
/// Java: `private static native void nRelease(long handle);`
#[no_mangle]
pub extern "system" fn Java_com_pmmlruntime_PmmlSession_nRelease(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: inverse of `Box::into_raw` in `nCreateSession`; Java zeroes its
    // handle after `close()` so double-free cannot happen via the wrapper.
    unsafe {
        let _ = Box::from_raw(handle as *mut Handle);
    }
}
