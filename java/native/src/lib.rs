//! pmmlruntime-jni — JNI shim over include/pmml_runtime.h PmmlApi
//! Holds PmmlSession* as jlong (like ai.onnxruntime.OrtSession nativeHandle).
//! All calls go through PmmlGetApi() table, not directly `use pmmlruntime::Session`.

use jni::objects::{JClass, JString};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;

// Placeholder — real impl links libpmmlruntime.so and calls PmmlGetApi(1)->CreateSession etc.
// extern "C" { fn PmmlGetApi(version: u32) -> *const std::ffi::c_void; }

#[no_mangle]
pub extern "system" fn Java_com_pmmlruntime_PmmlSession_nCreateSession(
    _env: JNIEnv, _class: JClass, _path: JString,
) -> jlong {
    0 // TODO: call PmmlGetApi()->CreateSessionFromArray and return handle as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_pmmlruntime_PmmlSession_nRelease(
    _env: JNIEnv, _class: JClass, _handle: jlong,
) {
    // TODO: api->ReleaseSession(handle as *mut PmmlSession)
}
