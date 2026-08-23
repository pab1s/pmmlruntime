//! C ABI — stable like onnxruntime_c_api.h
//! v1 minimal: PmmlEnv, PmmlSession, PmmlCreate/Run/Release.

// For v1 we expose opaque pointers and status codes.
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[repr(C)]
pub struct PmmlEnv {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PmmlSession {
    _private: [u8; 0],
}

#[repr(C)]
pub enum PmmlStatusCode {
    Ok = 0,
    Error = 1,
}

/// Create Env. Caller must call `PmmlReleaseEnv`.
#[no_mangle]
pub extern "C" fn PmmlCreateEnv(env_out: *mut *mut PmmlEnv) -> i32 {
    if env_out.is_null() {
        return PmmlStatusCode::Error as i32;
    }
    // In v1 we just allocate a dummy
    let _env = Box::new(PmmlEnv { _private: [] });
    unsafe {
        *env_out = Box::into_raw(_env);
    }
    PmmlStatusCode::Ok as i32
}

#[no_mangle]
pub extern "C" fn PmmlReleaseEnv(env: *mut PmmlEnv) {
    if env.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(env);
    }
}

/// Create Session from file path.
#[no_mangle]
pub extern "C" fn PmmlCreateSession(
    _env: *mut PmmlEnv,
    path: *const c_char,
    session_out: *mut *mut PmmlSession,
) -> i32 {
    if path.is_null() || session_out.is_null() {
        return PmmlStatusCode::Error as i32;
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let _path_str = c_str.to_string_lossy().into_owned();
    // Stub: allocate dummy session
    let sess = Box::new(PmmlSession { _private: [] });
    unsafe {
        *session_out = Box::into_raw(sess);
    }
    PmmlStatusCode::Ok as i32
}

#[no_mangle]
pub extern "C" fn PmmlReleaseSession(session: *mut PmmlSession) {
    if session.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(session);
    }
}

pub fn placeholder() {}
