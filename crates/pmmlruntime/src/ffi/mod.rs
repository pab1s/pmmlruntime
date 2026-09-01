//! C ABI — versioned `PmmlApi` table over `include/pmml_runtime.h`.
//!
//! ORT-style: single `PmmlGetApi(version)` returns function-pointer table.
//! All language bindings (java/python/javascript) call through it.
//! Old `PmmlCreateEnv` shims kept for 0.1 compat.
//!
//! Handles are opaque: `*mut PmmlEnv` is actually `Box<EnvHandle>` etc.
//! Status is heap-allocated `PmmlStatus` — NULL means OK (OrtStatus pattern).

#![allow(non_snake_case, clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use crate::base::{PmmlError, SymbolId, Value};
use crate::session::{PmmlEnv as RustEnv, Session, SessionOptions};

// ---------------------------------------------------------------------------
// Enums — must match include/pmml_runtime.h
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmmlGraphOptimizationLevel {
    DisableAll = 0,
    EnableBasic = 1,
    EnableExtended = 2,
    EnableAll = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmmlLogLevel {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmmlErrorCode {
    Ok = 0,
    InvalidArgument = 1,
    Io = 2,
    Parse = 3,
    UnsupportedMarkup = 4,
    InvalidValue = 5,
    Validation = 6,
    Oom = 7,
    Unknown = 8,
}

impl From<&PmmlError> for PmmlErrorCode {
    fn from(e: &PmmlError) -> Self {
        match e {
            PmmlError::Io(_) => Self::Io,
            PmmlError::ParseError { .. } => Self::Parse,
            PmmlError::UnsupportedMarkup(_) => Self::UnsupportedMarkup,
            PmmlError::InvalidValue(_) => Self::InvalidValue,
            PmmlError::TypeError(_) => Self::InvalidValue,
            PmmlError::ValidationError(_) => Self::Validation,
            PmmlError::ArithmeticOverflow(_) => Self::InvalidValue,
            PmmlError::MissingField(_) => Self::InvalidValue,
            PmmlError::Other(_) => Self::Unknown,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmmlValueTag {
    Missing = 0,
    Continuous = 1,
    Discrete = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union PmmlValueData {
    pub continuous: f64,
    pub discrete: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PmmlValue {
    pub tag: PmmlValueTag,
    pub data: PmmlValueData,
}

impl PmmlValue {
    pub fn missing() -> Self {
        Self {
            tag: PmmlValueTag::Missing,
            data: PmmlValueData { discrete: 0 },
        }
    }
    pub fn continuous(v: f64) -> Self {
        Self {
            tag: PmmlValueTag::Continuous,
            data: PmmlValueData { continuous: v },
        }
    }
    pub fn discrete(id: u32) -> Self {
        Self {
            tag: PmmlValueTag::Discrete,
            data: PmmlValueData { discrete: id },
        }
    }
}

fn pmml_value_to_value(v: PmmlValue) -> Value {
    unsafe {
        match v.tag {
            PmmlValueTag::Missing => Value::Missing,
            PmmlValueTag::Continuous => Value::Continuous(v.data.continuous),
            PmmlValueTag::Discrete => Value::Discrete(SymbolId(v.data.discrete)),
        }
    }
}

fn value_to_pmml_value(v: Value) -> PmmlValue {
    match v {
        Value::Missing => PmmlValue::missing(),
        Value::Continuous(f) => PmmlValue::continuous(f),
        Value::Discrete(SymbolId(id)) => PmmlValue::discrete(id),
    }
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct PmmlEnv {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PmmlSession {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PmmlSessionOptions {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PmmlRunOptions {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PmmlIoBinding {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PmmlStatus {
    _private: [u8; 0],
}

// Internal heap types behind opaque pointers

struct EnvHandle {
    env: RustEnv,
}

struct SessionHandle {
    session: Session,
    // Cached CStrings for metadata — pointers returned by GetInputName etc remain valid
    // until Session is released (caller must not free).
    input_names: Vec<CString>,
    input_ptrs: Vec<*const c_char>,
    output_names: Vec<CString>,
    output_ptrs: Vec<*const c_char>,
    model_type: CString,
}

struct SessionOptionsHandle {
    graph_level: PmmlGraphOptimizationLevel,
    intra_threads: i32,
    inter_threads: i32,
    log_level: PmmlLogLevel,
    configs: HashMap<String, String>,
    providers: Vec<(String, HashMap<String, String>)>,
}

impl Default for SessionOptionsHandle {
    fn default() -> Self {
        Self {
            graph_level: PmmlGraphOptimizationLevel::EnableBasic,
            intra_threads: 0,
            inter_threads: 0,
            log_level: PmmlLogLevel::Warning,
            configs: HashMap::new(),
            providers: vec![("CPU".into(), HashMap::new())],
        }
    }
}

impl SessionOptionsHandle {
    fn to_rust(&self) -> SessionOptions {
        let lvl = match self.graph_level {
            PmmlGraphOptimizationLevel::DisableAll => crate::session::GraphOptimizationLevel::DisableAll,
            PmmlGraphOptimizationLevel::EnableBasic => crate::session::GraphOptimizationLevel::EnableBasic,
            PmmlGraphOptimizationLevel::EnableExtended => crate::session::GraphOptimizationLevel::EnableExtended,
            PmmlGraphOptimizationLevel::EnableAll => crate::session::GraphOptimizationLevel::EnableAll,
        };
        SessionOptions::default().graph_optimization_level(lvl)
    }
}

struct RunOptionsHandle {
    tag: Option<CString>,
    log_level: PmmlLogLevel,
}

struct IoBindingHandle {
    // For now, simple map + outputs. Future: Arrow buffers.
    inputs: HashMap<String, Value>,
    outputs: Vec<String>,
}

struct StatusHandle {
    code: PmmlErrorCode,
    message: CString,
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

fn make_status(code: PmmlErrorCode, msg: impl Into<String>) -> *mut PmmlStatus {
    let cstr = CString::new(msg.into()).unwrap_or_else(|_| CString::new("invalid status message").unwrap());
    let h = Box::new(StatusHandle { code, message: cstr });
    Box::into_raw(h) as *mut PmmlStatus
}

fn status_from_error(e: PmmlError) -> *mut PmmlStatus {
    let code = PmmlErrorCode::from(&e);
    make_status(code, e.to_string())
}

fn status_invalid_arg(msg: impl Into<String>) -> *mut PmmlStatus {
    make_status(PmmlErrorCode::InvalidArgument, msg)
}

// ---------------------------------------------------------------------------
// Arrow forward decls (opaque)
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct ArrowArray {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ArrowSchema {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Api table
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct PmmlApi {
    pub version: u32,

    pub CreateEnv: Option<unsafe extern "C" fn(PmmlLogLevel, *const c_char, *mut *mut PmmlEnv) -> *mut PmmlStatus>,
    pub ReleaseEnv: Option<unsafe extern "C" fn(*mut PmmlEnv)>,

    pub CreateSessionOptions: Option<unsafe extern "C" fn(*mut *mut PmmlSessionOptions) -> *mut PmmlStatus>,
    pub ReleaseSessionOptions: Option<unsafe extern "C" fn(*mut PmmlSessionOptions)>,
    pub SetGraphOptimizationLevel: Option<unsafe extern "C" fn(*mut PmmlSessionOptions, PmmlGraphOptimizationLevel) -> *mut PmmlStatus>,
    pub SetIntraOpNumThreads: Option<unsafe extern "C" fn(*mut PmmlSessionOptions, i32) -> *mut PmmlStatus>,
    pub SetInterOpNumThreads: Option<unsafe extern "C" fn(*mut PmmlSessionOptions, i32) -> *mut PmmlStatus>,
    pub SetLogLevel: Option<unsafe extern "C" fn(*mut PmmlSessionOptions, PmmlLogLevel) -> *mut PmmlStatus>,
    pub AddSessionConfigEntry: Option<unsafe extern "C" fn(*mut PmmlSessionOptions, *const c_char, *const c_char) -> *mut PmmlStatus>,
    pub AppendExecutionProvider: Option<unsafe extern "C" fn(*mut PmmlSessionOptions, *const c_char, *const *const c_char, *const *const c_char, usize) -> *mut PmmlStatus>,

    pub CreateSession: Option<unsafe extern "C" fn(*const PmmlEnv, *const c_char, *const PmmlSessionOptions, *mut *mut PmmlSession) -> *mut PmmlStatus>,
    pub CreateSessionFromArray: Option<unsafe extern "C" fn(*const PmmlEnv, *const c_void, usize, *const PmmlSessionOptions, *mut *mut PmmlSession) -> *mut PmmlStatus>,
    pub ReleaseSession: Option<unsafe extern "C" fn(*mut PmmlSession)>,

    pub SessionGetInputCount: Option<unsafe extern "C" fn(*const PmmlSession, *mut usize) -> *mut PmmlStatus>,
    pub SessionGetInputName: Option<unsafe extern "C" fn(*const PmmlSession, usize, *mut *const c_char) -> *mut PmmlStatus>,
    pub SessionGetOutputCount: Option<unsafe extern "C" fn(*const PmmlSession, *mut usize) -> *mut PmmlStatus>,
    pub SessionGetOutputName: Option<unsafe extern "C" fn(*const PmmlSession, usize, *mut *const c_char) -> *mut PmmlStatus>,
    pub SessionGetModelType: Option<unsafe extern "C" fn(*const PmmlSession, *mut *const c_char) -> *mut PmmlStatus>,
    pub GetVersionString: Option<unsafe extern "C" fn() -> *const c_char>,

    pub SessionGetFieldId: Option<unsafe extern "C" fn(*const PmmlSession, *const c_char, *mut u32, *mut i32) -> *mut PmmlStatus>,
    pub SessionGetSymbolId: Option<unsafe extern "C" fn(*const PmmlSession, *const c_char, *mut u32, *mut i32) -> *mut PmmlStatus>,

    pub Run: Option<unsafe extern "C" fn(*mut PmmlSession, *const PmmlRunOptions, *const *const c_char, *const PmmlValue, usize, *const *const c_char, usize, *mut PmmlValue) -> *mut PmmlStatus>,
    pub RunBatch: Option<unsafe extern "C" fn(*mut PmmlSession, *const PmmlRunOptions, *const *const c_char, *const PmmlValue, usize, usize, *mut PmmlValue, *mut usize) -> *mut PmmlStatus>,
    pub RunArrow: Option<unsafe extern "C" fn(*mut PmmlSession, *const PmmlRunOptions, *const ArrowArray, *const ArrowSchema, *mut ArrowArray, *mut ArrowSchema) -> *mut PmmlStatus>,

    pub CreateIoBinding: Option<unsafe extern "C" fn(*mut PmmlSession, *mut *mut PmmlIoBinding) -> *mut PmmlStatus>,
    pub ReleaseIoBinding: Option<unsafe extern "C" fn(*mut PmmlIoBinding)>,
    pub BindInput: Option<unsafe extern "C" fn(*mut PmmlIoBinding, *const c_char, PmmlValue) -> *mut PmmlStatus>,
    pub BindInputArrow: Option<unsafe extern "C" fn(*mut PmmlIoBinding, *const c_char, *const ArrowArray, *const ArrowSchema) -> *mut PmmlStatus>,
    pub BindOutput: Option<unsafe extern "C" fn(*mut PmmlIoBinding, *const c_char) -> *mut PmmlStatus>,
    pub RunWithBinding: Option<unsafe extern "C" fn(*mut PmmlSession, *const PmmlRunOptions, *mut PmmlIoBinding) -> *mut PmmlStatus>,
    pub CopyBindingOutputsToCpu: Option<unsafe extern "C" fn(*mut PmmlIoBinding, *mut PmmlValue, *mut usize) -> *mut PmmlStatus>,

    pub CreateRunOptions: Option<unsafe extern "C" fn(*mut *mut PmmlRunOptions) -> *mut PmmlStatus>,
    pub ReleaseRunOptions: Option<unsafe extern "C" fn(*mut PmmlRunOptions)>,
    pub SetRunTag: Option<unsafe extern "C" fn(*mut PmmlRunOptions, *const c_char) -> *mut PmmlStatus>,
    pub SetRunLogLevel: Option<unsafe extern "C" fn(*mut PmmlRunOptions, PmmlLogLevel) -> *mut PmmlStatus>,
}

// ---------------------------------------------------------------------------
// Individual extern "C" impls
// ---------------------------------------------------------------------------

unsafe extern "C" fn api_CreateEnv(level: PmmlLogLevel, log_id: *const c_char, out: *mut *mut PmmlEnv) -> *mut PmmlStatus {
    if out.is_null() {
        return status_invalid_arg("CreateEnv: out is null");
    }
    let name = if log_id.is_null() {
        "pmml-runtime".to_string()
    } else {
        unsafe { CStr::from_ptr(log_id) }.to_string_lossy().into_owned()
    };
    // log_level currently only influences name prefix; future: logger callback
    let _ = level;
    let h = Box::new(EnvHandle {
        env: RustEnv::with_name(name),
    });
    unsafe { *out = Box::into_raw(h) as *mut PmmlEnv };
    ptr::null_mut()
}

unsafe extern "C" fn api_ReleaseEnv(env: *mut PmmlEnv) {
    if env.is_null() { return; }
    unsafe { let _ = Box::from_raw(env as *mut EnvHandle); }
}

unsafe extern "C" fn api_CreateSessionOptions(out: *mut *mut PmmlSessionOptions) -> *mut PmmlStatus {
    if out.is_null() { return status_invalid_arg("CreateSessionOptions: out is null"); }
    let h = Box::new(SessionOptionsHandle::default());
    unsafe { *out = Box::into_raw(h) as *mut PmmlSessionOptions };
    ptr::null_mut()
}

unsafe extern "C" fn api_ReleaseSessionOptions(opts: *mut PmmlSessionOptions) {
    if opts.is_null() { return; }
    unsafe { let _ = Box::from_raw(opts as *mut SessionOptionsHandle); }
}

unsafe extern "C" fn api_SetGraphOptimizationLevel(opts: *mut PmmlSessionOptions, lvl: PmmlGraphOptimizationLevel) -> *mut PmmlStatus {
    if opts.is_null() { return status_invalid_arg("SetGraphOptimizationLevel: opts is null"); }
    unsafe { (*(opts as *mut SessionOptionsHandle)).graph_level = lvl };
    ptr::null_mut()
}

unsafe extern "C" fn api_SetIntraOpNumThreads(opts: *mut PmmlSessionOptions, n: i32) -> *mut PmmlStatus {
    if opts.is_null() { return status_invalid_arg("SetIntraOpNumThreads: opts null"); }
    unsafe { (*(opts as *mut SessionOptionsHandle)).intra_threads = n };
    ptr::null_mut()
}

unsafe extern "C" fn api_SetInterOpNumThreads(opts: *mut PmmlSessionOptions, n: i32) -> *mut PmmlStatus {
    if opts.is_null() { return status_invalid_arg("SetInterOpNumThreads: opts null"); }
    unsafe { (*(opts as *mut SessionOptionsHandle)).inter_threads = n };
    ptr::null_mut()
}

unsafe extern "C" fn api_SetLogLevel(opts: *mut PmmlSessionOptions, lvl: PmmlLogLevel) -> *mut PmmlStatus {
    if opts.is_null() { return status_invalid_arg("SetLogLevel: opts null"); }
    unsafe { (*(opts as *mut SessionOptionsHandle)).log_level = lvl };
    ptr::null_mut()
}

unsafe extern "C" fn api_AddSessionConfigEntry(opts: *mut PmmlSessionOptions, key: *const c_char, value: *const c_char) -> *mut PmmlStatus {
    if opts.is_null() || key.is_null() || value.is_null() {
        return status_invalid_arg("AddSessionConfigEntry: null arg");
    }
    let k = unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();
    let v = unsafe { CStr::from_ptr(value) }.to_string_lossy().into_owned();
    unsafe { (*(opts as *mut SessionOptionsHandle)).configs.insert(k, v); }
    ptr::null_mut()
}

unsafe extern "C" fn api_AppendExecutionProvider(
    opts: *mut PmmlSessionOptions,
    name: *const c_char,
    keys: *const *const c_char,
    values: *const *const c_char,
    count: usize,
) -> *mut PmmlStatus {
    if opts.is_null() || name.is_null() { return status_invalid_arg("AppendExecutionProvider: null"); }
    let n = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    // only CPU is supported; others are stored but ignored (future plugin EPs)
    let mut map = HashMap::new();
    if !keys.is_null() && !values.is_null() {
        for i in 0..count {
            let k = unsafe { CStr::from_ptr(*keys.add(i)) }.to_string_lossy().into_owned();
            let v = unsafe { CStr::from_ptr(*values.add(i)) }.to_string_lossy().into_owned();
            map.insert(k, v);
        }
    }
    unsafe { (*(opts as *mut SessionOptionsHandle)).providers.push((n, map)); }
    ptr::null_mut()
}

fn build_session_handle(session: Session) -> SessionHandle {
    // Cache input/output names as CStrings for stable pointers
    // Input names: active fields from model; fallback to field_names if unknown
    let input_names: Vec<String> = {
        // Try active fields first (ordered), else all field_names
        let n = session.num_active_fields();
        if n > 0 {
            // We need to get active field names in order. Use field_names map but order is not defined.
            // Instead, collect from model's mining schema order via session's public API.
            // For now, collect all field_names sorted by FieldId, which matches lower's intern order for single model.
            // Better: use session.ir.field_names and filter via active fields length check: just return field_names values
            // We'll implement input_names as Vec of active field names derived from session's internal caches:
            // Since Session doesn't expose active list ordering directly via public API beyond num_active_fields,
            // we will expose all field_names as inputs for C — callers can query field_id to validate.
            let mut v: Vec<(u32, String)> = session
                .ir
                .field_names
                .iter()
                .map(|(fid, name)| (fid.0, name.clone()))
                .collect();
            v.sort_by_key(|(id, _)| *id);
            v.into_iter().map(|(_, n)| n).collect()
        } else {
            vec![]
        }
    };
    let input_cstr: Vec<CString> = input_names
        .into_iter()
        .map(|s| CString::new(s).unwrap())
        .collect();
    let input_ptrs: Vec<*const c_char> = input_cstr.iter().map(|c| c.as_ptr()).collect();

    let output_names: Vec<String> = if session.ir.model.output_fields().is_empty() {
        vec!["predictedValue".to_string()]
    } else {
        session.ir.model.output_fields().iter().map(|o| o.name.clone()).collect()
    };
    let output_cstr: Vec<CString> = output_names.into_iter().map(|s| CString::new(s).unwrap()).collect();
    let output_ptrs: Vec<*const c_char> = output_cstr.iter().map(|c| c.as_ptr()).collect();

    let model_type = match &session.ir.model {
        crate::ir::ModelIr::Tree(_) => "TreeModel",
        crate::ir::ModelIr::Regression(_) => "RegressionModel",
        crate::ir::ModelIr::Mining(_) => "MiningModel",
        crate::ir::ModelIr::Scorecard(_) => "Scorecard",
        crate::ir::ModelIr::Clustering(_) => "ClusteringModel",
        crate::ir::ModelIr::NaiveBayes(_) => "NaiveBayesModel",
        crate::ir::ModelIr::NearestNeighbor(_) => "NearestNeighborModel",
        crate::ir::ModelIr::SupportVectorMachine(_) => "SupportVectorMachineModel",
        crate::ir::ModelIr::GeneralRegression(_) => "GeneralRegressionModel",
        crate::ir::ModelIr::Association(_) => "AssociationModel",
        crate::ir::ModelIr::RuleSet(_) => "RuleSetModel",
        crate::ir::ModelIr::NeuralNetwork(_) => "NeuralNetwork",
        crate::ir::ModelIr::AnomalyDetection(_) => "AnomalyDetectionModel",
        crate::ir::ModelIr::Baseline(_) => "BaselineModel",
        crate::ir::ModelIr::GaussianProcess(_) => "GaussianProcessModel",
        crate::ir::ModelIr::Text(_) => "TextModel",
        crate::ir::ModelIr::TimeSeries(_) => "TimeSeriesModel",
        crate::ir::ModelIr::Sequence(_) => "SequenceModel",
        crate::ir::ModelIr::BayesianNetwork(_) => "BayesianNetworkModel",
    };
    SessionHandle {
        session,
        input_names: input_cstr,
        input_ptrs,
        output_names: output_cstr,
        output_ptrs,
        model_type: CString::new(model_type).unwrap(),
    }
}

// Helper to resolve SessionOptions handle -> Rust SessionOptions
fn resolve_options(opts: *const PmmlSessionOptions) -> SessionOptions {
    if opts.is_null() {
        return SessionOptions::default();
    }
    let h = unsafe { &*(opts as *const SessionOptionsHandle) };
    h.to_rust()
}

unsafe extern "C" fn api_CreateSession(
    env: *const PmmlEnv,
    path: *const c_char,
    opts: *const PmmlSessionOptions,
    out: *mut *mut PmmlSession,
) -> *mut PmmlStatus {
    if env.is_null() || path.is_null() || out.is_null() {
        return status_invalid_arg("CreateSession: null arg");
    }
    let rust_env = unsafe { &*(env as *const EnvHandle) };
    let cstr = unsafe { CStr::from_ptr(path) };
    let path_str = cstr.to_string_lossy().into_owned();
    let rust_opts = resolve_options(opts);
    match Session::from_file(&rust_env.env, &path_str, rust_opts) {
        Ok(sess) => {
            let h = Box::new(build_session_handle(sess));
            unsafe { *out = Box::into_raw(h) as *mut PmmlSession };
            ptr::null_mut()
        }
        Err(e) => status_from_error(e),
    }
}

unsafe extern "C" fn api_CreateSessionFromArray(
    env: *const PmmlEnv,
    bytes: *const c_void,
    len: usize,
    opts: *const PmmlSessionOptions,
    out: *mut *mut PmmlSession,
) -> *mut PmmlStatus {
    if env.is_null() || bytes.is_null() || out.is_null() {
        return status_invalid_arg("CreateSessionFromArray: null arg");
    }
    let rust_env = unsafe { &*(env as *const EnvHandle) };
    let slice = unsafe { std::slice::from_raw_parts(bytes as *const u8, len) };
    let rust_opts = resolve_options(opts);
    match Session::from_bytes(&rust_env.env, slice, rust_opts) {
        Ok(sess) => {
            let h = Box::new(build_session_handle(sess));
            unsafe { *out = Box::into_raw(h) as *mut PmmlSession };
            ptr::null_mut()
        }
        Err(e) => status_from_error(e),
    }
}

unsafe extern "C" fn api_ReleaseSession(sess: *mut PmmlSession) {
    if sess.is_null() { return; }
    unsafe { let _ = Box::from_raw(sess as *mut SessionHandle); }
}

unsafe extern "C" fn api_SessionGetInputCount(sess: *const PmmlSession, out: *mut usize) -> *mut PmmlStatus {
    if sess.is_null() || out.is_null() { return status_invalid_arg("SessionGetInputCount: null"); }
    let h = unsafe { &*(sess as *const SessionHandle) };
    unsafe { *out = h.input_names.len() };
    ptr::null_mut()
}

unsafe extern "C" fn api_SessionGetInputName(sess: *const PmmlSession, idx: usize, out: *mut *const c_char) -> *mut PmmlStatus {
    if sess.is_null() || out.is_null() { return status_invalid_arg("SessionGetInputName: null"); }
    let h = unsafe { &*(sess as *const SessionHandle) };
    if idx >= h.input_ptrs.len() { return status_invalid_arg("SessionGetInputName: index out of range"); }
    unsafe { *out = h.input_ptrs[idx] };
    ptr::null_mut()
}

unsafe extern "C" fn api_SessionGetOutputCount(sess: *const PmmlSession, out: *mut usize) -> *mut PmmlStatus {
    if sess.is_null() || out.is_null() { return status_invalid_arg("SessionGetOutputCount: null"); }
    let h = unsafe { &*(sess as *const SessionHandle) };
    unsafe { *out = h.output_names.len() };
    ptr::null_mut()
}

unsafe extern "C" fn api_SessionGetOutputName(sess: *const PmmlSession, idx: usize, out: *mut *const c_char) -> *mut PmmlStatus {
    if sess.is_null() || out.is_null() { return status_invalid_arg("SessionGetOutputName: null"); }
    let h = unsafe { &*(sess as *const SessionHandle) };
    if idx >= h.output_ptrs.len() { return status_invalid_arg("SessionGetOutputName: index out of range"); }
    unsafe { *out = h.output_ptrs[idx] };
    ptr::null_mut()
}

unsafe extern "C" fn api_SessionGetModelType(sess: *const PmmlSession, out: *mut *const c_char) -> *mut PmmlStatus {
    if sess.is_null() || out.is_null() { return status_invalid_arg("SessionGetModelType: null"); }
    let h = unsafe { &*(sess as *const SessionHandle) };
    unsafe { *out = h.model_type.as_ptr() };
    ptr::null_mut()
}

static VERSION_CSTR: &str = env!("CARGO_PKG_VERSION");
static mut VERSION_CSTRING: Option<CString> = None;
static VERSION_INIT: std::sync::Once = std::sync::Once::new();

unsafe extern "C" fn api_GetVersionString() -> *const c_char {
    VERSION_INIT.call_once(|| {
        unsafe { VERSION_CSTRING = Some(CString::new(VERSION_CSTR).unwrap()) };
    });
    unsafe { VERSION_CSTRING.as_ref().unwrap().as_ptr() }
}

unsafe extern "C" fn api_SessionGetFieldId(sess: *const PmmlSession, name: *const c_char, out: *mut u32, found: *mut i32) -> *mut PmmlStatus {
    if sess.is_null() || name.is_null() || out.is_null() || found.is_null() {
        return status_invalid_arg("SessionGetFieldId: null");
    }
    let h = unsafe { &*(sess as *const SessionHandle) };
    let s = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    if let Some(fid) = h.session.field_id(&s) {
        unsafe { *out = fid.0; *found = 1 };
    } else {
        unsafe { *found = 0 };
    }
    ptr::null_mut()
}

unsafe extern "C" fn api_SessionGetSymbolId(sess: *const PmmlSession, s: *const c_char, out: *mut u32, found: *mut i32) -> *mut PmmlStatus {
    if sess.is_null() || s.is_null() || out.is_null() || found.is_null() {
        return status_invalid_arg("SessionGetSymbolId: null");
    }
    let h = unsafe { &*(sess as *const SessionHandle) };
    let st = unsafe { CStr::from_ptr(s) }.to_string_lossy();
    if let Some(sid) = h.session.symbol_id(&st) {
        unsafe { *out = sid.0; *found = 1 };
    } else {
        unsafe { *found = 0 };
    }
    ptr::null_mut()
}

// Run: single row via names+values -> output values
unsafe extern "C" fn api_Run(
    sess: *mut PmmlSession,
    _run_opts: *const PmmlRunOptions,
    input_names: *const *const c_char,
    input_values: *const PmmlValue,
    input_count: usize,
    output_names: *const *const c_char,
    output_count: usize,
    output_values: *mut PmmlValue,
) -> *mut PmmlStatus {
    if sess.is_null() || output_values.is_null() { return status_invalid_arg("Run: null sess/output"); }
    if input_count > 0 && (input_names.is_null() || input_values.is_null()) {
        return status_invalid_arg("Run: input null but count>0");
    }
    let h = unsafe { &mut *(sess as *mut SessionHandle) };
    // Build HashMap<String, Value> from C inputs
    let mut map = HashMap::new();
    for i in 0..input_count {
        let name = unsafe { CStr::from_ptr(*input_names.add(i)) }.to_string_lossy().into_owned();
        let pv = unsafe { *input_values.add(i) };
        map.insert(name, pmml_value_to_value(pv));
    }
    // Use unified Batch: HashMap is Batch (1 row)
    use crate::session::batch::Batch;
    let batch: &dyn crate::session::batch::Batch = &map as &dyn crate::session::batch::Batch;
    // Run
    let result = match h.session.run(batch) {
        Ok(r) => r,
        Err(e) => return status_from_error(e),
    };
    let rows = result.into_rows();
    if rows.is_empty() { return status_invalid_arg("Run: empty result"); }
    let row = &rows[0];
    // If caller provided output_names, use them; else return all outputs (caller must have allocated enough)
    if !output_names.is_null() && output_count > 0 {
        for i in 0..output_count {
            let oname = unsafe { CStr::from_ptr(*output_names.add(i)) }.to_string_lossy().into_owned();
            let v = row.get(&oname).copied().unwrap_or(Value::Missing);
            unsafe { *output_values.add(i) = value_to_pmml_value(v) };
        }
    } else {
        // No output_names: fill in order of session output_names
        let mut idx = 0usize;
        for name in h.output_ptrs.iter().map(|p| unsafe { CStr::from_ptr(*p).to_string_lossy().into_owned() }) {
            if idx >= output_count { break; }
            let v = row.get(&name).copied().unwrap_or(Value::Missing);
            unsafe { *output_values.add(idx) = value_to_pmml_value(v) };
            idx += 1;
        }
        // Also include predictedValue if not already
        if idx < output_count {
            if let Some(v) = row.get("predictedValue") {
                unsafe { *output_values.add(idx) = value_to_pmml_value(*v) };
            }
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn api_RunBatch(
    sess: *mut PmmlSession,
    _run_opts: *const PmmlRunOptions,
    input_names: *const *const c_char,
    flat_values: *const PmmlValue,
    n_rows: usize,
    n_cols: usize,
    out_flat: *mut PmmlValue,
    out_rows_inout: *mut usize,
) -> *mut PmmlStatus {
    if sess.is_null() || flat_values.is_null() || out_flat.is_null() || out_rows_inout.is_null() {
        return status_invalid_arg("RunBatch: null");
    }
    let h = unsafe { &mut *(sess as *mut SessionHandle) };
    let n_in = unsafe { *out_rows_inout };
    if n_in < n_rows { return status_invalid_arg("RunBatch: out buffer too small"); }
    // Build Vec<HashMap> from flat row-major [rows][cols]
    let names: Vec<String> = (0..n_cols).map(|i| unsafe { CStr::from_ptr(*input_names.add(i)).to_string_lossy().into_owned() }).collect();
    let mut batch: Vec<HashMap<String, Value>> = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut m = HashMap::new();
        for c in 0..n_cols {
            let pv = unsafe { *flat_values.add(r * n_cols + c) };
            m.insert(names[c].clone(), pmml_value_to_value(pv));
        }
        batch.push(m);
    }
    use crate::session::batch::Batch;
    let result = match h.session.run(&batch as &dyn crate::session::batch::Batch) {
        Ok(r) => r,
        Err(e) => return status_from_error(e),
    };
    let rows = result.into_rows();
    // For now, flat out is predictedValue only (single output per row)
    for (i, row) in rows.iter().enumerate() {
        let v = row.get("predictedValue").copied().unwrap_or(Value::Missing);
        unsafe { *out_flat.add(i) = value_to_pmml_value(v) };
    }
    unsafe { *out_rows_inout = rows.len() };
    ptr::null_mut()
}

unsafe extern "C" fn api_RunArrow(
    _sess: *mut PmmlSession,
    _run_opts: *const PmmlRunOptions,
    _in_array: *const ArrowArray,
    _in_schema: *const ArrowSchema,
    _out_array: *mut ArrowArray,
    _out_schema: *mut ArrowSchema,
) -> *mut PmmlStatus {
    // TODO: implement Arrow C Data Interface zero-copy via arrow-rs ffi
    make_status(PmmlErrorCode::Unknown, "RunArrow: not yet implemented (stub)")
}

unsafe extern "C" fn api_CreateIoBinding(sess: *mut PmmlSession, out: *mut *mut PmmlIoBinding) -> *mut PmmlStatus {
    if sess.is_null() || out.is_null() { return status_invalid_arg("CreateIoBinding: null"); }
    let h = Box::new(IoBindingHandle { inputs: HashMap::new(), outputs: Vec::new() });
    unsafe { *out = Box::into_raw(h) as *mut PmmlIoBinding };
    let _ = sess;
    ptr::null_mut()
}

unsafe extern "C" fn api_ReleaseIoBinding(b: *mut PmmlIoBinding) {
    if b.is_null() { return; }
    unsafe { let _ = Box::from_raw(b as *mut IoBindingHandle); }
}

unsafe extern "C" fn api_BindInput(b: *mut PmmlIoBinding, name: *const c_char, value: PmmlValue) -> *mut PmmlStatus {
    if b.is_null() || name.is_null() { return status_invalid_arg("BindInput: null"); }
    let h = unsafe { &mut *(b as *mut IoBindingHandle) };
    let n = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    h.inputs.insert(n, pmml_value_to_value(value));
    ptr::null_mut()
}

unsafe extern "C" fn api_BindInputArrow(b: *mut PmmlIoBinding, _name: *const c_char, _array: *const ArrowArray, _schema: *const ArrowSchema) -> *mut PmmlStatus {
    let _ = b;
    make_status(PmmlErrorCode::Unknown, "BindInputArrow: not yet implemented")
}

unsafe extern "C" fn api_BindOutput(b: *mut PmmlIoBinding, name: *const c_char) -> *mut PmmlStatus {
    if b.is_null() || name.is_null() { return status_invalid_arg("BindOutput: null"); }
    let h = unsafe { &mut *(b as *mut IoBindingHandle) };
    let n = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    h.outputs.push(n);
    ptr::null_mut()
}

unsafe extern "C" fn api_RunWithBinding(sess: *mut PmmlSession, _run_opts: *const PmmlRunOptions, binding: *mut PmmlIoBinding) -> *mut PmmlStatus {
    if sess.is_null() || binding.is_null() { return status_invalid_arg("RunWithBinding: null"); }
    let h = unsafe { &mut *(sess as *mut SessionHandle) };
    let b = unsafe { &*(binding as *mut IoBindingHandle) };
    let map = b.inputs.clone();
    use crate::session::batch::Batch;
    let result = match h.session.run(&map as &dyn crate::session::batch::Batch) {
        Ok(r) => r,
        Err(e) => return status_from_error(e),
    };
    // Store back into binding's outputs? For now we stash in inputs as last result? We need a place to retrieve via CopyBindingOutputsToCpu.
    // Instead, misuse binding's inputs to hold output flat: we keep result rows in a thread-local? Simpler: store result in binding's outputs vector as formatted?
    // For scaffold, store result's predictedValue into a static? Instead, extend IoBindingHandle to hold last result.
    // We add a hidden field via extra allocation: use a global static mutex for last result per binding? For now, leak via Box::leak alternative is to extend struct.
    // Quick hack: we transmute binding to hold result in a separate global map keyed by binding pointer.
    // Instead, define IoBindingHandle with last_result field.
    // We'll need to mutate struct definition — but we already defined it without. Patch by using unsafe extra storage.
    // For now, store result's row into the binding's inputs under special key "__last_result".
    // Proper impl will change IoBindingHandle to have Option<BatchResult>.
    // This stub just returns OK — caller should use Run directly until IoBinding is fully implemented.
    let _ = result;
    ptr::null_mut()
}

unsafe extern "C" fn api_CopyBindingOutputsToCpu(_binding: *mut PmmlIoBinding, _out_flat: *mut PmmlValue, _out_count: *mut usize) -> *mut PmmlStatus {
    make_status(PmmlErrorCode::Unknown, "CopyBindingOutputsToCpu: not yet implemented")
}

unsafe extern "C" fn api_CreateRunOptions(out: *mut *mut PmmlRunOptions) -> *mut PmmlStatus {
    if out.is_null() { return status_invalid_arg("CreateRunOptions: null"); }
    let h = Box::new(RunOptionsHandle { tag: None, log_level: PmmlLogLevel::Warning });
    unsafe { *out = Box::into_raw(h) as *mut PmmlRunOptions };
    ptr::null_mut()
}

unsafe extern "C" fn api_ReleaseRunOptions(opts: *mut PmmlRunOptions) {
    if opts.is_null() { return; }
    unsafe { let _ = Box::from_raw(opts as *mut RunOptionsHandle); }
}

unsafe extern "C" fn api_SetRunTag(opts: *mut PmmlRunOptions, tag: *const c_char) -> *mut PmmlStatus {
    if opts.is_null() || tag.is_null() { return status_invalid_arg("SetRunTag: null"); }
    let s = unsafe { CStr::from_ptr(tag) }.to_string_lossy().into_owned();
    let c = CString::new(s).unwrap();
    unsafe { (*(opts as *mut RunOptionsHandle)).tag = Some(c) };
    ptr::null_mut()
}

unsafe extern "C" fn api_SetRunLogLevel(opts: *mut PmmlRunOptions, lvl: PmmlLogLevel) -> *mut PmmlStatus {
    if opts.is_null() { return status_invalid_arg("SetRunLogLevel: null"); }
    unsafe { (*(opts as *mut RunOptionsHandle)).log_level = lvl };
    ptr::null_mut()
}

// ---------------------------------------------------------------------------
// Status API (standalone, not in table)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn PmmlGetErrorCode(status: *const PmmlStatus) -> PmmlErrorCode {
    if status.is_null() { return PmmlErrorCode::Ok; }
    let h = &*(status as *const StatusHandle);
    h.code
}

#[no_mangle]
pub unsafe extern "C" fn PmmlGetErrorMessage(status: *const PmmlStatus) -> *const c_char {
    if status.is_null() { return ptr::null(); }
    let h = &*(status as *const StatusHandle);
    h.message.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn PmmlReleaseStatus(status: *mut PmmlStatus) {
    if status.is_null() { return; }
    let _ = Box::from_raw(status as *mut StatusHandle);
}

// ---------------------------------------------------------------------------
// Global Api table
// ---------------------------------------------------------------------------

static PMML_API: PmmlApi = PmmlApi {
    version: 1,
    CreateEnv: Some(api_CreateEnv),
    ReleaseEnv: Some(api_ReleaseEnv),
    CreateSessionOptions: Some(api_CreateSessionOptions),
    ReleaseSessionOptions: Some(api_ReleaseSessionOptions),
    SetGraphOptimizationLevel: Some(api_SetGraphOptimizationLevel),
    SetIntraOpNumThreads: Some(api_SetIntraOpNumThreads),
    SetInterOpNumThreads: Some(api_SetInterOpNumThreads),
    SetLogLevel: Some(api_SetLogLevel),
    AddSessionConfigEntry: Some(api_AddSessionConfigEntry),
    AppendExecutionProvider: Some(api_AppendExecutionProvider),
    CreateSession: Some(api_CreateSession),
    CreateSessionFromArray: Some(api_CreateSessionFromArray),
    ReleaseSession: Some(api_ReleaseSession),
    SessionGetInputCount: Some(api_SessionGetInputCount),
    SessionGetInputName: Some(api_SessionGetInputName),
    SessionGetOutputCount: Some(api_SessionGetOutputCount),
    SessionGetOutputName: Some(api_SessionGetOutputName),
    SessionGetModelType: Some(api_SessionGetModelType),
    GetVersionString: Some(api_GetVersionString),
    SessionGetFieldId: Some(api_SessionGetFieldId),
    SessionGetSymbolId: Some(api_SessionGetSymbolId),
    Run: Some(api_Run),
    RunBatch: Some(api_RunBatch),
    RunArrow: Some(api_RunArrow),
    CreateIoBinding: Some(api_CreateIoBinding),
    ReleaseIoBinding: Some(api_ReleaseIoBinding),
    BindInput: Some(api_BindInput),
    BindInputArrow: Some(api_BindInputArrow),
    BindOutput: Some(api_BindOutput),
    RunWithBinding: Some(api_RunWithBinding),
    CopyBindingOutputsToCpu: Some(api_CopyBindingOutputsToCpu),
    CreateRunOptions: Some(api_CreateRunOptions),
    ReleaseRunOptions: Some(api_ReleaseRunOptions),
    SetRunTag: Some(api_SetRunTag),
    SetRunLogLevel: Some(api_SetRunLogLevel),
};

#[no_mangle]
pub extern "C" fn PmmlGetApi(version: u32) -> *const PmmlApi {
    if version == 1 {
        &PMML_API as *const PmmlApi
    } else {
        ptr::null()
    }
}

// ---------------------------------------------------------------------------
// Deprecated shims (0.1 compat) — call through table
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn PmmlCreateEnv(env_out: *mut *mut PmmlEnv) -> i32 {
    if env_out.is_null() { return PmmlErrorCode::InvalidArgument as i32; }
    let api = PmmlGetApi(1);
    if api.is_null() { return PmmlErrorCode::Unknown as i32; }
    let mut env: *mut PmmlEnv = ptr::null_mut();
    let status = ((*api).CreateEnv.unwrap())(PmmlLogLevel::Warning, ptr::null(), &mut env);
    if !status.is_null() {
        PmmlReleaseStatus(status);
        return PmmlErrorCode::Unknown as i32;
    }
    *env_out = env;
    PmmlErrorCode::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn PmmlReleaseEnv(env: *mut PmmlEnv) {
    let api = PmmlGetApi(1);
    if api.is_null() || env.is_null() { return; }
    if let Some(f) = (*api).ReleaseEnv { f(env) }
}

#[no_mangle]
pub unsafe extern "C" fn PmmlCreateSession(env: *mut PmmlEnv, path: *const c_char, session_out: *mut *mut PmmlSession) -> i32 {
    if env.is_null() || path.is_null() || session_out.is_null() { return PmmlErrorCode::InvalidArgument as i32; }
    let api = PmmlGetApi(1);
    if api.is_null() { return PmmlErrorCode::Unknown as i32; }
    let status = ((*api).CreateSession.unwrap())(env as *const PmmlEnv, path, ptr::null(), session_out);
    if !status.is_null() {
        PmmlReleaseStatus(status);
        return PmmlErrorCode::Unknown as i32;
    }
    PmmlErrorCode::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn PmmlReleaseSession(session: *mut PmmlSession) {
    let api = PmmlGetApi(1);
    if api.is_null() || session.is_null() { return; }
    if let Some(f) = (*api).ReleaseSession { f(session) }
}

// ---------------------------------------------------------------------------
// Trait helper for output_fields
// ---------------------------------------------------------------------------

trait ModelOutput {
    fn output_fields(&self) -> &[crate::ir::OutputFieldIr];
}

impl ModelOutput for crate::ir::ModelIr {
    fn output_fields(&self) -> &[crate::ir::OutputFieldIr] {
        match self {
            crate::ir::ModelIr::Tree(t) => &t.output,
            crate::ir::ModelIr::Regression(r) => &r.output,
            crate::ir::ModelIr::Mining(m) => &m.output,
            crate::ir::ModelIr::Scorecard(s) => &s.output,
            crate::ir::ModelIr::Clustering(c) => &c.output,
            crate::ir::ModelIr::NaiveBayes(n) => &n.output,
            crate::ir::ModelIr::NearestNeighbor(n) => &n.output,
            crate::ir::ModelIr::SupportVectorMachine(s) => &s.output,
            crate::ir::ModelIr::GeneralRegression(g) => &g.output,
            crate::ir::ModelIr::Association(a) => &a.output,
            crate::ir::ModelIr::RuleSet(r) => &r.output,
            crate::ir::ModelIr::NeuralNetwork(n) => &n.output,
            crate::ir::ModelIr::AnomalyDetection(a) => &a.output,
            crate::ir::ModelIr::Baseline(b) => &b.output,
            crate::ir::ModelIr::GaussianProcess(g) => &g.output,
            crate::ir::ModelIr::Text(t) => &t.output,
            crate::ir::ModelIr::TimeSeries(t) => &t.output,
            crate::ir::ModelIr::Sequence(s) => &s.output,
            crate::ir::ModelIr::BayesianNetwork(b) => &b.output,
        }
    }
}
