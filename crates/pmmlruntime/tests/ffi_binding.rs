use pmmlruntime::ffi::{PmmlLogLevel, PmmlValue, PmmlGetApi};
use std::ffi::CString;
#[test]
fn ffi_iobinding_roundtrip() {
    unsafe {
        let api = &*PmmlGetApi(1);
        assert_eq!(api.version, 1);
        let log_id = CString::new("test").unwrap();
        let mut env: *mut pmmlruntime::ffi::PmmlEnv = std::ptr::null_mut();
        assert!((api.CreateEnv.unwrap())(PmmlLogLevel::Warning, log_id.as_ptr(), &mut env).is_null());
        let bytes = std::fs::read("bench/pmml/DecisionTreeIris.pmml")
            .or_else(|_| std::fs::read("../../bench/pmml/DecisionTreeIris.pmml"))
            .unwrap();
        let mut sess: *mut pmmlruntime::ffi::PmmlSession = std::ptr::null_mut();
        assert!((api.CreateSessionFromArray.unwrap())(env as *const _, bytes.as_ptr() as *const _, bytes.len(), std::ptr::null(), &mut sess).is_null());
        let mut b: *mut pmmlruntime::ffi::PmmlIoBinding = std::ptr::null_mut();
        assert!((api.CreateIoBinding.unwrap())(sess, &mut b).is_null());
        let k = CString::new("Petal.Length").unwrap();
        let v = PmmlValue::continuous(1.4);
        assert!((api.BindInput.unwrap())(b, k.as_ptr(), v).is_null());
        let ko = CString::new("predictedValue").unwrap();
        assert!((api.BindOutput.unwrap())(b, ko.as_ptr()).is_null());
        assert!((api.RunWithBinding.unwrap())(sess, std::ptr::null(), b).is_null());
        let mut out = [PmmlValue::missing(); 4];
        let mut n = out.len();
        assert!((api.CopyBindingOutputsToCpu.unwrap())(b, out.as_mut_ptr(), &mut n).is_null());
        assert!(n >= 1);
        (api.ReleaseIoBinding.unwrap())(b);
        (api.ReleaseSession.unwrap())(sess);
        (api.ReleaseEnv.unwrap())(env);
    }
}
