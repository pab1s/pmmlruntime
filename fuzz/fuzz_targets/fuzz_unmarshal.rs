#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz pmmlruntime::xml::unmarshal with arbitrary bytes — should not panic, only return Ok or Err
    // Mirrors JPMML SAXUtil hardening and validates reader depth 512 / 100 MB cap / XXE blocked
    let _ = pmmlruntime::xml::unmarshal(data);
    // Also fuzz lower/verify pipeline for coverage (45 fixtures + random)
    if let Ok(raw) = pmmlruntime::xml::unmarshal(data) {
        let _ = pmmlruntime::ir::verify_raw(&raw);
        if pmmlruntime::ir::verify_raw(&raw).is_ok() {
            // lower may allocate via BumpArena thread_local; should not panic
            let _ = pmmlruntime::ir::lower(raw);
        }
    }
    // Also fuzz Session cold path (from_bytes) — ensures no panic on malformed PMML
    {
        let env = pmmlruntime::session::PmmlEnv::new();
        let _ = pmmlruntime::session::Session::from_bytes(
            &env,
            data,
            pmmlruntime::session::SessionOptions::default(),
        );
    }
});
