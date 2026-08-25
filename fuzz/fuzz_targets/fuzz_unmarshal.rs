#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz pmml_xml::unmarshal with arbitrary bytes — should not panic, only return Ok or Err
    // This mirrors JPMML `SAXUtil` hardening and validates reader depth/XXE handling
    let _ = pmml_xml::unmarshal(data);
    // Also fuzz lower/verify pipeline for coverage
    if let Ok(raw) = pmml_xml::unmarshal(data) {
        let _ = pmml_ir::verify_raw(&raw);
        if pmml_ir::verify_raw(&raw).is_ok() {
            let _ = pmml_ir::lower(raw);
        }
    }
});
