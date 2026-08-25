//! Hardened XML reader — mirrors Java `SAXUtil` security.

use pmml_core::error::PmmlError;
use pmml_core::Result;
use quick_xml::events::Event;
use quick_xml::Reader;

const MAX_DEPTH: usize = 512;
const MAX_FILE_BYTES: usize = 100 * 1024 * 1024; // 100 MB

/// Hardened PMML reader. DTD / external entities disabled, depth/file limits enforced.
pub struct PmmlReader<'a> {
    reader: Reader<&'a [u8]>,
    depth: usize,
}

impl<'a> PmmlReader<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() > MAX_FILE_BYTES {
            return Err(PmmlError::ValidationError(format!(
                "PMML file too large: {} > {}",
                bytes.len(),
                MAX_FILE_BYTES
            )));
        }
        let mut reader = Reader::from_reader(bytes);
        reader.config_mut().trim_text(true);
        reader.config_mut().expand_empty_elements = true;
        // quick-xml 0.37 does not expand entities by default; we keep it disabled
        // No DTD handling — external entities are not resolved
        Ok(Self { reader, depth: 0 })
    }

    /// Read next event, tracking depth.
    pub fn read_event(&mut self) -> Result<Event<'a>> {
        let mut buf = Vec::new();
        loop {
            let ev = self
                .reader
                .read_event_into(&mut buf)
                .map_err(|e| PmmlError::ParseError {
                    context: "xml".into(),
                    message: e.to_string(),
                })?;
            match &ev {
                Event::Start(_) => {
                    self.depth += 1;
                    if self.depth > MAX_DEPTH {
                        return Err(PmmlError::ValidationError(format!(
                            "XML depth exceeds limit {MAX_DEPTH}"
                        )));
                    }
                    return Ok(ev.into_owned());
                }
                Event::End(_) => {
                    self.depth = self.depth.saturating_sub(1);
                    return Ok(ev.into_owned());
                }
                Event::Empty(_) => {
                    // depth doesn't increase for empty
                    return Ok(ev.into_owned());
                }
                Event::Eof => return Ok(Event::Eof),
                Event::Decl(_) | Event::Comment(_) | Event::PI(_) => {
                    buf.clear();
                    continue;
                }
                _ => {
                    return Ok(ev.into_owned());
                }
            }
        }
    }

    pub fn config_mut(&mut self) -> &mut quick_xml::reader::Config {
        self.reader.config_mut()
    }
}

// For direct quick-xml usage where we control depth ourselves, expose inner
pub fn new_reader(bytes: &[u8]) -> Result<Reader<&[u8]>> {
    if bytes.len() > MAX_FILE_BYTES {
        return Err(PmmlError::ValidationError(format!(
            "PMML file too large: {} > {}",
            bytes.len(),
            MAX_FILE_BYTES
        )));
    }
    let mut r = Reader::from_reader(bytes);
    r.config_mut().trim_text(true);
    r.config_mut().expand_empty_elements = true;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_limit() {
        // MAX_DEPTH is 512 per plan D4
        let mut xml = String::from("<PMML>");
        for _ in 0..520 {
            xml.push_str("<a>");
        }
        let bytes = xml.into_bytes();
        let mut r = PmmlReader::from_bytes(&bytes).unwrap();
        let mut err = None;
        loop {
            match r.read_event() {
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(e) => {
                    err = Some(e.to_string());
                    break;
                }
            }
        }
        assert!(err.is_some(), "should hit depth limit 512");
        assert!(err.unwrap().contains("depth"));
    }

    #[test]
    fn depth_limit_ok_at_511() {
        let mut xml = String::from("<PMML>");
        for _ in 0..511 {
            xml.push_str("<a>");
        }
        for _ in 0..511 {
            xml.push_str("</a>");
        }
        xml.push_str("</PMML>");
        let bytes = xml.into_bytes();
        let mut r = PmmlReader::from_bytes(&bytes).unwrap();
        let mut got_error = false;
        loop {
            match r.read_event() {
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(_) => {
                    got_error = true;
                    break;
                }
            }
        }
        assert!(!got_error, "511 depth should be ok");
    }

    #[test]
    fn xxe_blocked() {
        // XXE payload — external entity should not be resolved, not reading file
        let xxe = r#"<?xml version="1.0"?>
<!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
<PMML version="4.4"><DataDictionary><DataField name="f" dataType="string" optype="categorical"/></DataDictionary>&xxe;</PMML>"#;
        // new_reader should not expand entity; quick-xml returns text "&xxe;" or error, but must not contain passwd content
        let bytes = xxe.as_bytes();
        let r = new_reader(bytes);
        assert!(r.is_ok(), "reader should be created (XXE DTD not expanded)");
        let reader = r.unwrap();
        // Actually unmarshal should handle XXE gracefully: either error or ignore entity
        // We verify that file content is not leaked: the raw bytes should not be read as entity expansion
        // Quick-xml 0.37 does not resolve external entities; test that we don't get passwd
        let mut inner = reader;
        let mut buf = Vec::new();
        let mut found_passwd = false;
        loop {
            match inner.read_event_into(&mut buf) {
                Ok(Event::Text(t)) => {
                    let txt = t.unescape().unwrap_or_default().into_owned();
                    if txt.contains("root:") {
                        found_passwd = true;
                    }
                }
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(_) => break,
            }
            buf.clear();
        }
        assert!(!found_passwd, "XXE should not leak file content");
    }
}
