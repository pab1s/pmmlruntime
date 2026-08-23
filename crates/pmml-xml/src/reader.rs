//! Hardened XML reader — mirrors Java `SAXUtil` security.

use pmml_core::error::PmmlError;
use pmml_core::Result;
use quick_xml::events::Event;
use quick_xml::Reader;

const MAX_DEPTH: usize = 128;
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
        let mut xml = String::from("<PMML>");
        for _ in 0..130 {
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
        assert!(err.is_some(), "should hit depth limit");
    }
}
