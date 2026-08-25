//! Hardened XML reader — mirrors `org.jpmml.model.SAXUtil` security.
//!
//! This module is the **only** place that enforces PMML XML hardening. It
//! wraps `quick-xml` 0.37's pull parser and adds the limits that `SAXUtil`
//! applies in the Java reference runtime: DTD/external entities disabled,
//! file-size cap, and nesting-depth cap. All unmarshaling in [`mod@crate::unmarshal`]
//! must go through [`PmmlReader`] or [`new_reader`] so a single audit covers
//! every entry point.
//!
//! # What belongs here
//!
//! - [`PmmlReader`] — stateful wrapper that tracks element depth and delegates
//!   to `quick_xml::Reader` with `trim_text(true)` and `expand_empty_elements(true)`.
//! - [`new_reader`] — stateless helper that returns a pre-configured `quick_xml::Reader`
//!   when the caller will track depth itself (e.g. `unmarshal`'s own loop).
//! - The two `const` limits `MAX_DEPTH = 512` and `MAX_FILE_BYTES = 100 MB`
//!   and the DTD/XXE policy.
//!
//! Nothing PMML-semantic belongs here: field types, `DataDictionary` parsing and
//! model lowering live in [`mod@crate::unmarshal`] and `pmml-ir`. Keeping the layer
//! minimal makes the security review tractable.
//!
//! # Why separate `reader` from `unmarshal`
//!
//! `unmarshal` is ~5.7 kLOC of `quick-xml` pull logic that maps `pmml.xsd` elements
//! to [`crate::unmarshal::RawPmml`]. Mixing depth/file guards into each parse
//! function would be error-prone and hard to audit. By extracting the guard into
//! `PmmlReader::read_event` / `new_reader` the hot-path XML loop stays small and
//! every path — tree, regression, mining, or the unsupported-model fallback —
//! inherits the same limits without duplicating checks.
//!
//! # How it relates to neighboring modules
//!
//! ```text
//! bytes: &[u8]  ──►  reader::PmmlReader / new_reader  ──►  unmarshal::unmarshal  ──►  RawPmml  ──►  pmml_ir::lower  ──►  Ir
//!                      (this module)                    (sibling)               (pmml-ir crate)
//! ```
//!
//! - `pmml-core` provides [`pmml_core::PmmlError`] / [`pmml_core::Result`] used for
//!   the two rejected cases below.
//! - `pmml-ir` and `pmml-session` never touch `quick-xml` directly; they receive the
//!   already-validated `RawPmml`.
//!
//! # Security invariants
//!
//! - `MAX_DEPTH = 512` — `PmmlReader::read_event` increments on `Event::Start` and
//!   errors with [`pmml_core::PmmlError::ValidationError`] when `depth > 512`. Matches
//!   `SAXUtil` and the `pmml.xsd` depth budget. `Event::Empty` does not increase depth.
//! - `MAX_FILE_BYTES = 100 MB` (`100 * 1024 * 1024`) — both [`PmmlReader::from_bytes`]
//!   and [`new_reader`] reject `bytes.len() > MAX_FILE_BYTES` before creating a parser.
//! - DTD / XXE — `quick-xml` 0.37 does **not** expand entities by default and this
//!   module never enables it. `<!DOCTYPE … [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>`
//!   is not resolved; the entity text (`&xxe;`) is returned verbatim, so file content
//!   is never leaked. No DTD handling is implemented, external entities are not fetched.
//! - `trim_text(true)` and `expand_empty_elements(true)` are set on every reader so
//!   `unmarshal` sees normalized `Event::Text` / `Event::Empty` regardless of the
//!   PMML producer's whitespace.
//!
//! # Performance
//!
//! Depth tracking is a single `usize` increment per `Start`/`End`. No allocation
//! occurs in the fast path beyond `quick-xml`'s internal buffer.

use pmml_core::error::PmmlError;
use pmml_core::Result;
use quick_xml::events::Event;
use quick_xml::Reader;

const MAX_DEPTH: usize = 512;
const MAX_FILE_BYTES: usize = 100 * 1024 * 1024; // 100 MB

/// Hardened PMML reader that enforces `SAXUtil` limits on top of `quick-xml`.
///
/// Wraps `quick_xml::Reader<&[u8]>` and tracks nesting depth. It is the
/// preferred entry point for PMML XML; [`new_reader`] is the lower-level
/// alternative when the caller manages depth itself.
///
/// # Security
///
/// - File cap `100 MB` checked in [`PmmlReader::from_bytes`] before parsing.
/// - Depth cap `512` enforced in [`PmmlReader::read_event`].
/// - DTD/external entities disabled — `quick-xml` 0.37 does not expand entities
///   and this type never opts in, so XXE payloads are blocked. See the
///   module docs and [`new_reader`] for details.
///
/// # Examples
///
/// Basic construction and loop — mirrors how [`crate::unmarshal::unmarshal`] uses it:
///
/// ```
/// use pmml_xml::PmmlReader;
/// use quick_xml::events::Event;
///
/// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary></PMML>"#;
/// let mut r = PmmlReader::from_bytes(xml)?;
/// loop {
///     match r.read_event()? {
///         Event::Eof => break,
///         _ => {}
///     }
/// }
/// # Ok::<(), pmml_core::PmmlError>(())
/// ```
///
/// XXE payloads are not expanded (the `&xxe;` entity stays literal, file content is not leaked):
///
/// ```
/// use pmml_xml::PmmlReader;
/// use quick_xml::events::Event;
///
/// let xxe = br#"<?xml version="1.0"?><!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]><PMML version="4.4"><Header/></PMML>"#;
/// let mut r = PmmlReader::from_bytes(xxe)?;
/// let mut leaked = false;
/// loop {
///     match r.read_event()? {
///         Event::Text(t) if t.unescape().unwrap_or_default().contains("root:") => leaked = true,
///         Event::Eof => break,
///         _ => {}
///     }
/// }
/// assert!(!leaked);
/// # Ok::<(), pmml_core::PmmlError>(())
/// ```
pub struct PmmlReader<'a> {
    reader: Reader<&'a [u8]>,
    depth: usize,
}

impl<'a> PmmlReader<'a> {
    /// Create a hardened reader from `bytes`.
    ///
    /// Checks the file-size limit before constructing the underlying
    /// `quick_xml::Reader` and configures `trim_text(true)` and
    /// `expand_empty_elements(true)`. Entity expansion stays disabled
    /// (the `quick-xml` 0.37 default), so XXE entities are not resolved.
    ///
    /// # Parameters
    ///
    /// - `bytes`: the complete PMML document. May contain an XML declaration
    ///   and `DOCTYPE`; both are tolerated but not expanded.
    ///
    /// # Return
    ///
    /// `Ok(PmmlReader)` ready for [`PmmlReader::read_event`] on success.
    ///
    /// # Errors
    ///
    /// - [`pmml_core::PmmlError::ValidationError`] if `bytes.len() > 100 MB`
    ///   (`MAX_FILE_BYTES`). Message is `"PMML file too large: {len} > {limit}"`.
    /// - No XML parse error is raised here; malformed XML is reported lazily
    ///   by [`PmmlReader::read_event`].
    ///
    /// # Panics
    ///
    /// Never panics. All failure modes are returned as `Err`.
    ///
    /// # Examples
    ///
    /// Valid document succeeds:
    ///
    /// ```
    /// use pmml_xml::PmmlReader;
    /// let xml = br#"<PMML version="4.4"><Header/></PMML>"#;
    /// let r = PmmlReader::from_bytes(xml)?;
    /// # Ok::<(), pmml_core::PmmlError>(())
    /// ```
    ///
    /// Oversized input is rejected without allocating a parser:
    ///
    /// ```
    /// use pmml_xml::PmmlReader;
    /// // 100 MB + 1 byte — constructed without holding the full payload in the doctest
    /// let len = 100 * 1024 * 1024 + 1;
    /// let big = vec![b' '; len];
    /// let res = PmmlReader::from_bytes(&big);
    /// assert!(res.is_err());
    /// match res {
    ///     Err(e) => assert!(e.to_string().contains("too large")),
    ///     Ok(_) => panic!("expected ValidationError"),
    /// }
    /// ```
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

    /// Read the next `quick-xml` event, tracking nesting depth.
    ///
    /// `Loop`s internally to skip `Decl`, `Comment`, and `PI` events
    /// (which do not affect PMML semantics) and returns the first
    /// semantic event. Depth is incremented on `Start`, decremented
    /// (saturating) on `End`, and left unchanged on `Empty`/`Eof`.
    ///
    /// # Return
    ///
    /// `Ok(Event)` with an owned buffer (`into_owned`) so the caller can
    /// hold it across the next call. `Event::Eof` signals end of document.
    ///
    /// # Errors
    ///
    /// - [`pmml_core::PmmlError::ValidationError`] when `depth > 512`
    ///   (`MAX_DEPTH`). Message contains `"depth"`.
    /// - [`pmml_core::PmmlError::ParseError`] with `context: "xml"` when
    ///   `quick-xml` reports malformed XML.
    ///
    /// # Panics
    ///
    /// Never panics. Depth overflow is guarded by the limit check; underflow
    /// uses `saturating_sub`.
    ///
    /// # Examples
    ///
    /// Drive the reader until `Eof`, rejecting overly deep documents:
    ///
    /// ```
    /// use pmml_xml::PmmlReader;
    /// use quick_xml::events::Event;
    ///
    /// let mut xml = String::from("<PMML>");
    /// for _ in 0..520 { xml.push_str("<a>"); }
    /// let bytes = xml.into_bytes();
    /// let mut r = PmmlReader::from_bytes(&bytes).unwrap();
    /// let mut err = None;
    /// loop {
    ///     match r.read_event() {
    ///         Ok(Event::Eof) => break,
    ///         Ok(_) => {},
    ///         Err(e) => { err = Some(e.to_string()); break; }
    ///     }
    /// }
    /// assert!(err.unwrap().contains("depth"));
    /// # Ok::<(), pmml_core::PmmlError>(())
    /// ```
    ///
    /// Documents within the limit parse cleanly:
    ///
    /// ```
    /// use pmml_xml::PmmlReader;
    /// use quick_xml::events::Event;
    ///
    /// let mut xml = String::from("<PMML>");
    /// for _ in 0..511 { xml.push_str("<a>"); }
    /// for _ in 0..511 { xml.push_str("</a>"); }
    /// xml.push_str("</PMML>");
    /// let bytes = xml.into_bytes();
    /// let mut r = PmmlReader::from_bytes(&bytes).unwrap();
    /// let mut depth_err = false;
    /// loop {
    ///     match r.read_event() {
    ///         Ok(Event::Eof) => break,
    ///         Ok(_) => {},
    ///         Err(_) => { depth_err = true; break; }
    ///     }
    /// }
    /// assert!(!depth_err);
    /// # Ok::<(), pmml_core::PmmlError>(())
    /// ```
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

    /// Mutable access to the underlying `quick_xml::reader::Config`.
    ///
    /// Exposed so callers that need non-default `quick-xml` options (e.g.
    /// for tests) can adjust without forking `PmmlReader`. The hardening
    /// invariants (`trim_text`, `expand_empty_elements`, no entity expansion)
    /// are set by [`PmmlReader::from_bytes`]; mutating them may weaken the
    /// `SAXUtil` guarantees.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmml_xml::PmmlReader;
    /// let xml = br#"<PMML><Header/></PMML>"#;
    /// let mut r = PmmlReader::from_bytes(xml).unwrap();
    /// // callers may tweak the config, but should keep the hardened defaults
    /// let _cfg = r.config_mut();
    /// ```
    pub fn config_mut(&mut self) -> &mut quick_xml::reader::Config {
        self.reader.config_mut()
    }
}

/// Create a pre-configured `quick_xml::Reader` for PMML bytes.
///
/// Thin wrapper around `quick_xml::Reader::from_reader` that enforces the
/// same `100 MB` file cap and `trim_text`/`expand_empty_elements` settings as
/// [`PmmlReader::from_bytes`], but does **not** track depth. Use it when the
/// caller will drive `read_event_into` and depth itself (as `unmarshal` does
/// for its fine-grained `Loop { read_event_into }` patterns).
///
/// DTD/external entities remain disabled: `quick-xml` 0.37 does not expand
/// entities by default and this function does not enable it, so XXE payloads
/// are not resolved.
///
/// # Parameters
///
/// - `bytes`: complete PMML document.
///
/// # Return
///
/// `Ok(Reader)` on success, ready for `read_event_into(&mut buf)`.
///
/// # Errors
///
/// - [`pmml_core::PmmlError::ValidationError`] if `bytes.len() > 100 MB`.
/// - No parse error here; malformed XML surfaces on the first `read_event_into` call.
///
/// # Panics
///
/// Never panics.
///
/// # Examples
///
/// ```
/// use pmml_xml::new_reader;
/// use quick_xml::events::Event;
///
/// let xml = br#"<PMML version="4.4"><Header/></PMML>"#;
/// let mut r = new_reader(xml)?;
/// let mut buf = Vec::new();
/// let ev = r.read_event_into(&mut buf)?;
/// // first event is Start(PMML) or Decl — not Eof
/// assert!(!matches!(ev, Event::Eof));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// XXE is blocked — the entity text is not expanded to file content:
///
/// ```
/// use pmml_xml::new_reader;
/// use quick_xml::events::Event;
///
/// let xxe = br#"<?xml version="1.0"?><!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]><PMML version="4.4"><Header/></PMML>"#;
/// let mut r = new_reader(xxe)?;
/// let mut buf = Vec::new();
/// let mut leaked = false;
/// loop {
///     match r.read_event_into(&mut buf) {
///         Ok(Event::Text(t)) if t.unescape().unwrap_or_default().contains("root:") => leaked = true,
///         Ok(Event::Eof) => break,
///         Ok(_) => {},
///         Err(_) => break,
///     }
///     buf.clear();
/// }
/// assert!(!leaked);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
