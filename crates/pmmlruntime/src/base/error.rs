//! Unified error type — `thiserror` based, no backtrace on the hot path.
//!
//! [`PmmlError`] mirrors JPMML `PMMLException` + `UnsupportedMarkupException`.
//! All `pmml-*` crates return `Result<T, PmmlError>`; the `Other` variant wraps
//! `anyhow::Error` for IO/arrow interop but is never used in hot scoring.

use thiserror::Error;

/// Top-level PMML error.
///
/// Variants mirror `org.jpmml.model.PMMLException` and `UnsupportedMarkupException`.
/// Use the constructor helpers ([`PmmlError::unsupported`], etc.) rather than
/// constructing variants directly to keep messages consistent.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::PmmlError;
/// let e = PmmlError::unsupported("AnomalyDetectionModel");
/// assert!(e.to_string().contains("unsupported markup"));
/// ```
#[derive(Error, Debug)]
pub enum PmmlError {
    /// PMML markup that is valid per `pmml.xsd` but not supported by this runtime.
    ///
    /// Examples: `AnomalyDetectionModel`, `BayesianNetworkModel`, `TimeSeriesModel`,
    /// or a `RESULT-FEATURE` like `confidenceIntervalLower`. See crate docs for the
    /// full unsupported list (kept in `pmml-ir::verify`).
    #[error("unsupported markup: {0}")]
    UnsupportedMarkup(String),

    /// A value could not be coerced to the target [`DataType`](crate::base::field::DataType) / [`OpType`](crate::base::field::OpType).
    #[error("invalid value: {0}")]
    InvalidValue(String),

    /// A required field (by `MiningSchema` or `Model` input) was missing and no `missingValueReplacement` was defined.
    #[error("missing field: {0}")]
    MissingField(String),

    /// XML parse error, including context (element/attribute name).
    #[error("parse error at {context}: {message}")]
    ParseError { context: String, message: String },

    /// Type mismatch at evaluation (e.g., `Discrete` where `Continuous` expected).
    #[error("type error: {0}")]
    TypeError(String),

    /// Structural validation failure: XML depth `>512`, file `>100 MB`, empty `MiningSchema`, etc.
    #[error("validation error: {0}")]
    ValidationError(String),

    /// `std::io` error (file read). Wrapped as string to avoid `std::io::Error` in hot path.
    #[error("io error: {0}")]
    Io(String),

    /// `checked_add`/`checked_mul` overflow while evaluating `Apply` or scoring.
    #[error("arithmetic overflow: {0}")]
    ArithmeticOverflow(String),

    /// Transparent wrapper for `anyhow::Error` — IO/Arrow interop, never hot path.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl PmmlError {
    /// Construct [`PmmlError::UnsupportedMarkup`] with a message.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::PmmlError;
    /// let e = PmmlError::unsupported("SequenceModel");
    /// assert!(e.to_string().contains("SequenceModel"));
    /// ```
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::UnsupportedMarkup(msg.into())
    }
    /// Construct [`PmmlError::InvalidValue`] with a message.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidValue(msg.into())
    }
    /// Construct [`PmmlError::MissingField`] for `field`.
    pub fn missing(field: impl Into<String>) -> Self {
        Self::MissingField(field.into())
    }
    /// Construct [`PmmlError::ParseError`] with `context` (element/attr) and `message`.
    pub fn parse(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ParseError {
            context: context.into(),
            message: message.into(),
        }
    }
}

/// Result alias used throughout the workspace.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{PmmlError, Result};
/// fn parse(s: &str) -> Result<u32> {
///     s.parse().map_err(|e| PmmlError::parse("field", format!("{e}")))
/// }
/// assert!(parse("42").is_ok());
/// ```
pub type Result<T> = std::result::Result<T, PmmlError>;
