//! Unified error type — `thiserror` based, no `anyhow` in hot path.

use thiserror::Error;

/// Top-level PMML error. Variants mirror JPMML `PMMLException` + `UnsupportedMarkupException`.
#[derive(Error, Debug)]
pub enum PmmlError {
    #[error("unsupported markup: {0}")]
    UnsupportedMarkup(String),

    #[error("invalid value: {0}")]
    InvalidValue(String),

    #[error("missing field: {0}")]
    MissingField(String),

    #[error("parse error at {context}: {message}")]
    ParseError { context: String, message: String },

    #[error("type error: {0}")]
    TypeError(String),

    #[error("validation error: {0}")]
    ValidationError(String),

    #[error("io error: {0}")]
    Io(String),

    #[error("arithmetic overflow: {0}")]
    ArithmeticOverflow(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl PmmlError {
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::UnsupportedMarkup(msg.into())
    }
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidValue(msg.into())
    }
    pub fn missing(field: impl Into<String>) -> Self {
        Self::MissingField(field.into())
    }
    pub fn parse(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ParseError {
            context: context.into(),
            message: message.into(),
        }
    }
}

/// Result alias.
pub type Result<T> = std::result::Result<T, PmmlError>;
