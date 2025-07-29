use std::str::Utf8Error;

use artificial_core::error::ArtificialError;
use reqwest::StatusCode;

/// High-level error type covering every failure mode the client can hit.
#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
    #[error("request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("couldn’t serialise body: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("OpenAI returned non-success status {status}: {body}")]
    Api { status: StatusCode, body: String },

    #[error("OpenAI format error: {0}")]
    Format(String),

    #[error("unknown error: {0}")]
    Unknown(String),
}

impl From<OpenAiError> for ArtificialError {
    fn from(value: OpenAiError) -> Self {
        ArtificialError::Backend(Box::new(value))
    }
}

impl From<Utf8Error> for OpenAiError {
    fn from(value: Utf8Error) -> Self {
        Self::Unknown(format!("UTF8 error: {value}"))
    }
}
