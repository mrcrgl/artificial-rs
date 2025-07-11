//! Unified error type exposed by **`artificial-core`**.
//!
//! Provider crates should convert their internal errors into one of these
//! variants before bubbling them up to the [`ArtificialClient`].  This keeps
//! the public API small while still conveying rich diagnostic information.


use thiserror::Error;

/// Convenient alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, ArtificialError>;

#[derive(Debug, Error)]
pub enum ArtificialError {
    /// A prompt targeted a provider (`provider_id`) for which no backend has
    /// been registered in the [`ArtificialClient`].
    #[error("backend for provider `{provider}` is not configured")]
    BackendNotConfigured { provider: &'static str },

    /// The selected backend is present but does not recognise or support the
    /// requested `model`.
    #[error("provider `{provider}` does not support model `{model}`")]
    ModelNotSupported {
        provider: &'static str,
        model: &'static str,
    },

    /// Failure while serialising or deserialising JSON payloads sent to / received
    /// from the LLM provider.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Generic forwarding of any backend-specific error that doesn’t fit another
    /// category.
    #[error("backend returned an error: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync + 'static>),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("invalid: {0}")]
    Invalid(String),
}
