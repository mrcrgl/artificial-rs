use std::{env, sync::Arc};

use artificial_core::error::{ArtificialError, Result};

use crate::client::{OpenAiClient, RetryPolicy};

/// Thin wrapper that wires the HTTP client [`OpenAiClient`] into a value that
/// implements [`artificial_core::backend::Backend`].
///
/// Think of it as the **service locator** for the OpenAI back-end:
///
/// * stores the API key (and optionally a custom base URL in the future),
/// * owns a shareable, connection-pooled `reqwest::Client`,
/// * provides a fluent [`OpenAiAdapterBuilder`] so callers don’t have to juggle
///   `Option<String>` manually.
///
/// The type itself purposefully exposes **no additional methods**—all user-
/// facing functionality sits on the generic [`artificial_core::ArtificialClient`]
/// once the adapter is plugged in.
pub struct OpenAiAdapter {
    pub(crate) client: Arc<OpenAiClient>,
}

impl OpenAiAdapter {}

/// Builder for [`OpenAiAdapter`].
///
/// # Typical usage
///
/// ```rust,no_run
/// use artificial_openai::OpenAiAdapterBuilder;
///
/// let backend = OpenAiAdapterBuilder::new_from_env()
///     .build()
///     .expect("OPENAI_API_KEY must be set");
/// ```
///
/// The builder pattern keeps future options (proxy URL, organisation ID, …)
/// backwards compatible without breaking existing `build()` calls.
#[derive(Default)]
pub struct OpenAiAdapterBuilder {
    pub(crate) api_key: Option<String>,
    pub(crate) retry: Option<RetryPolicy>,
}

impl OpenAiAdapterBuilder {
    /// Create an *empty* builder. Remember to supply an API key manually.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience constructor that tries to load the `OPENAI_API_KEY`
    /// environment variable.
    ///
    /// # Panics
    ///
    /// Never panics. Missing keys only surface during [`Self::build`].
    pub fn new_from_env() -> Self {
        Self {
            api_key: env::var("OPENAI_API_KEY").ok(),
            retry: None,
        }
    }

    /// Set a retry policy for OpenAI HTTP calls.
    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Finalise the builder and return a ready-to-use adapter.
    ///
    /// # Errors
    ///
    /// * [`ArtificialError::Invalid`] – if the API key is missing.
    pub fn build(self) -> Result<OpenAiAdapter> {
        let api_key = self.api_key.ok_or(ArtificialError::Invalid(
            "missing env variable: `OPENAI_API_KEY`".into(),
        ))?;

        let client = if let Some(retry) = self.retry {
            OpenAiClient::new(api_key).with_retry_policy(retry)
        } else {
            OpenAiClient::new(api_key)
        };

        Ok(OpenAiAdapter {
            client: Arc::new(client),
        })
    }
}
