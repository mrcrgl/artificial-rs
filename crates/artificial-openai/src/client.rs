use std::time::Duration;

use reqwest::{
    Client as HttpClient,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};

use crate::{
    api_v1::chat_completion::{ChatCompletionRequest, ChatCompletionResponse},
    error::OpenAiError,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Minimal HTTP client for OpenAI’s *chat/completions* endpoint.
///
/// * Non-streaming only (one request ▶ one response).
/// * Accepts and returns the `api_v1` request / response structs defined
///   in this crate.
/// * Shares a single `reqwest::Client`, so cloning `OpenAiClient` is cheap.
#[derive(Clone)]
pub struct OpenAiClient {
    api_key: String,
    http: HttpClient,
    base: String,
}

impl OpenAiClient {
    /// Convenience constructor building a default `reqwest` client:
    /// 30 s timeout, HTTP/2 prior knowledge, Rustls TLS.
    pub fn new(api_key: impl Into<String>) -> Self {
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("building reqwest client");

        Self::with_http(api_key, http, None)
    }

    /// Build with a custom `reqwest::Client` in case the caller needs proxy
    /// settings, custom TLS, etc.
    pub fn with_http(
        api_key: impl Into<String>,
        http: HttpClient,
        base_url: Option<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            http,
            base: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
        }
    }

    /// Perform a **non-streaming** chat completion.
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, OpenAiError> {
        // Build headers once.
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
        );

        let url = format!("{}/chat/completions", self.base);
        let resp = self
            .http
            .post(url)
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::Api { status, body });
        }

        let bytes = resp.bytes().await?;
        let parsed: ChatCompletionResponse = serde_json::from_slice(&bytes)?;
        Ok(parsed)
    }
}
