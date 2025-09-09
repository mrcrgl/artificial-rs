use async_stream::try_stream;

use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::{
    Client as HttpClient,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use std::time::Duration;

use crate::{
    api_v1::{ChatCompletionChunkResponse, ChatCompletionRequest, ChatCompletionResponse},
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

    /// Perform a **streaming** chat completion.
    pub fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
    ) -> impl Stream<Item = Result<ChatCompletionChunkResponse, OpenAiError>> + '_ {
        use reqwest::header::{ACCEPT, HeaderValue};

        // 1) enforce streaming flag
        request.stream = Some(true);

        // 2) headers (incl. SSE accept)
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
        );
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));

        let url = format!("{}/chat/completions", self.base);

        // 3) async stream wrapper
        try_stream! {
            let resp = self.http.post(url).headers(headers).json(&request).send().await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return  Err(OpenAiError::Api { status, body })?;
            }

            let mut bytes_stream = resp.bytes_stream();
            let mut buf = Vec::new();

            while let Some(chunk) = bytes_stream.next().await {
                let chunk = chunk?;
                buf.extend_from_slice(&chunk);

                while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                    let frame: Vec<u8> = buf.drain(..pos + 2).collect();
                    let frame_str = std::str::from_utf8(&frame)?;

                    if let Some(data) = frame_str.strip_prefix("data: ") {
                        let data = data.trim();
                        if data == "[DONE]" { return; }

                        let parsed: ChatCompletionChunkResponse = serde_json::from_str(data)?;
                        yield parsed;
                    }
                }
            }
        }
    }
}
