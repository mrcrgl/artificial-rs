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
    error::{OpenAiError, OpenAiRateLimitHeaders},
};

fn parse_retry_after_seconds(headers: &reqwest::header::HeaderMap) -> Duration {
    use reqwest::header::RETRY_AFTER;
    if let Some(val) = headers.get(RETRY_AFTER).and_then(|hv| hv.to_str().ok())
        && let Ok(secs) = val.trim().parse::<u64>()
    {
        return Duration::from_secs(secs);
    }
    Duration::from_secs(0)
}

fn header_u32(headers: &reqwest::header::HeaderMap, name: &str) -> Option<u32> {
    headers
        .get(name)
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
}

fn header_string(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|hv| hv.to_str().ok())
        .map(|s| s.to_string())
}

fn extract_rate_limit_info(
    headers: &reqwest::header::HeaderMap,
) -> (Option<Duration>, Option<String>, OpenAiRateLimitHeaders) {
    let retry_after = {
        let d = parse_retry_after_seconds(headers);
        if d.as_secs() > 0 { Some(d) } else { None }
    };

    let info = OpenAiRateLimitHeaders {
        limit_requests: header_u32(headers, "x-ratelimit-limit-requests"),
        remaining_requests: header_u32(headers, "x-ratelimit-remaining-requests"),
        reset_requests: header_string(headers, "x-ratelimit-reset-requests"),
        limit_tokens: header_u32(headers, "x-ratelimit-limit-tokens"),
        remaining_tokens: header_u32(headers, "x-ratelimit-remaining-tokens"),
        reset_tokens: header_string(headers, "x-ratelimit-reset-tokens"),
    };

    // Prefer request reset, fall back to token reset.
    let reset_at = info
        .reset_requests
        .clone()
        .or_else(|| info.reset_tokens.clone());

    (retry_after, reset_at, info)
}
#[cfg(feature = "tracing")]
fn log_rate_limit_tight(headers: &reqwest::header::HeaderMap, context: &str) {
    let rem_reqs = header_u32(headers, "x-ratelimit-remaining-requests").unwrap_or(u32::MAX);
    let rem_tokens = header_u32(headers, "x-ratelimit-remaining-tokens").unwrap_or(u32::MAX);
    let lim_reqs = header_u32(headers, "x-ratelimit-limit-requests").unwrap_or(0);
    let lim_tokens = header_u32(headers, "x-ratelimit-limit-tokens").unwrap_or(0);

    // Heuristics: warn when headroom is tight
    let tight_reqs = rem_reqs <= 2 || (lim_reqs > 0 && rem_reqs as f32 / lim_reqs as f32 <= 0.05);
    let tight_tokens =
        rem_tokens <= 128 || (lim_tokens > 0 && rem_tokens as f32 / lim_tokens as f32 <= 0.05);

    if tight_reqs || tight_tokens {
        tracing::warn!(
            context,
            remaining_requests = rem_reqs,
            limit_requests = lim_reqs,
            remaining_tokens = rem_tokens,
            limit_tokens = lim_tokens,
            "rate limit headroom is tight"
        );
    } else {
        tracing::debug!(
            context,
            remaining_requests = rem_reqs,
            limit_requests = lim_reqs,
            remaining_tokens = rem_tokens,
            limit_tokens = lim_tokens,
            "rate limit status"
        );
    }
}

#[derive(Clone, Debug)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub respect_retry_after: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            respect_retry_after: true,
        }
    }
}

impl RetryPolicy {
    fn backoff_for(&self, attempt: u32) -> Duration {
        let pow = attempt.min(10);
        let backoff = self.base_delay.saturating_mul(1 << pow);
        if backoff > self.max_delay {
            self.max_delay
        } else {
            backoff
        }
    }
}

#[derive(Clone, Debug)]
pub struct HttpTimeoutConfig {
    /// TCP/TLS connection timeout.
    pub connect_timeout: Option<Duration>,
    /// Total timeout for non-streaming requests.
    pub request_timeout: Option<Duration>,
    /// Total timeout for streaming requests.
    ///
    /// `None` keeps streams open indefinitely (default).
    pub stream_timeout: Option<Duration>,
}

impl Default for HttpTimeoutConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Some(Duration::from_secs(10)),
            request_timeout: Some(Duration::from_secs(30)),
            stream_timeout: None,
        }
    }
}

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
    retry: RetryPolicy,
    timeouts: HttpTimeoutConfig,
}

impl OpenAiClient {
    /// Convenience constructor building a default `reqwest` client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::new_with_timeouts(api_key, HttpTimeoutConfig::default())
    }

    /// Convenience constructor with explicit timeout configuration.
    pub fn new_with_timeouts(api_key: impl Into<String>, timeouts: HttpTimeoutConfig) -> Self {
        let mut builder = HttpClient::builder();
        if let Some(connect_timeout) = timeouts.connect_timeout {
            builder = builder.connect_timeout(connect_timeout);
        }
        let http = builder.build().expect("building reqwest client");

        Self::with_http_and_timeouts(api_key, http, None, timeouts)
    }

    /// Build with a custom `reqwest::Client` in case the caller needs proxy
    /// settings, custom TLS, etc.
    #[allow(dead_code)]
    pub fn with_http(
        api_key: impl Into<String>,
        http: HttpClient,
        base_url: Option<String>,
    ) -> Self {
        Self::with_http_and_timeouts(api_key, http, base_url, HttpTimeoutConfig::default())
    }

    /// Build with a custom `reqwest::Client` and timeout configuration.
    pub fn with_http_and_timeouts(
        api_key: impl Into<String>,
        http: HttpClient,
        base_url: Option<String>,
        timeouts: HttpTimeoutConfig,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            http,
            base: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
            retry: RetryPolicy::default(),
            timeouts,
        }
    }

    /// Allow callers to override the default retry policy.
    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    // Internal: send POST with retry/backoff handling.
    async fn post_json_with_retry(
        &self,
        url: String,
        headers: HeaderMap,
        request: &ChatCompletionRequest,
        request_timeout: Option<Duration>,
    ) -> Result<reqwest::Response, OpenAiError> {
        let mut attempt: u32 = 0;
        loop {
            let mut req = self
                .http
                .post(url.clone())
                .headers(headers.clone())
                .json(request);
            if let Some(timeout) = request_timeout {
                req = req.timeout(timeout);
            }
            let res = req.send().await;

            match res {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        #[cfg(feature = "tracing")]
                        {
                            log_rate_limit_tight(resp.headers(), "success");
                        }
                        return Ok(resp);
                    }

                    let should_retry = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status.is_server_error();

                    if should_retry && attempt < self.retry.max_retries {
                        let mut delay = self.retry.backoff_for(attempt);
                        #[allow(unused_assignments)]
                        let mut hdr_delay = Duration::from_secs(0);
                        if self.retry.respect_retry_after {
                            hdr_delay = parse_retry_after_seconds(resp.headers());
                            if hdr_delay > delay {
                                delay = hdr_delay;
                            }
                        }
                        #[cfg(feature = "tracing")]
                        {
                            tracing::info!(
                                attempt = attempt,
                                status = %status,
                                backoff_ms = delay.as_millis() as u64,
                                retry_after_ms = hdr_delay.as_millis() as u64,
                                "retrying request due to transient status"
                            );
                            log_rate_limit_tight(resp.headers(), "retrying");
                        }
                        // Blocking sleep to avoid introducing a new async runtime dependency.
                        std::thread::sleep(delay);
                        attempt += 1;
                        continue;
                    } else {
                        let status = resp.status();
                        let headers_map = resp.headers().clone();
                        let body = resp.text().await.unwrap_or_default();
                        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                            let (retry_after, reset_at, headers) =
                                extract_rate_limit_info(&headers_map);
                            #[cfg(feature = "tracing")]
                            {
                                let ra_ms = retry_after.map(|d| d.as_millis() as u64);
                                tracing::warn!(
                                    status = %status,
                                    retry_after_ms = ?ra_ms,
                                    reset_at = ?reset_at,
                                    "rate limited; giving up after retries"
                                );
                            }
                            return Err(OpenAiError::RateLimited {
                                status,
                                body,
                                retry_after,
                                reset_at,
                                headers,
                            });
                        } else {
                            return Err(OpenAiError::Api { status, body });
                        }
                    }
                }
                Err(err) => {
                    // Retry on transport errors up to max_retries.
                    if attempt < self.retry.max_retries
                        && (err.is_timeout() || err.is_connect() || !err.is_status())
                    {
                        let delay = self.retry.backoff_for(attempt);
                        #[cfg(feature = "tracing")]
                        {
                            tracing::info!(
                                attempt = attempt,
                                backoff_ms = delay.as_millis() as u64,
                                "retrying after transport error"
                            );
                        }
                        std::thread::sleep(delay);
                        attempt += 1;
                        continue;
                    } else {
                        return Err(OpenAiError::Http(err));
                    }
                }
            }
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
            .post_json_with_retry(url, headers, &request, self.timeouts.request_timeout)
            .await?;

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
            let resp = self
                .post_json_with_retry(url, headers, &request, self.timeouts.stream_timeout)
                .await?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use crate::api_v1::{ChatCompletionMessage, Content, MessageRole};

    fn run_single_response_server(delay: Duration, body: String, content_type: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind tcp listener");
        let addr = listener.local_addr().expect("listener addr");
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept connection");
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut req_buf = [0_u8; 8192];
            let _ = stream.read(&mut req_buf);
            thread::sleep(delay);
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            let _ = stream.flush();
        });

        format!("http://{addr}")
    }

    fn sample_request() -> ChatCompletionRequest {
        ChatCompletionRequest::new(
            "gpt-4o-mini".to_string(),
            vec![ChatCompletionMessage {
                role: MessageRole::User,
                content: Some(Content::Text("hello".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
        )
    }

    #[tokio::test]
    async fn non_streaming_respects_request_timeout() {
        let base_url = run_single_response_server(
            Duration::from_millis(200),
            r#"{"id":"x","object":"chat.completion","created":0,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop","finish_details":null}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"system_fingerprint":null}"#.to_string(),
            "application/json",
        );

        let client = OpenAiClient::with_http_and_timeouts(
            "test-key",
            reqwest::Client::new(),
            Some(base_url),
            HttpTimeoutConfig {
                connect_timeout: Some(Duration::from_secs(1)),
                request_timeout: Some(Duration::from_millis(50)),
                stream_timeout: None,
            },
        )
        .with_retry_policy(RetryPolicy {
            max_retries: 0,
            ..RetryPolicy::default()
        });

        let err = client
            .chat_completion(sample_request())
            .await
            .expect_err("non-stream request should timeout");
        match err {
            OpenAiError::Http(inner) => assert!(inner.is_timeout()),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_uses_stream_timeout_not_request_timeout() {
        let sse_body = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            r#"{"id":"x","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#
        );
        let base_url =
            run_single_response_server(Duration::from_millis(120), sse_body, "text/event-stream");

        let client = OpenAiClient::with_http_and_timeouts(
            "test-key",
            reqwest::Client::new(),
            Some(base_url),
            HttpTimeoutConfig {
                connect_timeout: Some(Duration::from_secs(1)),
                request_timeout: Some(Duration::from_millis(10)),
                stream_timeout: None,
            },
        )
        .with_retry_policy(RetryPolicy {
            max_retries: 0,
            ..RetryPolicy::default()
        });

        let mut stream = Box::pin(client.chat_completion_stream(sample_request()));
        let first = stream
            .next()
            .await
            .expect("stream should produce first chunk")
            .expect("first chunk should parse");
        assert_eq!(first.choices.len(), 1);
    }
}
