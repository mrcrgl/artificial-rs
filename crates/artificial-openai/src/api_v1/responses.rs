#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Types for OpenAI's Responses API (reasoning-capable models like o4-mini, o3-mini, etc.).
///
/// This module intentionally keeps the shapes flexible with `serde_json::Value` where the
/// upstream schema changes frequently. The goal is to enable integration with both the
/// non-streaming and the streaming (SSE) variants of the `/v1/responses` endpoint without
/// forcing the rest of the crate into tight coupling with the exact wire format.
///
/// High-level guidance:
/// - Non-streaming: POST /v1/responses with `ResponsesRequest`, receive `ResponsesResponse`.
/// - Streaming:     POST /v1/responses with `ResponsesRequest { stream: Some(true), .. }`,
///                  then decode each SSE `data:` line as `ResponseStreamEvent`.
///
/// Note: The Responses API’s usage accounting and output structure differ from the classic
/// chat/completions API. To stay robust against upstream changes, we model many fields as
/// `serde_json::Value`.
///
/// Parsing SSE events:
/// - Each SSE frame contains an event "type" in the JSON payload (not the SSE `event:` line).
/// - Deserialize the payload as `ResponseStreamEvent` (internally tagged with `"type"`).
/// - Terminate the stream once you receive `ResponseStreamEvent::Completed` or `ResponseStreamEvent::Error`.
///
/// See: https://platform.openai.com/docs/guides/reasoning and Responses API docs.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponsesRequest {
    /// Model identifier (e.g., "o4-mini", "o3-mini", "gpt-4o", ...).
    pub model: String,

    /// Either `input` or `messages` is required by the API:
    /// - `input`: free-form input (string or structured) for non-chat use cases.
    /// - `messages`: chat-style input compatible with the Responses API.
    ///
    /// Keep both as `Value` to allow the caller to supply the exact shape they need.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<ResponseInput>>,

    /// Reasoning controls for models that support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningOptions>,

    /// Output controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Enable SSE when set to true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Some Responses API features (like JSON output) are configured here.
    /// Leave as `Value` to avoid tight coupling to a specific upstream schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,

    /// Arbitrary fields the caller may want to pass through while we iterate on the schema.
    /// This offers forward compatibility without breaking deserialization.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

impl ResponsesRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: None,
            reasoning: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            response_format: None,
            extra: Default::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseInput {
    #[serde(rename = "message")]
    Message,
}

/// Reasoning control block. The exact options are subject to change upstream, but
/// "effort" is the primary control currently exposed for o-series models.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReasoningOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

/// Non-streaming Responses API object.
///
/// The Responses API returns structured output that can vary based on the
/// requested format (text, JSON object, tool calls, etc.). To remain flexible,
/// `output` is typed as `serde_json::Value`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,

    /// The structured output from the model (shape is model/feature dependent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,

    /// Usage accounting; Responses API may differ from chat/completions.
    /// Keep as `Value` to remain compatible with upstream changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,

    /// Additional fields we don't explicitly model yet.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

/// Streaming event envelope for the Responses API.
///
/// The API sends JSON objects with a `"type"` field indicating the event kind.
/// We model the most common event types explicitly and keep a catch-all for
/// forward compatibility. Each variant includes only the fields we commonly
/// need, plus an `extra` map to retain unknown fields for debugging or later use.
///
/// Typical termination conditions:
/// - `Completed` – the model finished generating output.
/// - `Error` – streaming aborted due to an error.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseStreamEvent {
    /// Delta fragments of output text (accumulate to display incremental text).
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        delta: String,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        output_index: Option<i64>,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, Value>,
    },

    /// Final, complete output text for a given output index.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        text: String,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        output_index: Option<i64>,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, Value>,
    },

    /// Emitted when the entire response is complete.
    #[serde(rename = "response.completed")]
    Completed {
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<Value>,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Value>,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, Value>,
    },

    /// Error emitted during streaming (terminal).
    #[serde(rename = "response.error")]
    Error {
        error: Value,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, Value>,
    },

    /// Some models can emit refusal/guardrail tokens incrementally.
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        delta: String,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, Value>,
    },

    /// Final refusal content for a given output.
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        text: String,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, Value>,
    },

    /// Catch-all for newer events not explicitly modeled above.
    #[serde(other)]
    Unknown,
}
