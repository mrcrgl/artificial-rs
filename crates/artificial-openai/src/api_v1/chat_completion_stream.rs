use serde::Deserialize;

use super::chat_completion::{FinishReason, MessageRole};

/// A delta message as returned by OpenAI when `stream = true`.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ChatCompletionMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A single streaming choice payload.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunkChoice {
    pub index: i64,
    pub delta: ChatCompletionMessageDelta,
    pub finish_reason: Option<FinishReason>,
}

/// The outermost object sent by OpenAI for each SSE chunk.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunkResponse {
    pub id: Option<String>,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}
