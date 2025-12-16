//! Generic message and role types used by the *artificial-core* crate.
//!
//! They deliberately mirror the concepts exposed by most provider APIs:
//! “system”, “user”, “assistant”, and “tool”.  By staying minimal and
//! provider-agnostic we can:
//!
//! * convert them into provider-specific structs via a simple `From`/`Into`,
//! * serialize them without pulling in heavyweight dependencies, and
//! * use them in unit tests without mocking a full transport layer.
//!
//! ## When to add more fields?
//!
//! Only if the additional data is **required by multiple back-ends** or
//! **fundamentally provider-independent**.  Otherwise extend the
//! provider-specific message type instead of bloating this one.
use std::fmt::Display;

use serde::{Deserialize, Serialize};

/// Lightweight container representing a single chat message that is
/// independent of any specific LLM provider.
///
/// * `message` – the raw UTF-8 content. Markdown is fine, but keep newlines
///   and indentation portable.
/// * `role` – see [`GenericRole`] for permitted values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericMessage {
    pub content: Option<String>,
    pub role: GenericRole,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<GenericFunctionCallIntent>>,
    pub tool_call_id: Option<String>,
}

impl GenericMessage {
    /// Convenience constructor mirroring the field order used by common HTTP
    /// APIs (`role`, then `content`).
    ///
    /// ```rust
    /// use artificial_core::generic::{GenericMessage, GenericRole};
    ///
    /// let sys = GenericMessage::new("You are a helpful bot.".into(),
    ///                               GenericRole::System);
    /// ```
    pub fn new(message: String, role: GenericRole) -> Self {
        Self {
            content: Some(message),
            role,
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn new_tool_call(tool_call_id: String, tool_calls: Vec<GenericFunctionCallIntent>) -> Self {
        Self {
            content: None,
            role: GenericRole::Assistant,
            name: None,
            tool_calls: Some(tool_calls),
            tool_call_id: Some(tool_call_id),
        }
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn with_tool_call_id(mut self, tool_call_id: impl ToString) -> Self {
        self.tool_call_id = Some(tool_call_id.to_string());
        self
    }
}

/// High-level chat roles recognised by most LLM providers.
///
/// The `Display` implementation renders the canonical lowercase name so you
/// can feed it directly into JSON without extra mapping logic.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenericRole {
    /// “System” messages define global behaviour and style guidelines.
    System,
    /// Messages produced by the assistant / model.
    Assistant,
    /// Messages originating from the human user.
    User,
    /// Special role used when a **tool call** or similar structured result is
    /// injected into the conversation.
    Tool,
}

impl Display for GenericRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericRole::System => write!(f, "system"),
            GenericRole::Assistant => write!(f, "assistant"),
            GenericRole::User => write!(f, "user"),
            GenericRole::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug)]
pub struct GenericChatCompletionResponse<T> {
    pub content: ResponseContent<T>,
    pub usage: Option<GenericUsageReport>,
}

#[derive(Debug)]
pub enum ResponseContent<T> {
    Finished(T),
    ToolCalls(GenericMessage),
}

#[derive(Debug, Clone)]
pub struct GenericUsageReport {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericFunctionCallIntent {
    pub id: String,
    pub function: GenericFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericFunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Copy)]
pub enum GenericStramingChatChunk {
    Created,
    Completed,
    Failed,
    OutputTextDelta,
    OutputTextDone,
}

#[derive(Debug, Clone)]
pub struct GenericFunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Plain text delta emitted by the assistant.
    TextDelta(String),

    /// A tool-call was initiated (OpenAI-style indexed stream).
    ToolCallStart {
        index: usize,
        id: Option<String>,
        name: Option<String>,
    },

    /// A partial arguments JSON fragment for tool-call at `index`.
    ToolCallArgumentsDelta {
        index: usize,
        arguments_fragment: String,
    },

    /// A completed tool-call intent (arguments parsed into JSON).
    ToolCallComplete {
        index: usize,
        intent: GenericFunctionCallIntent,
    },

    /// The assistant finished the message (e.g. stop or tool_calls).
    MessageEnd,

    /// Optional token usage report at the end of the stream.
    Usage(GenericUsageReport),
}

/// Provider-agnostic trait for streaming structured events (text + tool-calls).
/// This complements the existing text-only `StreamingChatProvider` trait.
pub trait StreamingEventsProvider: crate::provider::ChatCompletionProvider {
    type EventStream<'s>: futures_core::stream::Stream<Item = crate::error::Result<StreamEvent>>
        + Send
        + 's
    where
        Self: 's;

    fn chat_complete_events_stream<'p, M>(
        &self,
        params: crate::provider::ChatCompleteParameters<M>,
    ) -> Self::EventStream<'p>
    where
        M: Into<Self::Message> + Send + Sync + 'p;
}
