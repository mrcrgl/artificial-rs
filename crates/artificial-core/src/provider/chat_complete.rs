use std::{future::Future, pin::Pin};

use crate::{
    error::Result,
    generic::{GenericChatCompletionResponse, GenericFunctionSpec, GenericMessage},
    model::Model,
};
use futures_core::stream::Stream;

/// A **backend** turns a chat prompt into a network call to a concrete provider
/// (OpenAI, Ollama, Anthropic, …) and parses the structured chat response.
///
/// The trait is intentionally minimal:
///
/// * **One associated type** – the in-memory `Message` representation this
///   provider accepts.
/// * **One async-ish method** – `chat_complete`, which performs a *single*
///   non-streaming round-trip and returns a value whose type is dictated by
///   the `PromptTemplate`.
pub trait ChatCompletionProvider: Send + Sync {
    /// Chat message type consumed by this backend.
    type Message: Send + Sync + 'static;

    /// Execute the chat prompt and deserialize the provider’s reply into
    /// `P::Output`.
    fn chat_complete<'p, M>(
        &self,
        params: ChatCompleteParameters<M>,
    ) -> Pin<
        Box<dyn Future<Output = Result<GenericChatCompletionResponse<GenericMessage>>> + Send + 'p>,
    >
    where
        M: Into<Self::Message> + Clone + Send + Sync + 'p;
}

/// A provider that can deliver the model’s answer **incrementally**.
///
/// The stream yields UTF-8 text *deltas* (similar to OpenAI’s SSE format).
/// Tool-call and richer payload support can be layered on later by
/// introducing a dedicated enum – starting with plain text keeps the API
/// minimal and backend-agnostic.
pub trait StreamingChatProvider: ChatCompletionProvider {
    /// The item type returned on the stream.  For now it is plain UTF-8 text
    /// chunks, but back-ends are free to wrap it in richer enums if needed.
    type Delta<'s>: Stream<Item = Result<String>> + Send + 's
    where
        Self: 's;

    /// Start a streaming chat completion.
    fn chat_complete_stream<'p, M>(&self, params: ChatCompleteParameters<M>) -> Self::Delta<'p>
    where
        M: Into<Self::Message> + Clone + Send + Sync + 'p;
}

#[derive(Debug, Clone)]
pub struct ChatCompleteParameters<M: Clone> {
    pub messages: Vec<M>,
    pub model: Model,
    pub tools: Option<Vec<GenericFunctionSpec>>,
    pub temperature: Option<f64>,
    pub response_format: Option<serde_json::Value>,
}

impl<M: Clone> ChatCompleteParameters<M> {
    pub fn new(messages: Vec<M>, model: Model) -> Self {
        Self {
            messages,
            model,
            tools: None,
            temperature: None,
            response_format: None,
        }
    }

    pub fn messages(&self) -> &Vec<M> {
        &self.messages
    }

    pub fn model(&self) -> Model {
        self.model.clone()
    }

    pub fn tools(&self) -> Option<&Vec<GenericFunctionSpec>> {
        self.tools.as_ref()
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_response_format(mut self, response_format: serde_json::Value) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn with_tools(mut self, tools: Vec<GenericFunctionSpec>) -> Self {
        self.tools = Some(tools);
        self
    }
}
