//! Generic, lightweight client that executes a [`PromptTemplate`] against a
//! single concrete [`Backend`].
//!
//! The client is **generic over the backend type `B`**, so the compiler
//! guarantees that:
//! * The prompt’s `Message` type matches what the backend expects.
//! * No dynamic dispatch or object-safety hurdles appear in user code.
//!
//! ```rust
//! use artificial_core::{ArtificialClient, generic::{GenericMessage, GenericRole},
//!                      template::*, model::*};
//!
//! struct Hello;
//!
//! impl PromptTemplate for Hello {
//!     type Output         = serde_json::Value;
//!     const MODEL: Model  = Model::OpenAi(OpenAiModel::Gpt4o);
//! }
//!
//! impl IntoPrompt for Hello {
//!     type Message = GenericMessage;
//!     fn into_prompt(self) -> Vec<Self::Message> {
//!         vec![GenericMessage::new("Say hello!".into(), GenericRole::User)]
//!     }
//! }
//!
//! # fn main() {}
//! ```
//!
//! Any backend crate (e.g. `artificial-openai`, `artificial-ollama`) just
//! implements Provider traits and the same client works out of the box.
use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    error::Result,
    generic::{GenericChatCompletionResponse, StreamingEventsProvider},
    provider::{
        ChatCompleteParameters, ChatCompletionProvider, PromptExecutionProvider,
        StreamingChatProvider,
    },
    template::{IntoPrompt, PromptTemplate},
};

/// A client bound to a single provider.
///
/// Clone the client if you need to share it across tasks—`B` controls whether
/// that’s cheap (e.g. wraps an `Arc`) or a deep copy.
#[derive(Debug, Clone)]
pub struct ArtificialClient<B> {
    backend: Arc<B>,
}

impl<B> ArtificialClient<B>
where
    B: PromptExecutionProvider,
{
    /// Create a new client that delegates all calls to `backend`.
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Access the underlying backend (e.g. to tweak provider-specific settings).
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B: PromptExecutionProvider> PromptExecutionProvider for ArtificialClient<B> {
    type Message = B::Message;

    fn prompt_execute<'a, 'p, P>(
        &'a self,
        prompt: P,
    ) -> Pin<Box<dyn Future<Output = Result<GenericChatCompletionResponse<P::Output>>> + Send + 'p>>
    where
        'a: 'p,
        P: PromptTemplate + Send + Sync + 'p,
        <P as IntoPrompt>::Message: Into<Self::Message>,
    {
        let backend = Arc::clone(&self.backend);
        Box::pin(async move { backend.prompt_execute(prompt).await })
    }
}

impl<B: ChatCompletionProvider> ChatCompletionProvider for ArtificialClient<B> {
    type Message = B::Message;

    fn chat_complete<'s, M>(
        &'s self,
        params: ChatCompleteParameters<M>,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<GenericChatCompletionResponse<crate::generic::GenericMessage>>,
                > + Send
                + 's,
        >,
    >
    where
        M: Into<Self::Message> + Clone + Send + Sync + 's,
    {
        self.backend.chat_complete(params)
    }
}

impl<B: StreamingChatProvider> StreamingChatProvider for ArtificialClient<B> {
    type Delta<'s>
        = B::Delta<'s>
    where
        Self: 's;

    fn chat_complete_stream<'s, M>(&'s self, params: ChatCompleteParameters<M>) -> Self::Delta<'s>
    where
        M: Into<Self::Message> + Clone + Send + Sync + 's,
    {
        self.backend.chat_complete_stream(params)
    }
}

impl<B: StreamingEventsProvider> StreamingEventsProvider for ArtificialClient<B> {
    type EventStream<'s>
        = B::EventStream<'s>
    where
        Self: 's;

    fn chat_complete_events_stream<'s, M>(
        &'s self,
        params: crate::provider::ChatCompleteParameters<M>,
    ) -> Self::EventStream<'s>
    where
        M: Into<Self::Message> + Clone + Send + Sync + 's,
    {
        self.backend.chat_complete_events_stream(params)
    }
}
