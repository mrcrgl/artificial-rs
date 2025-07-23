use std::{future::Future, pin::Pin};

use crate::{
    error::Result,
    template::{IntoPrompt, PromptTemplate},
};

/// A **backend** turns a prompt into a network call to a concrete provider
/// (OpenAI, Ollama, Anthropic, …) and parses the structured response.
///
/// The trait is intentionally minimal:
///
/// * **One associated type** – the in-memory `Message` representation this
///   provider accepts.
/// * **One async-ish method** – `complete`, which performs a *single*
///   non-streaming round-trip and returns a value whose type is dictated by
///   the `PromptTemplate`.
///
/// The method returns a [`Pin<Box<dyn Future>>`] so we stay object-safe
/// without pulling in `async_trait`.
pub trait PromptExecutionProvider: Send + Sync {
    /// Chat message type consumed by this backend.
    ///
    /// A simple setup can re-use `crate::generic::GenericMessage`.
    /// Providers with richer wire formats (function calls, images …) can
    /// supply their own struct.
    type Message: Send + Sync + 'static;

    /// Execute the prompt and deserialize the provider’s reply into
    /// `P::Output`.
    ///
    /// The blanket constraint `P: PromptTemplate<Message = Self::Message>`
    /// guarantees at **compile time** that callers only feed the backend
    /// messages it understands.
    fn prompt_execute<'p, P>(
        &'p self,
        prompt: P,
    ) -> Pin<Box<dyn Future<Output = Result<P::Output>> + Send + 'p>>
    where
        P: PromptTemplate + Send + Sync + 'p,
        <P as IntoPrompt>::Message: Into<Self::Message>;
}

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
    fn chat_complete<'p, P>(
        &'p self,
        prompt: P,
    ) -> Pin<Box<dyn Future<Output = Result<P::Output>> + Send + 'p>>
    where
        P: PromptTemplate + Send + Sync + 'p,
        <P as IntoPrompt>::Message: Into<Self::Message>;
}
