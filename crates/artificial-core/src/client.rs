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
//! implements [`Backend`] and the same client works out of the box.

use crate::{
    backend::Backend,
    error::Result,
    template::{IntoPrompt, PromptTemplate},
};

/// A client bound to a single provider backend `B`.
///
/// Clone the client if you need to share it across tasks—`B` controls whether
/// that’s cheap (e.g. wraps an `Arc`) or a deep copy.
#[derive(Debug, Clone)]
pub struct ArtificialClient<B> {
    backend: B,
}

impl<B> ArtificialClient<B>
where
    B: Backend,
{
    /// Create a new client that delegates all calls to `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Run a prompt on the backend and return the deserialised output.
    ///
    /// # Errors
    ///
    /// Any provider-specific failure is converted into
    /// [`crate::error::ArtificialError`] and bubbled up transparently.
    pub async fn chat_complete<P>(&self, prompt: P) -> Result<P::Output>
    where
        P: PromptTemplate + Send + 'static,
        <P as IntoPrompt>::Message: Into<B::Message>,
    {
        self.backend.chat_complete(prompt).await
    }

    /// Access the underlying backend (e.g. to tweak provider-specific settings).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Consume the client and return the inner backend.
    pub fn into_backend(self) -> B {
        self.backend
    }
}
