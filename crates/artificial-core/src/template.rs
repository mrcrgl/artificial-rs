//! Abstractions that tie a **prompt** to a concrete **model** and a **typed
//! response**.
//!
//! The *artificial* framework purposely keeps the public surface small.  A
//! developer usually needs only two traits to go from “some string fragments”
//! to “ready-to-send payload”:
//!
//! 1. [`IntoPrompt`] – turns *any* value into a list of chat messages.
//! 2. [`PromptTemplate`] – adds metadata such as the target model and the
//!    expected JSON response schema.
//!
//! Provider back-ends (e.g. `artificial-openai`) accept *any* `P` that
//! implements **both** traits.  Thanks to Rust’s powerful type system the
//! compiler guarantees at compile time that
//!
//! * the message type produced by the prompt matches what the back-end expects,
//! * the JSON returned by the provider can be deserialised into `P::Output`.
//!
//! ```rust
//! use artificial_core::template::{IntoPrompt, PromptTemplate};
//! use artificial_core::generic::{GenericMessage, GenericRole};
//! use artificial_core::model::{Model, OpenAiModel};
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, JsonSchema)]
//! #[serde(deny_unknown_fields)]
//! struct Hello { greeting: String }
//!
//! struct HelloPrompt;
//!
//! impl IntoPrompt for HelloPrompt {
//!     type Message = GenericMessage;
//!     fn into_prompt(self) -> Vec<Self::Message> {
//!         vec![GenericMessage::new("Say hello!".into(), GenericRole::User)]
//!     }
//! }
//!
//! impl PromptTemplate for HelloPrompt {
//!     type Output = Hello;
//!     const MODEL: Model = Model::OpenAi(OpenAiModel::Gpt4oMini);
//! }
//! ```
//!
//! See `examples/openai_hello_world.rs` for a fully working program.
use std::any::Any;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::model::Model;

/// High-level description of a prompt.
///
/// Implement this trait **in addition** to [`IntoPrompt`] to specify:
///
/// * `Output` – the strongly-typed Rust struct you expect from the LLM.
/// * `MODEL`  – the identifier of the model that should handle the request.
///
/// The blanket constraints on `Output` (`JsonSchema + Deserialize + Any`)
/// enable the OpenAI adapter to automatically derive a JSON Schema and to
/// down-cast the erased type if necessary.
pub trait PromptTemplate: IntoPrompt {
    /// Type produced by the LLM and returned to the caller.
    type Output: JsonSchema + for<'de> Deserialize<'de> + Any;

    /// Logical model identifier.  The back-end will map this to its own naming
    /// scheme (`"gpt-4o-mini"`, `"claude-3-haiku"`, …).
    const MODEL: Model;
}

/// Converts a value into a series of chat messages.
///
/// Provider crates typically use [`crate::generic::GenericMessage`], but a
/// back-end can require its own richer struct.  By making the `Message` type
/// an **associated type** we keep the trait flexible without resorting to
/// dynamic dispatch.
pub trait IntoPrompt {
    /// Chat message representation emitted by the prompt.
    type Message: Send + Sync + 'static;

    /// Consume `self` and return **all** messages in the desired order.
    fn into_prompt(self) -> Vec<Self::Message>;
}

/// Convenience implementation so a single [`crate::generic::GenericMessage`]
/// can be passed directly to the client without wrapping it in a struct.
impl IntoPrompt for crate::generic::GenericMessage {
    type Message = crate::generic::GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        vec![self]
    }
}
