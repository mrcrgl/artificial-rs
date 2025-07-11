//! Simple **builder** that concatenates multiple values implementing
//! [`IntoPrompt`](artificial_core::template::IntoPrompt).
//!
//! ```text
//! ┌───────────────┐    IntoPrompt     ┌────────────────┐
//! │ CurrentDate   │ ─────────────────►│ Vec<Message>   │
//! ├───────────────┤                   ├────────────────┤
//! │ StaticFragment│ ─────────────────►│ Vec<Message>   │
//! ├───────────────┤                   ├────────────────┤
//! │ …             │ ─────────────────►│ Vec<Message>   │
//! └───────────────┘                   └────────────────┘
//!            ▲                                     │
//!            └────────── PromptChain::build() ◄────┘
//! ```
//!
//! # Motivation
//!
//! In real-world prompts you often want to **compose** smaller, reusable
//! *fragments*—for example:
//!
//! * a static role description,
//! * the current date/time,
//! * the active user profile,
//! * recent chat history,
//! * a final user instruction.
//!
//! `PromptChain` lets you line up these fragments in a clear, linear fashion
//! **without** mutable vectors or verbose `extend()` calls.
//!
//! # Usage
//!
//! ```rust,ignore
//! use artificial_prompt::chain::PromptChain;
//! use artificial_types::fragments::{CurrentDateFragment, StaticFragment};
//! use artificial_core::generic::{GenericMessage, GenericRole};
//!
//! let messages: Vec<GenericMessage> = PromptChain::new()
//!     .with(StaticFragment::new("You are a helpful bot.", GenericRole::System))
//!     .with(CurrentDateFragment::new())
//!     .with(StaticFragment::new("Convert the text to uppercase.", GenericRole::User))
//!     .build();
//!
//! assert_eq!(messages.len(), 3);
//! ```
//!
//! The generic parameter `Message` allows back-ends to plug in their own, richer
//! message types while reusing the same chaining logic.
use artificial_core::template::IntoPrompt;

/// Lightweight container that accumulates messages produced by
/// [`IntoPrompt`] implementors.
///
/// The single `Vec` field is kept private so the only way to obtain the result
/// is through [`Self::build`], ensuring the builder API remains fluent.
pub struct PromptChain<Message>(Vec<Message>);

impl<Message> Default for PromptChain<Message> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Message> PromptChain<Message> {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self(vec![])
    }

    /// Append the messages produced by `with` to the chain.
    ///
    /// The method takes `self` **by value** to encourage concise
    /// call-chaining:
    ///
    /// ```rust
    /// # use artificial_prompt::chain::PromptChain;
    /// # use artificial_core::generic::{GenericMessage, GenericRole};
    /// #
    /// # let msg = GenericMessage::new("hi".into(), GenericRole::User);
    /// let vec = PromptChain::new()
    ///     .with(msg)
    ///     .build();
    /// ```
    pub fn with(mut self, with: impl IntoPrompt<Message = Message>) -> Self {
        self.0.append(&mut with.into_prompt());
        self
    }

    /// Consume the builder and return the accumulated messages.
    pub fn build(self) -> Vec<Message> {
        self.0
    }
}
