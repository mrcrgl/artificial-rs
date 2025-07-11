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

/// Lightweight container representing a single chat message that is
/// independent of any specific LLM provider.
///
/// * `message` – the raw UTF-8 content. Markdown is fine, but keep newlines
///   and indentation portable.
/// * `role` – see [`GenericRole`] for permitted values.
#[derive(Debug, Clone)]
pub struct GenericMessage {
    pub message: String,
    pub role: GenericRole,
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
        Self { message, role }
    }
}

/// High-level chat roles recognised by most LLM providers.
///
/// The `Display` implementation renders the canonical lowercase name so you
/// can feed it directly into JSON without extra mapping logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
