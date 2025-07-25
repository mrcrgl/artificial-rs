//! A minimal fragment that injects a *static* string into the prompt.
//!
//! Use this when you have pre-determined text (role description, safety
//! notice, system instruction …) that never changes between invocations.
//!
//! ```rust
//! use artificial_types::fragments::StaticFragment;
//! use artificial_core::generic::GenericRole;
//!
//! let sys_msg = StaticFragment::new(
//!     "You are a multilingual proof-reading engine.",
//!     GenericRole::System,
//! );
//! ```
//!
//! # Why a dedicated type?
//!
//! 1. It keeps the [`IntoPrompt`] API symmetrical – every fragment, no matter
//!    how simple, implements the same trait.
//! 2. You can attach metadata (`role`) so the provider sees the correct
//!    message type without manual wrapping.
//! 3. Unlike `&'static str`, this struct can carry a *borrowed* slice with
//!    lifetime `'a`, allowing the caller to reference larger inline strings
//!    without `String` allocation.
//!
//! The `From<&str>` blanket impl defaults to `GenericRole::System` for
//! convenience since system messages are the most common static fragments.

use artificial_core::{
    generic::{GenericMessage, GenericRole},
    template::IntoPrompt,
};

/// A borrowed static string bundled with an LLM chat role.
///
/// The tuple struct keeps the footprint at *exactly two machine words*
/// (`&str` + `GenericRole`) while still offering ergonomic constructors.
pub struct StaticFragment<'a>((&'a str, GenericRole));

/// Shorthand so you can write `StaticFragment::from("…")` without specifying
/// the role each time.  Defaults to **system**.
impl<'a> From<&'a str> for StaticFragment<'a> {
    fn from(value: &'a str) -> Self {
        Self((value, GenericRole::System))
    }
}

impl<'a> StaticFragment<'a> {
    /// Create a new fragment with explicit role.
    pub fn new(value: &'a str, role: GenericRole) -> Self {
        Self((value, role))
    }
}

impl IntoPrompt for StaticFragment<'_> {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        vec![Self::Message {
            role: self.0.1,
            name: None,
            message: self.0.0.to_string(),
        }]
    }
}
