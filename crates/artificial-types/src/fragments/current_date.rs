//! A small **prompt fragment** that injects the current date and time into a
//! `GenericMessage`.
//!
//! Many tasks benefit from temporal context—think “remind me in three days” or
//! “schedule for next Wednesday”.  Hard-coding the timestamp at call-site is
//! brittle and violates DRY, so this helper does the job for you.
//!
//! # What it adds
//!
//! ```markdown
//! **Current ISO Timestamp**: 2025-04-20T12:34:56Z
//! **Current Date and Time**: 2025-04-20 12:34:56
//! **Current Weekday**: Sunday
//! **Timezone**: UTC
//!
//! You are currently reasoning in the context of Sunday, 2025-04-20, 12:34:56, UTC
//!
//! Use this information when interpreting natural language expressions like
//! 'next week' or 'in 3 days'.
//! ```
//!
//! # Example
//!
//! ```rust
//! use artificial_types::fragments::CurrentDateFragment;
//! use artificial_prompt::chain::PromptChain;
//!
//! let messages = PromptChain::new()
//!     .with(CurrentDateFragment::new())
//!     .build();
//!
//! assert_eq!(messages[0].role.to_string(), "system");
//! ```
//!
//! The fragment is fully **stateless**—you can create and reuse it as often as
//! needed without side effects.

use artificial_core::{
    generic::{GenericMessage, GenericRole},
    template::IntoPrompt,
};
use artificial_prompt::builder::PromptBuilder;
use chrono::Datelike as _;

/// Injects the current UTC timestamp/date/weekday as a system message.
#[derive(Default)]
pub struct CurrentDateFragment;

impl CurrentDateFragment {
    /// Convenience constructor (equivalent to `Self::default()`).
    pub fn new() -> Self {
        Self
    }
}

impl IntoPrompt for CurrentDateFragment {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        let now = chrono::Utc::now();

        let builder = PromptBuilder::new()
            .add_key_value("Current ISO Timestamp", now.to_rfc3339())
            .add_key_value("Current Date and Time", now.format("%Y-%m-%d %H:%M:%S"))
            .add_key_value("Current Weekday", now.weekday().to_string())
            .add_key_value("Timezone", "UTC")
            .add_line(format!(
                "You are currently reasoning in the context of {}, {}, {}, {}",
                now.weekday(),
                now.format("%Y-%m-%d"),
                now.format("%H:%M:%S"),
                now.timezone()
            ))
            .add_blank_line()
            .add_line(
                "Use this information when interpreting natural language \
                 expressions like 'next week' or 'in 3 days'.",
            );

        vec![GenericMessage::new(builder.finalize(), GenericRole::System)]
    }
}
