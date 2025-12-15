//! Model identifiers used throughout the **artificial** workspace.
//!
//! The enum hierarchy keeps the *public* API blissfully simple while allowing
//! each provider crate to map the variants onto its own naming scheme.  As a
//! consequence you never have to type literal strings such as `"gpt-4o-mini"`
//! in your application code—pick an enum variant instead and let the adapter
//! translate it.
//!
//! # Adding more models
//!
//! 1. **Provider–specific enum**
//!    Add the variant to the sub-enum (`OpenAiModel`, `AnthropicModel`, …).
//! 2. **Mapping layer**
//!    Update the mapping function in the provider crate
//!    (`artificial-openai::model_map::map_model`, etc.).
//! 3. **Compile-time safety**
//!    The compiler will tell you if you forgot to handle the new variant in
//!    `From<T> for Model` or in provider match statements.
//!
//! # Example
//!
//! ```rust
//! use artificial_core::model::{Model, OpenAiModel};
//! assert_eq!(Model::from(OpenAiModel::Gpt4oMini),
//!            Model::OpenAi(OpenAiModel::Gpt4oMini));
//! ```

/// Universal identifier for an LLM model.
///
/// * `OpenAi` – Enumerated list of officially supported OpenAI models.
/// * `Custom` – Any provider / model name not yet covered by a dedicated enum. Use this if you run a self-hosted or beta model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Model {
    /// Built-in OpenAI models (chat completion API).
    OpenAi(OpenAiModel),
    /// Fully qualified provider/model ID (`"provider:model-name"` or similar).
    Custom(&'static str),
}

/// Exhaustive list of models **officially** supported by the OpenAI back-end.
///
/// Keeping the list small avoids accidental typos while still allowing
/// arbitrary model names through [`Model::Custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAiModel {
    Gpt5,
    Gpt5Nano,
    Gpt5Mini,
    Gpt5_1,
    Gpt5_2,
    Gpt4o,
    Gpt4oMini,
    O3,
    O3Mini,
    O4Mini,
}

impl From<OpenAiModel> for Model {
    fn from(val: OpenAiModel) -> Self {
        Model::OpenAi(val)
    }
}
