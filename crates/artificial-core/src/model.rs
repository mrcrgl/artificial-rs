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

use std::str::FromStr;

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
    Gpt4_1,
    Gpt4_1Mini,
    Gpt4_1Nano,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelParseError(pub String);

impl std::fmt::Display for ModelParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown model identifier: {}", self.0)
    }
}

impl std::error::Error for ModelParseError {}

impl AsRef<str> for OpenAiModel {
    fn as_ref(&self) -> &str {
        match self {
            OpenAiModel::Gpt5 => "gpt-5",
            OpenAiModel::Gpt5Nano => "gpt-5-nano",
            OpenAiModel::Gpt5Mini => "gpt-5-mini",
            OpenAiModel::Gpt5_1 => "gpt-5.1",
            OpenAiModel::Gpt5_2 => "gpt-5.2",
            OpenAiModel::Gpt4_1 => "gpt-4.1",
            OpenAiModel::Gpt4_1Mini => "gpt-4.1-mini",
            OpenAiModel::Gpt4_1Nano => "gpt-4.1-nano",
            OpenAiModel::Gpt4o => "gpt-4o",
            OpenAiModel::Gpt4oMini => "gpt-4o-mini",
            OpenAiModel::O3 => "o3",
            OpenAiModel::O3Mini => "o3-mini",
            OpenAiModel::O4Mini => "o4-mini",
        }
    }
}

impl FromStr for OpenAiModel {
    type Err = ModelParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gpt-5" => Ok(OpenAiModel::Gpt5),
            "gpt-5-nano" => Ok(OpenAiModel::Gpt5Nano),
            "gpt-5-mini" => Ok(OpenAiModel::Gpt5Mini),
            "gpt-5.1" => Ok(OpenAiModel::Gpt5_1),
            "gpt-5.2" => Ok(OpenAiModel::Gpt5_2),
            "gpt-4.1" => Ok(OpenAiModel::Gpt4_1),
            "gpt-4.1-mini" => Ok(OpenAiModel::Gpt4_1Mini),
            "gpt-4.1-nano" => Ok(OpenAiModel::Gpt4_1Nano),
            "gpt-4o" => Ok(OpenAiModel::Gpt4o),
            "gpt-4o-mini" => Ok(OpenAiModel::Gpt4oMini),
            "o3" => Ok(OpenAiModel::O3),
            "o3-mini" => Ok(OpenAiModel::O3Mini),
            "o4-mini" => Ok(OpenAiModel::O4Mini),
            _ => Err(ModelParseError(s.to_string())),
        }
    }
}

impl AsRef<str> for Model {
    fn as_ref(&self) -> &str {
        match self {
            Model::OpenAi(model) => model.as_ref(),
            Model::Custom(custom) => custom,
        }
    }
}

impl FromStr for Model {
    type Err = ModelParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Model::OpenAi(OpenAiModel::from_str(s)?))
    }
}

#[cfg(test)]
mod tests {
    use super::{Model, OpenAiModel};
    use std::str::FromStr;

    #[test]
    fn openai_model_from_str_roundtrips() {
        let models = [
            "gpt-5",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "o3",
            "o3-mini",
            "o4-mini",
        ];

        for model in models {
            let parsed = OpenAiModel::from_str(model).expect("model should parse");
            assert_eq!(parsed.as_ref(), model);
        }
    }

    #[test]
    fn model_as_ref_covers_openai_and_custom() {
        let openai = Model::OpenAi(OpenAiModel::Gpt5Mini);
        assert_eq!(openai.as_ref(), "gpt-5-mini");

        let custom = Model::Custom("provider:custom-1");
        assert_eq!(custom.as_ref(), "provider:custom-1");
    }
}
