//! # Ask the Droid – Context-Aware Advice Example
//!
//! This example demonstrates three core ideas that run through the entire
//! *artificial-rs* workspace:
//!
//! 1. **Prompt composition** with [`PromptChain`]. We glue together a shared
//!    *system* instruction, followed by an arbitrary chat history.
//! 2. **Typed responses** using [`ThinkResult<T>`]. The model must answer with
//!    JSON that deserialises into `Advice`, **and** add reasoning / confidence
//!    metadata on top.
//! 3. **Drop-in back-ends**. By targeting the [`Backend`] trait we can swap the
//!    OpenAI adapter for any other provider without changing user code.
//!
//! ```text
//! ┌──────────┐      ┌──────────────────┐      ┌──────────────────┐
//! │  Prompt  │ ===► │  OpenAiAdapter   │ ===► │  ThinkResult<…>  │
//! └──────────┘      └──────────────────┘      └──────────────────┘
//! ```
//!
//! ## Running the example
//!
//! ```bash
//! export OPENAI_API_KEY=sk-…      # mandatory
//! cargo run -p artificial --example openai_advice
//! ```
//!
//! Expected output (truncated):
//!
//! ```text
//! Status: Succeed
//! Reasoning: Selected the most actionable suggestion.
//! Confidence: 0.82
//! LLM says:
//!  "Try visualising lifetimes as a graph; it clarifies borrow scopes."
//! ```
//!
//! The `status / reasoning / confidence` fields are filled by the LLM itself,
//! which makes the response self-auditable.
//!
//! ## Note
//!
//! * `OpenAiModel::Gpt4o` is overkill; feel free to downgrade if you prefer
//!   cheaper tokens.
//! * Replace `Advice` with `serde_json::Value` if you don’t care about schema
//!   validation—but you really should!
//!
//! ---------------------------------------------------------------------------

use artificial::{
    ArtificialClient,
    provider::ChatCompletionProvider as _,
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    template::PromptTemplate,
};
use artificial_openai::OpenAiAdapterBuilder;
use artificial_prompt::chain::PromptChain;
use artificial_types::{fragments::StaticFragment, outputs::result::ThinkResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Common “base system” rules shared by all examples (R2-D2 operating manual).
const BASE_SYSTEM_ROLE: &str = include_str!("data/role/base_system.md");

/// What we want the LLM to emit **inside** the `ThinkResult::data` field.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Advice {
    /// A single, punchy suggestion.
    suggestion: String,
}

/// High-level prompt wrapper that carries the chat history.
struct AdvicePrompt {
    history: Vec<GenericMessage>,
}

impl AdvicePrompt {
    /// Convert an array of `(text, role)` tuples into typed messages.
    ///
    /// ```rust
    /// let history = [
    ///   ("I failed my Rust borrow checker again.", GenericRole::User),
    ///   ("Have you tried more lifetimes?", GenericRole::Assistant),
    /// ];
    /// let prompt = AdvicePrompt::new(&history);
    /// ```
    fn new(history: &[(&str, GenericRole)]) -> Self {
        let msgs = history
            .iter()
            .map(|(txt, role)| GenericMessage::new((*txt).into(), *role))
            .collect();

        Self { history: msgs }
    }
}

/// Convert the prompt into a vector of provider-agnostic messages.
impl artificial::template::IntoPrompt for AdvicePrompt {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        // Start with the shared system fragment, then append chat history.
        let mut chain = PromptChain::new().with(StaticFragment::from(BASE_SYSTEM_ROLE));

        for message in self.history {
            chain = chain.with(message);
        }

        chain.build()
    }
}

/// Tell the compiler which model we want and which type to deserialize into.
impl PromptTemplate for AdvicePrompt {
    type Output = ThinkResult<Advice>;
    const MODEL: Model = Model::OpenAi(OpenAiModel::Gpt4o);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Build backend (needs `OPENAI_API_KEY` in environment)
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;

    // 2. Wrap backend in a generic client
    let client = ArtificialClient::new(backend);

    // 3. Provide some interesting chat history
    let history = [
        ("I failed my Rust borrow checker again.", GenericRole::User),
        ("Keep calm and add more lifetimes.", GenericRole::Assistant),
        ("Any other tips?", GenericRole::User),
    ];

    // 4. Execute the prompt
    let advice = client.chat_complete(AdvicePrompt::new(&history)).await?;

    // 5. Print structured response
    println!("Status: {:?}", advice.status);
    println!("Reasoning: {}", advice.reasoning);
    println!("Confidence: {}", advice.confidence);
    println!(
        "LLM says:\n {}",
        advice.data.map(|d| d.suggestion).unwrap_or_default()
    );

    Ok(())
}
