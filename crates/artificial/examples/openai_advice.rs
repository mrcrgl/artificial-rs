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

use artificial::generic::ResponseContent;
use artificial::openai::OpenAiAdapterBuilder;
use artificial::prompt::chain::PromptChain;
use artificial::types::{fragments::StaticFragment, outputs::result::ThinkResult};
use artificial::{
    ArtificialClient,
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    provider::PromptExecutionProvider as _,
    template::PromptTemplate,
};
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
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;

    let client = ArtificialClient::new(backend);

    let history = [
        ("I failed my Rust borrow checker again.", GenericRole::User),
        ("Keep calm and add more lifetimes.", GenericRole::Assistant),
        ("Any other tips?", GenericRole::User),
    ];

    let response = client.prompt_execute(AdvicePrompt::new(&history)).await?;

    let ResponseContent::Finished(content) = response.content else {
        panic!("expected finished");
    };

    println!("Status: {:?}", content.status);
    println!("Reasoning: {}", content.reasoning);
    println!("Confidence: {}", content.confidence);
    println!(
        "LLM says:\n {}",
        content.data.map(|d| d.suggestion).unwrap_or_default()
    );
    if let Some(usage) = response.usage {
        println!(
            "Tokens – prompt: {}, completion: {}, total: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    Ok(())
}
