use artificial::openai::OpenAiAdapterBuilder;
use artificial::prompt::chain::PromptChain;
use artificial::types::fragments::StaticFragment;
use artificial::{
    ArtificialClient,
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    provider::PromptExecutionProvider as _,
    template::{IntoPrompt, PromptTemplate},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// # Hello, JSON! – Minimal yet *typed* prompt
///
/// This example is the “smallest viable program” that
///
/// 1. **Builds** an OpenAI backend (`OpenAiAdapter`).
/// 2. **Creates** a prompt consisting of two messages:
///    * a *system* instruction (loaded from the shared `base_system.md`)
///    * a *user* request (“Mayday Mayday!”)
/// 3. **Asks** the model to respond with **valid JSON** that can be
///    deserialised into the `HelloResponse` struct below.
/// 4. **Prints** the strongly-typed Rust value.
///
/// ## How to run
///
/// ```bash
/// export OPENAI_API_KEY=sk-…          # your key, free tier works fine
/// cargo run -p artificial --example openai_hello_world
/// ```
///
/// You should see output similar to:
///
/// ```text
/// Response: HelloResponse { greeting: "Beep-boop, assistance on the way!" }
/// ```
///
/// Adjust the model (`OpenAiModel::*`) or the messages as desired.
///
/// ## Note on the schema pipeline
///
/// Because `HelloResponse` implements [`schemars::JsonSchema`] and is used as
/// `PromptTemplate::Output`, the OpenAI backend automatically injects the JSON
/// schema in the request (`response_format = json_schema`), so the LLM can
/// *only* reply with valid JSON that matches our struct.
////////////////////////////////////////////////////////////////////////////////

/// “Base system” instructions that every prompt in this workspace adds.
/// The file now contains a fun but precise R2-D2 operating manual.
const BASE_SYSTEM_ROLE: &str = include_str!("data/role/base_system.md");

/// The *shape* of the answer we expect from the model.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct HelloResponse {
    greeting: String,
}

/// A tiny prompt that leverages `PromptChain` to showcase fragment composition.
struct HelloPrompt;

impl IntoPrompt for HelloPrompt {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        PromptChain::new()
            // Shared “system” fragment with general behavioural rules.
            .with(StaticFragment::from(BASE_SYSTEM_ROLE))
            // The actual user request.
            .with(StaticFragment::new("Mayday Mayday!", GenericRole::User))
            .build()
    }
}

/// Tell `artificial-core` which model we want and which type we expect back.
impl PromptTemplate for HelloPrompt {
    type Output = HelloResponse;
    const MODEL: Model = Model::OpenAi(OpenAiModel::Gpt4oMini);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Build the backend from the environment (needs OPENAI_API_KEY).
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;

    // 2. Wrap it inside the generic client.
    let client = ArtificialClient::new(backend);

    // 3. Run our prompt and await the typed result.
    let response = client.prompt_execute(HelloPrompt).await?;

    // 4. Done – enjoy a well-typed greeting from the galaxy.
    println!("Response: {response:?}");

    Ok(())
}
