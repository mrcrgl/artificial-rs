use artificial::openai::OpenAiAdapterBuilder;
use artificial::{
    ArtificialClient,
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    provider::{ChatCompleteParameters, ChatCompletionProvider as _},
};

/// # Chat Completion – Direct `chat_complete` Example
///
/// This example bypasses the higher-level [`PromptExecutionProvider`]
/// abstraction and calls [`ChatCompletionProvider::chat_complete`] directly.
/// That means:
///
/// 1. **You** assemble the full list of chat messages.
/// 2. **You** pick the model.
/// 3. The backend returns a [`GenericChatCompletionResponse`] that contains a
///    single assistant message (plus token usage statistics).
///
/// ```bash
/// export OPENAI_API_KEY=sk-…      # mandatory
/// cargo run -p artificial --example openai_chat_complete
/// ```
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;

    let client = ArtificialClient::new(backend);

    let messages = vec![
        GenericMessage::new(
            "You are a concise, witty assistant.".into(),
            GenericRole::System,
        ),
        GenericMessage::new(
            "Why is the Rust borrow checker important?".into(),
            GenericRole::User,
        ),
    ];

    let params = ChatCompleteParameters::new(messages, Model::OpenAi(OpenAiModel::Gpt4oMini));

    let response = client.chat_complete(params).await?;

    if let Some(answer) = response.content.content {
        println!("Assistant: {answer}");
    } else {
        println!("Assistant returned no textual content 🤖");
    }

    if let Some(usage) = response.usage {
        println!(
            "Tokens – prompt: {}, completion: {}, total: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    Ok(())
}
