//! # Streaming Chat Completion – Real-time Example
//!
//! This example shows how to consume incremental text **deltas** from the
//! OpenAI back-end via [`StreamingChatProvider::chat_complete_stream`].
//!
//! Whereas `PromptExecutionProvider` collects the full reply before returning,
//! streaming lets you render partial output as soon as it arrives—perfect for
//! live terminals, web sockets, or any UX where latency matters.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-…      # mandatory
//! cargo run -p artificial --example openai_chat_stream
//! ```
//!
//! You should see the assistant’s reply appear character-by-character.
//!
//! ---------------------------------------------------------------------------

use artificial::openai::OpenAiAdapterBuilder;
use artificial::{
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    provider::{ChatCompleteParameters, StreamingChatProvider as _},
};
use futures_util::StreamExt; // for `next`
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Spin up the OpenAI back-end (needs `OPENAI_API_KEY` in the env).
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;

    // 2. Create a tiny conversation.
    let messages = vec![
        GenericMessage::new(
            "You are a real-time narrator. Respond sentence by sentence.".into(),
            GenericRole::System,
        ),
        GenericMessage::new(
            "Tell me a short story about Rustaceans exploring space.".into(),
            GenericRole::User,
        ),
    ];

    // 3. Bundle messages + model into `ChatCompleteParameters`.
    let params = ChatCompleteParameters::new(messages, Model::OpenAi(OpenAiModel::Gpt4oMini));

    // 4. Kick off the streaming request.
    let mut stream = backend.chat_complete_stream(params);

    // 5. Render the assistant’s output as it flows in.
    print!("Assistant: ");
    io::stdout().flush().ok();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                print!("{text}");
                io::stdout().flush().ok();
            }
            Err(e) => {
                eprintln!("\n\nError while streaming: {e}");
                break;
            }
        }
    }

    println!("\n\nStream finished ✅");
    Ok(())
}
