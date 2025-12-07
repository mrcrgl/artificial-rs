//! # Streaming Chat Completion with Tool Calls
//!
//! This example demonstrates how to consume streaming events (text deltas and
//! tool-call intents) and perform a tool round-trip mid-conversation using
//! `StreamingEventsProvider`.
//!
//! Usage:
//!   export OPENAI_API_KEY=sk-…
//!   cargo run -p artificial --example openai_chat_stream_tools --features openai
//!
//! You should see partial assistant text, then a tool-call being executed, and
//! finally the assistant’s follow-up message streamed to the terminal.
//!
//! Notes:
//! - The example uses a mocked `get_weather` tool. Replace it with real logic.
//! - The assistant can decide to call the tool; if it does, we handle it and
//!   continue the chat with the tool result injected.

use artificial::openai::OpenAiAdapterBuilder;
use artificial::{
    StreamingEventsProvider as _,
    generic::{GenericFunctionSpec, GenericMessage, GenericRole, StreamEvent},
    model::{Model, OpenAiModel},
    provider::ChatCompleteParameters,
};
use futures_util::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1) Backend: OpenAI adapter (reads OPENAI_API_KEY)
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;

    // 2) Define a tool: current_weather
    let weather_api_tool = GenericFunctionSpec {
        name: "current_weather".to_string(),
        description: "Fetch the current weather report (temperature in °C and condition)."
            .to_string(),
        parameters: serde_json::json!({
          "type": "object",
          "additionalProperties": false,
          "required": ["location", "unit"],
          "properties": {
            "location": { "type": "string", "description": "City name, e.g. Berlin" },
            "unit":     { "type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius" }
          }
        }),
    };

    // 3) Initial messages
    let mut messages = vec![
        GenericMessage::new(
            "You are a helpful assistant that uses tools. Keep replies short.".into(),
            GenericRole::System,
        ),
        GenericMessage::new(
            "What's the weather like in Berlin in celsius?".into(),
            GenericRole::User,
        ),
    ];

    // 4) First streaming call: model may emit text and/or tool-call intents
    let params =
        ChatCompleteParameters::new(messages.clone(), Model::OpenAi(OpenAiModel::Gpt4oMini))
            .with_tools(vec![weather_api_tool.clone()]);

    let mut stream = backend.chat_complete_events_stream(params);

    print!("Assistant: ");
    io::stdout().flush().ok();

    // Collect tool-call intents so we can execute them after the model finishes its turn.
    let mut tool_intents: Vec<artificial::generic::GenericFunctionCallIntent> = Vec::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::TextDelta(s)) => {
                // Render partial text
                print!("{s}");
                io::stdout().flush().ok();
            }
            Ok(StreamEvent::ToolCallStart { index, id, name }) => {
                // You can log/debug these; not strictly necessary for execution.
                eprintln!("\n[debug] tool-call[{index}] start: id={id:?}, name={name:?}");
            }
            Ok(StreamEvent::ToolCallArgumentsDelta {
                index,
                arguments_fragment,
            }) => {
                // Optional: show streamed JSON fragments for debugging.
                eprintln!("\n[debug] tool-call[{index}] args += {arguments_fragment:?}");
            }
            Ok(StreamEvent::ToolCallComplete { index, intent }) => {
                eprintln!(
                    "\n[debug] tool-call[{index}] complete: {} {:?}",
                    intent.function.name, intent.function.arguments
                );
                tool_intents.push(intent);
            }
            Ok(StreamEvent::MessageEnd) => {
                break;
            }
            Ok(StreamEvent::Usage(_usage)) => {
                // Not currently surfaced by the OpenAI implementation during streaming;
                // kept for API completeness. You can print usage here if provided.
            }
            Err(e) => {
                eprintln!("\n\nError while streaming: {e}");
                return Ok(());
            }
        }
    }

    // 5) If the model requested tool calls, execute them and continue the conversation.
    if !tool_intents.is_empty() {
        // Push the assistant message carrying the tool-calls into the history,
        // so the model can attribute tool results to the correct call IDs.
        messages.push(GenericMessage {
            content: None,
            role: GenericRole::Assistant,
            name: None,
            tool_calls: Some(tool_intents.clone()),
            tool_call_id: None,
        });

        // Execute tool calls and push tool results to the conversation
        for intent in &tool_intents {
            match intent.function.name.as_str() {
                "current_weather" => {
                    let (location, unit) = {
                        let args = &intent.function.arguments;
                        let location = args
                            .as_object()
                            .and_then(|o| o.get("location"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("Berlin")
                            .to_string();

                        let unit = args
                            .as_object()
                            .and_then(|o| o.get("unit"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("celsius")
                            .to_string();

                        (location, unit)
                    };

                    let tool_output = get_weather(location, unit);

                    // The tool's result is a message with role "tool" and the tool_call_id set.
                    messages.push(
                        GenericMessage::new(tool_output, GenericRole::Tool)
                            .with_tool_call_id(intent.id.clone()),
                    );
                }
                other => {
                    eprintln!("[warn] Unsupported tool requested: {other}");
                }
            }
        }

        // 6) Second streaming call: the model sees tool results and produces the final answer
        let params2 = ChatCompleteParameters::new(messages, Model::OpenAi(OpenAiModel::Gpt4oMini))
            .with_tools(vec![weather_api_tool]);

        let mut stream2 = backend.chat_complete_events_stream(params2);

        print!("\nAssistant (after tool): ");
        io::stdout().flush().ok();

        while let Some(event) = stream2.next().await {
            match event {
                Ok(StreamEvent::TextDelta(s)) => {
                    print!("{s}");
                    io::stdout().flush().ok();
                }
                Ok(StreamEvent::MessageEnd) => break,
                Ok(_) => {}
                Err(e) => {
                    eprintln!("\n\nError while streaming follow-up: {e}");
                    break;
                }
            }
        }

        println!();
    } else {
        // No tools needed; we already streamed the final answer above.
        println!();
    }

    Ok(())
}

/// Mock implementation of `current_weather`.
///
/// Replace this with a real API call if desired. The return value should be a
/// JSON string compatible with the tool contract returned by the model.
fn get_weather(location: String, unit: String) -> String {
    // Fake a tiny response
    let (temperature, condition) = match location.to_lowercase().as_str() {
        "berlin" => (12, "Cloudy"),
        "london" => (10, "Rain"),
        "san francisco" => (16, "Fog"),
        _ => (20, "Sunny"),
    };

    format!(
        r#"{{"location":"{location}","unit":"{unit}","temperature":{temperature},"condition":"{condition}"}}"#
    )
}
