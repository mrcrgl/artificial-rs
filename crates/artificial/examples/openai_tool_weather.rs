use artificial::generic::{GenericFunctionSpec, ResponseContent};
use artificial::openai::OpenAiAdapterBuilder;
use artificial::{
    ArtificialClient,
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    provider::{ChatCompleteParameters, ChatCompletionProvider as _},
};

/// ---------------------------------------------------------------------------
/// Example  –  OpenAI “Function Calling”
///
/// **Running the demo**
/// ```bash
/// export OPENAI_API_KEY=sk-…      # mandatory
/// cargo run -p artificial --example openai_tool_weather
/// ```
/// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;
    let client = ArtificialClient::new(backend);

    let weather_api_tool = GenericFunctionSpec {
        name: "current_weather".to_string(),
        description: "Fetch the current weather report (temperature in °C and condition)."
            .to_string(),
        parameters: serde_json::json!({
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location", "unit"],
          "additionalProperties": false
        }),
    };

    let mut messages = vec![];

    messages.push(GenericMessage::new(
        "What’s the current weather in Berlin?".into(),
        GenericRole::User,
    ));

    loop {
        let params =
            ChatCompleteParameters::new(messages.clone(), Model::OpenAi(OpenAiModel::Gpt4oMini))
                .with_tools(vec![weather_api_tool.clone()]);

        let response = client.chat_complete(params).await?;

        let message = match &response.content {
            ResponseContent::Finished(message) => message,
            ResponseContent::ToolCalls(message) => message,
        };

        messages.push(message.clone());

        match response.content {
            ResponseContent::ToolCalls(generic_message) => {
                if let Some(tool_calls) = &generic_message.tool_calls {
                    for tool_call in tool_calls {
                        match tool_call.function.name.as_str() {
                            "current_weather" => {
                                let location = tool_call
                                    .function
                                    .arguments
                                    .as_object()
                                    .and_then(|o| o.get("location"))
                                    .and_then(|o| o.as_str())
                                    .map(String::from)
                                    .unwrap_or("Berlin".to_string());
                                let unit = tool_call
                                    .function
                                    .arguments
                                    .as_object()
                                    .and_then(|o| o.get("unit"))
                                    .and_then(|o| o.as_str())
                                    .map(String::from);

                                messages.push(
                                    GenericMessage::new(
                                        get_weather(location, unit),
                                        GenericRole::Tool,
                                    )
                                    .with_tool_call_id(tool_call.id.clone()),
                                );
                                continue;
                            }
                            other => panic!("tool not registered: {other}"),
                        }
                    }
                }
            }
            ResponseContent::Finished(content) => {
                if let Some(answer) = content.content {
                    println!("LLM answered:\n{answer}");
                    break;
                } else {
                    println!("Assistant returned no textual content 🤖");
                }
            }
        }
    }

    Ok(())
}

fn get_weather(location: String, unit: Option<String>) -> String {
    serde_json::json!({
        "location": location,
        "value": 32.2,
        "unit": unit.unwrap_or("celsius".to_string())
    })
    .to_string()
}
