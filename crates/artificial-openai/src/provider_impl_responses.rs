use std::{future::Future, pin::Pin, sync::Arc};

use async_stream::try_stream;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::{Map, Value};

use artificial_core::{
    error::{ArtificialError, Result},
    generic::{
        GenericChatCompletionResponse, GenericMessage, GenericRole, GenericUsageReport,
        ResponseContent, StreamEvent,
    },
    provider::StreamingEventsProvider,
    provider::{ChatCompleteParameters, ChatCompletionProvider, StreamingChatProvider},
};

use crate::{
    OpenAiAdapter,
    api_v1::{ChatCompletionMessage, Content, MessageRole, ResponseStreamEvent, ResponsesRequest},
    error::OpenAiError,
    model_map::map_model,
};

/// Responses-based provider implementation for OpenAI (non-streaming and streaming).
///
/// This replaces the legacy chat/completions pipeline and targets `/v1/responses`
/// so that reasoning models (o-series) and new features are supported.
impl ChatCompletionProvider for OpenAiAdapter {
    /// Keep using the existing provider-specific message type to minimise conversion friction.
    type Message = ChatCompletionMessage;

    fn chat_complete<'p, M>(
        &self,
        params: ChatCompleteParameters<M>,
    ) -> Pin<
        Box<dyn Future<Output = Result<GenericChatCompletionResponse<GenericMessage>>> + Send + 'p>,
    >
    where
        M: Into<Self::Message> + Send + Sync + 'p,
    {
        let client = Arc::clone(&self.client);

        Box::pin(async move {
            // Map model to provider name.
            let model = map_model(&params.model)
                .ok_or_else(|| {
                    ArtificialError::InvalidRequest(format!(
                        "backend does not support selected model: {:?}",
                        params.model
                    ))
                })?
                .to_string();

            // Convert provider messages -> Responses API messages JSON shape.
            let messages: Vec<ChatCompletionMessage> =
                params.messages.into_iter().map(Into::into).collect();
            let messages_json = to_responses_messages(&messages)?;

            // Build Responses request.
            let mut req = ResponsesRequest::new(model);
            req.messages = Some(messages_json);
            req.temperature = params.temperature;
            req.response_format = params.response_format;

            // Execute non-streaming call.
            let mut resp = client.response(req).await?;

            // Usage (best-effort mapping).
            let usage = resp
                .extra
                .get("usage")
                .cloned()
                .or(resp.usage.clone())
                .and_then(extract_usage);

            // Extract assistant text from Responses output (best-effort).
            let text = resp
                .output
                .as_ref()
                .and_then(extract_output_text)
                .unwrap_or_default();

            let message = GenericMessage {
                role: GenericRole::Assistant,
                content: Some(text),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            };

            Ok(GenericChatCompletionResponse {
                content: ResponseContent::Finished(message),
                usage,
            })
        })
    }
}

impl StreamingChatProvider for OpenAiAdapter {
    type Delta<'s>
        = Pin<Box<dyn Stream<Item = Result<String>> + Send + 's>>
    where
        Self: 's;

    fn chat_complete_stream<'p, M>(&self, params: ChatCompleteParameters<M>) -> Self::Delta<'p>
    where
        M: Into<Self::Message> + Send + Sync + 'p,
    {
        let client = self.client.clone();

        Box::pin(try_stream! {
            // Map model and messages
            let model = map_model(&params.model)
                .ok_or_else(|| ArtificialError::InvalidRequest(format!(
                    "backend does not support selected model: {:?}",
                    params.model
                )))?
                .to_string();

            let messages: Vec<ChatCompletionMessage> =
                params.messages.into_iter().map(Into::into).collect();
            let messages_json = to_responses_messages(&messages)?;

            let mut req = ResponsesRequest::new(model);
            req.messages = Some(messages_json);
            req.temperature = params.temperature;
            req.response_format = params.response_format;
            req.stream = Some(true);

            let stream = client.response_stream(req);
            futures_util::pin_mut!(stream);

            while let Some(evt) = stream.next().await {
                match evt.map_err(ArtificialError::from)? {
                    ResponseStreamEvent::OutputTextDelta { delta, .. } => {
                        if !delta.is_empty() {
                            yield delta;
                        }
                    }
                    ResponseStreamEvent::Completed { .. } => {
                        break;
                    }
                    ResponseStreamEvent::Error { error, .. } => {
                        Err(OpenAiError::Unknown(format!("responses stream error: {error}")))?
                    }
                    _ => {
                        // ignore other events in the text-only stream
                    }
                }
            }
        })
    }
}

impl StreamingEventsProvider for OpenAiAdapter {
    type EventStream<'s>
        = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + 's>>
    where
        Self: 's;

    fn chat_complete_events_stream<'p, M>(
        &self,
        params: ChatCompleteParameters<M>,
    ) -> Self::EventStream<'p>
    where
        M: Into<Self::Message> + Send + Sync + 'p,
    {
        let client = self.client.clone();

        Box::pin(try_stream! {
            // Map model and messages
            let model = map_model(&params.model)
                .ok_or_else(|| ArtificialError::InvalidRequest(format!(
                    "backend does not support selected model: {:?}",
                    params.model
                )))?
                .to_string();

            let messages: Vec<ChatCompletionMessage> =
                params.messages.into_iter().map(Into::into).collect();
            let messages_json = to_responses_messages(&messages)?;

            let mut req = ResponsesRequest::new(model);
            req.messages = Some(messages_json);
            req.temperature = params.temperature;
            req.response_format = params.response_format;
            req.stream = Some(true);

            let stream = client.response_stream(req);
            futures_util::pin_mut!(stream);

            while let Some(evt) = stream.next().await {
                match evt.map_err(ArtificialError::from)? {
                    ResponseStreamEvent::OutputTextDelta { delta, .. } => {
                        if !delta.is_empty() {
                            yield StreamEvent::TextDelta(delta);
                        }
                    }
                    ResponseStreamEvent::OutputTextDone { .. } => {
                        // no-op; we'll send MessageEnd on Completed
                    }
                    ResponseStreamEvent::Completed { usage, .. } => {
                        if let Some(u) = usage.and_then(extract_usage) {
                            yield StreamEvent::Usage(u);
                        }
                        yield StreamEvent::MessageEnd;
                        break;
                    }
                    ResponseStreamEvent::Error { error, .. } => {
                        Err(OpenAiError::Unknown(format!("responses stream error: {error}")))?
                    }
                    _ => {
                        // ignore other events for now
                    }
                }
            }
        })
    }
}

/// Convert existing provider message type into Responses API "messages" JSON shape:
/// [
///   { "role": "system"|"user"|"assistant"|"tool",
///     "content": [ { "type": "text", "text": "..." } ] }
/// ]
fn to_responses_messages(msgs: &[ChatCompletionMessage]) -> Result<Value> {
    let mut arr = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = match m.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool | MessageRole::Function => "tool",
        };

        // Map text content
        let text = match &m.content {
            Some(Content::Text(s)) => s.clone(),
            None => String::new(),
        };

        let content_blocks = vec![json_obj(&[
            ("type", Value::String("text".into())),
            ("text", Value::String(text)),
        ])];

        let msg_obj = json_obj(&[
            ("role", Value::String(role.into())),
            ("content", Value::Array(content_blocks)),
        ]);

        arr.push(msg_obj);
    }
    Ok(Value::Array(arr))
}

/// Extract a concatenated text from the Responses `output` value.
/// Best-effort for typical shapes:
/// - Array of blocks: [{ "type": "output_text", "text": "..." }, ...]
/// - Or direct object with "text".
fn extract_output_text(output: &Value) -> Option<String> {
    match output {
        Value::Array(items) => {
            let mut buf = String::new();
            for it in items {
                if let Some(obj) = it.as_object() {
                    match obj.get("type").and_then(|v| v.as_str()) {
                        Some("output_text") | None => {
                            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                                buf.push_str(text);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some(buf)
        }
        Value::Object(obj) => {
            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                Some(text.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract token usage from a flexible `usage` object.
/// Tries both Responses-style and Chat-style naming.
fn extract_usage(usage: Value) -> Option<GenericUsageReport> {
    match usage {
        Value::Object(obj) => extract_usage_from_obj(&obj),
        _ => None,
    }
}

fn extract_usage_from_obj(obj: &Map<String, Value>) -> Option<GenericUsageReport> {
    // Try Responses-style keys
    let input = obj
        .get("input_tokens")
        .and_then(Value::as_i64)
        .or_else(|| obj.get("prompt_tokens").and_then(Value::as_i64));
    let output = obj
        .get("output_tokens")
        .and_then(Value::as_i64)
        .or_else(|| obj.get("completion_tokens").and_then(Value::as_i64));
    let total = obj
        .get("total_tokens")
        .and_then(Value::as_i64)
        .or_else(|| match (input, output) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        });

    match (input, output, total) {
        (Some(i), Some(o), Some(t)) => Some(GenericUsageReport {
            prompt_tokens: i,
            completion_tokens: o,
            total_tokens: t,
        }),
        _ => None,
    }
}

fn json_obj(entries: &[(&str, Value)]) -> Value {
    let mut map = Map::new();
    for (k, v) in entries {
        map.insert((*k).to_string(), v.clone());
    }
    Value::Object(map)
}
