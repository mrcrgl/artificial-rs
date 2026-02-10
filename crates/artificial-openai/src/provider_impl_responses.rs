use std::{future::Future, pin::Pin, sync::Arc};

use async_stream::try_stream;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::{Map, Value};

use artificial_core::{
    error::{ArtificialError, Result},
    generic::{
        GenericChatCompletionResponse, GenericFunctionCall, GenericFunctionCallIntent,
        GenericFunctionSpec, GenericMessage, GenericRole, GenericUsageReport, ResponseContent,
        StreamEvent,
    },
    provider::StreamingEventsProvider,
    provider::{ChatCompleteParameters, ChatCompletionProvider, StreamingChatProvider},
};

use crate::{
    OpenAiAdapter,
    api_v1::{
        ChatCompletionMessage, Content, MessageRole, ResponseStreamEvent, ResponseTool,
        ResponseToolChoice, ResponseToolChoiceLiteral, ResponsesRequest,
    },
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
            req.input = Some(messages_json);
            req.temperature = params.temperature;
            req.response_format = params.response_format;
            if let Some(tools) = params.tools {
                let tool_vals: Vec<ResponseTool> =
                    tools.into_iter().map(tool_spec_from_generic).collect();
                req.tools = Some(tool_vals);
                req.tool_choice =
                    Some(ResponseToolChoice::Literal(ResponseToolChoiceLiteral::Auto));
            }

            // Execute non-streaming call.
            let mut resp = client.response(req).await?;

            // Usage (best-effort mapping).
            let usage = resp
                .extra
                .get("usage")
                .cloned()
                .or(resp.usage.clone())
                .and_then(extract_usage);

            // Try function-call detection first; if present, return tool-calls.
            if let Some(intents) = resp
                .output
                .as_ref()
                .and_then(extract_function_calls)
                .filter(|v| !v.is_empty())
            {
                let message = GenericMessage {
                    role: GenericRole::Assistant,
                    content: None,
                    name: None,
                    tool_calls: Some(intents),
                    tool_call_id: None,
                };
                return Ok(GenericChatCompletionResponse {
                    content: ResponseContent::ToolCalls(message),
                    usage,
                });
            }

            // Otherwise extract assistant text from Responses output (best-effort).
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
            req.input = Some(messages_json);
            req.temperature = params.temperature;
            req.response_format = params.response_format;
            req.stream = Some(true);
            // If tools are present, put them as a top-level field
            if let Some(tools) = params.tools {
                let tool_vals: Vec<ResponseTool> = tools.into_iter().map(tool_spec_from_generic).collect();
                req.tools = Some(tool_vals);
                req.tool_choice = Some(ResponseToolChoice::Literal(ResponseToolChoiceLiteral::Auto));
            }

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

            let single_tool_name: Option<String> = params.tools.as_ref().and_then(|ts| {
                if ts.len() == 1 { Some(ts[0].name.clone()) } else { None }
            });

            let mut req = ResponsesRequest::new(model);
            req.input = Some(messages_json);
            req.temperature = params.temperature;
            req.response_format = params.response_format;
            req.stream = Some(true);
            // If tools are present, put them as a top-level field
            if let Some(tools) = params.tools {
                let tool_vals: Vec<ResponseTool> = tools.into_iter().map(tool_spec_from_generic).collect();
                req.tools = Some(tool_vals);
                req.tool_choice = Some(ResponseToolChoice::Literal(ResponseToolChoiceLiteral::Auto));
            }

            let stream = client.response_stream(req);
            futures_util::pin_mut!(stream);

            // Track function-call states: (id, name, accumulated arguments)
            let mut fn_states: Vec<(Option<String>, Option<String>, String)> = Vec::new();

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
                    ResponseStreamEvent::FunctionCallArgumentsDelta { id, name, delta, .. } => {
                        // Find or create state by id or name
                        let mut created = false;
                        let idx = if let Some(ref sid) = id {
                            if let Some(i) = fn_states.iter().position(|s| s.0.as_ref() == Some(sid)) {
                                i
                            } else {
                                let i = fn_states.len();
                                fn_states.push((id.clone(), name.clone(), String::new()));
                                created = true;
                                i
                            }
                        } else if let Some(ref sname) = name {
                            if let Some(i) = fn_states.iter().position(|s| s.0.is_none() && s.1.as_ref() == Some(sname)) {
                                i
                            } else {
                                let i = fn_states.len();
                                fn_states.push((None, name.clone(), String::new()));
                                created = true;
                                i
                            }
                        } else {
                            let i = fn_states.len();
                            fn_states.push((None, None, String::new()));
                            created = true;
                            i
                        };

                        if created {
                            yield StreamEvent::ToolCallStart { index: idx, id: id.clone(), name: name.clone() };
                        }

                        if !delta.is_empty() {
                            fn_states[idx].2.push_str(&delta);
                            yield StreamEvent::ToolCallArgumentsDelta { index: idx, arguments_fragment: delta };
                        }
                    }
                    ResponseStreamEvent::FunctionCallArgumentsDone { id, name, arguments, .. } => {
                        // Finalize function call and emit a complete intent
                        let mut created = false;
                        let idx = if let Some(ref sid) = id {
                            if let Some(i) = fn_states.iter().position(|s| s.0.as_ref() == Some(sid)) {
                                i
                            } else {
                                let i = fn_states.len();
                                fn_states.push((id.clone(), name.clone(), String::new()));
                                created = true;
                                i
                            }
                        } else if let Some(ref sname) = name {
                            if let Some(i) = fn_states.iter().position(|s| s.0.is_none() && s.1.as_ref() == Some(sname)) {
                                i
                            } else {
                                let i = fn_states.len();
                                fn_states.push((None, name.clone(), String::new()));
                                created = true;
                                i
                            }
                        } else {
                            let i = fn_states.len();
                            fn_states.push((None, None, String::new()));
                            created = true;
                            i
                        };

                        if created {
                            yield StreamEvent::ToolCallStart { index: idx, id: id.clone(), name: name.clone() };
                        }

                        if id.is_some() { fn_states[idx].0 = id.clone(); }
                        if name.is_some() { fn_states[idx].1 = name.clone(); }

                        let id_for_intent = fn_states[idx].0.clone().unwrap_or_else(|| format!("fncall-{idx}"));
                        let name_for_intent = fn_states[idx]
                            .1
                            .clone()
                            .and_then(|n| if n == "function" { None } else { Some(n) })
                            .or_else(|| single_tool_name.clone())
                            .unwrap_or_else(|| "function".to_string());

                        let args_json: Value = serde_json::from_str(&arguments).unwrap_or(Value::String(arguments));

                        let intent = GenericFunctionCallIntent {
                            id: id_for_intent,
                            function: GenericFunctionCall { name: name_for_intent, arguments: args_json },
                        };

                        yield StreamEvent::ToolCallComplete { index: idx, intent };
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

        if matches!(m.role, MessageRole::Tool | MessageRole::Function) {
            // Inject tool result as a typed Responses input item:
            // { "type": "tool_result", "call_id": "<id>", "content": [{ "type": "output_text", "text": "<tool output>" }] }
            let mut tr = Map::new();
            if let Some(call_id) = &m.tool_call_id {
                tr.insert("call_id".to_string(), Value::String(call_id.clone()));
            }

            tr.insert(
                "type".to_string(),
                Value::String("function_call_output".into()),
            );
            let out_blocks = vec![json_obj(&[
                ("type", Value::String("input_text".into())),
                ("text", Value::String(text)),
            ])];
            tr.insert("output".to_string(), Value::Array(out_blocks));
            arr.push(Value::Object(tr));
            continue;
        }

        if matches!(m.role, MessageRole::Assistant) {
            if let Some(calls) = &m.tool_calls {
                // Emit function_call items so a subsequent function_call_output can reference a matching call_id.
                for tc in calls {
                    let mut fc = Map::new();
                    fc.insert("type".to_string(), Value::String("function_call".into()));
                    // Use "id" here; function_call_output will reference this via "call_id".
                    fc.insert("id".to_string(), Value::String(tc.id.clone()));
                    fc.insert("name".to_string(), Value::String(tc.function.name.clone()));
                    let arg_blocks = vec![json_obj(&[
                        ("type", Value::String("input_text".into())),
                        ("text", Value::String(tc.function.arguments.clone())),
                    ])];
                    fc.insert("arguments".to_string(), Value::Array(arg_blocks));
                    arr.push(Value::Object(fc));
                }
                // If the assistant message only contained tool-calls and no text, we're done.
                if text.is_empty() {
                    continue;
                }
            }
        }

        let block_type = match m.role {
            MessageRole::Assistant => "output_text",
            _ => "input_text",
        };

        let content_blocks = vec![json_obj(&[
            ("type", Value::String(block_type.into())),
            ("text", Value::String(text)),
        ])];

        // Build message object and include optional tool metadata when present.
        let mut msg_map = Map::new();
        msg_map.insert("role".to_string(), Value::String(role.into()));
        msg_map.insert("content".to_string(), Value::Array(content_blocks));

        // tool_call_id is not sent in Responses input

        // tool_calls are not sent in Responses input

        arr.push(Value::Object(msg_map));
    }
    Ok(Value::Array(arr))
}

/// Extract a concatenated text from the Responses `output` value.
/// Best-effort for typical shapes:
/// - Array of blocks: [{ "type": "output_text", "text": "..." }, ...]
/// - Or direct object with "text".
fn extract_output_text(output: &Value) -> Option<String> {
    fn collect_message_content_text(obj: &Map<String, Value>, buf: &mut String) {
        if let Some(content) = obj.get("content").and_then(|v| v.as_array()) {
            for part in content {
                if let Some(p) = part.as_object() {
                    let typ = p.get("type").and_then(|v| v.as_str());
                    if typ == Some("output_text") || typ == Some("summary_text") {
                        if let Some(text) = p.get("text").and_then(|v| v.as_str()) {
                            buf.push_str(text);
                        }
                    }
                }
            }
        }
    }

    match output {
        Value::Array(items) => {
            let mut buf = String::new();
            for it in items {
                if let Some(obj) = it.as_object() {
                    match obj.get("type").and_then(|v| v.as_str()) {
                        Some("message") => {
                            collect_message_content_text(obj, &mut buf);
                        }
                        Some("output_text") => {
                            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                                buf.push_str(text);
                            }
                        }
                        _ => {
                            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                                buf.push_str(text);
                            }
                        }
                    }
                }
            }
            if buf.is_empty() { None } else { Some(buf) }
        }
        Value::Object(obj) => {
            if obj.get("type").and_then(|v| v.as_str()) == Some("message") {
                let mut buf = String::new();
                collect_message_content_text(obj, &mut buf);
                if buf.is_empty() { None } else { Some(buf) }
            } else if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
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

fn tool_spec_from_generic(spec: GenericFunctionSpec) -> ResponseTool {
    ResponseTool::Function {
        name: spec.name,
        description: Some(spec.description),
        parameters: spec.parameters,
        strict: Some(true),
    }
}

fn extract_function_calls(output: &Value) -> Option<Vec<GenericFunctionCallIntent>> {
    fn parse_one(obj: &Map<String, Value>) -> Option<GenericFunctionCallIntent> {
        let typ = obj.get("type").and_then(|v| v.as_str());
        if typ != Some("function_call") && typ != Some("tool_call") {
            return None;
        }

        let id = obj
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "function_call".to_string());

        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                obj.get("function")
                    .and_then(|v| v.get("name").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "function".to_string());

        let args_val = if let Some(s) = obj.get("arguments").and_then(|v| v.as_str()) {
            serde_json::from_str::<Value>(s).unwrap_or(Value::String(s.to_string()))
        } else if let Some(func) = obj.get("function").and_then(|v| v.as_object()) {
            if let Some(s) = func.get("arguments").and_then(|v| v.as_str()) {
                serde_json::from_str::<Value>(s).unwrap_or(Value::String(s.to_string()))
            } else {
                Value::Null
            }
        } else {
            Value::Null
        };

        Some(GenericFunctionCallIntent {
            id,
            function: GenericFunctionCall {
                name,
                arguments: args_val,
            },
        })
    }

    let mut out = Vec::new();
    match output {
        Value::Array(items) => {
            for it in items {
                if let Some(obj) = it.as_object() {
                    if let Some(intent) = parse_one(obj) {
                        out.push(intent);
                        continue;
                    }

                    if obj.get("type").and_then(|v| v.as_str()) == Some("message") {
                        if let Some(tool_calls) = obj.get("tool_calls").and_then(|v| v.as_array()) {
                            for tc in tool_calls {
                                if let Some(tc_obj) = tc.as_object() {
                                    let id = tc_obj
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let name = tc_obj
                                        .get("function")
                                        .and_then(|v| v.get("name").and_then(|v| v.as_str()))
                                        .unwrap_or("function")
                                        .to_string();
                                    let args = tc_obj
                                        .get("function")
                                        .and_then(|v| v.get("arguments"))
                                        .and_then(|v| v.as_str())
                                        .map(|s| {
                                            serde_json::from_str::<Value>(s)
                                                .unwrap_or(Value::String(s.to_string()))
                                        })
                                        .unwrap_or(Value::Null);

                                    out.push(GenericFunctionCallIntent {
                                        id: if id.is_empty() {
                                            format!("fncall-{}", out.len())
                                        } else {
                                            id
                                        },
                                        function: GenericFunctionCall {
                                            name,
                                            arguments: args,
                                        },
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        Value::Object(obj) => {
            if let Some(intent) = parse_one(obj) {
                out.push(intent);
            }
        }
        _ => {}
    }

    if out.is_empty() { None } else { Some(out) }
}

fn json_obj(entries: &[(&str, Value)]) -> Value {
    let mut map = Map::new();
    for (k, v) in entries {
        map.insert((*k).to_string(), v.clone());
    }
    Value::Object(map)
}
