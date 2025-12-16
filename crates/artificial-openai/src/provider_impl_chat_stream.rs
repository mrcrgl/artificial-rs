use std::pin::Pin;

use crate::OpenAiAdapter;
use crate::api_v1::ChatCompletionRequest;
use crate::api_v1::FinishReason;
use artificial_core::error::{ArtificialError, Result};
use artificial_core::generic::{GenericFunctionCall, GenericFunctionCallIntent, StreamEvent};
use artificial_core::provider::StreamingEventsProvider;
use artificial_core::provider::{ChatCompleteParameters, StreamingChatProvider};
use futures_core::stream::Stream;
use std::collections::HashMap;

impl StreamingChatProvider for OpenAiAdapter {
    type Delta<'s>
        = Pin<Box<dyn Stream<Item = Result<String>> + Send + 's>>
    where
        Self: 's;

    fn chat_complete_stream<'s, M>(&'s self, params: ChatCompleteParameters<M>) -> Self::Delta<'s>
    where
        M: Into<Self::Message> + Clone + Send + Sync + 's,
    {
        let client = self.client.clone();

        Box::pin(async_stream::try_stream! {
        use futures_util::StreamExt;

        let request: ChatCompletionRequest = params.try_into()?;


            let stream = client.chat_completion_stream(request);
            futures_util::pin_mut!(stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(ArtificialError::from)?;
                for choice in chunk.choices {
                    if let Some(text) = choice.delta.content {
                        yield text;
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

    fn chat_complete_events_stream<'s, M>(
        &'s self,
        params: ChatCompleteParameters<M>,
    ) -> Self::EventStream<'s>
    where
        M: Into<Self::Message> + Clone + Send + Sync + 's,
    {
        let client = self.client.clone();

        Box::pin(async_stream::try_stream! {
            use futures_util::StreamExt;

            let request: ChatCompletionRequest = params.try_into()?;

            // Track tool-call argument fragments and first-seen id/name per tool index.
            let mut tool_args: HashMap<usize, String> = HashMap::new();
            let mut tool_seen: HashMap<usize, (Option<String>, Option<String>)> = HashMap::new();

            let stream = client.chat_completion_stream(request);
            futures_util::pin_mut!(stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(ArtificialError::from)?;

                for choice in chunk.choices {
                    // Process only the first choice to match current non-streaming behavior.
                    if choice.index != 0 { continue; }

                    // Text deltas
                    if let Some(delta) = choice.delta.content
                        && !delta.is_empty() {
                            yield StreamEvent::TextDelta(delta);
                        }

                    // Tool-call deltas
                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for tc in tool_calls {
                            let entry = tool_seen.entry(tc.index).or_insert((None, None));

                            if let Some(id) = tc.id.clone() {
                                if entry.0.is_none() {
                                    entry.0 = Some(id.clone());
                                    yield StreamEvent::ToolCallStart {
                                        index: tc.index,
                                        id: Some(id),
                                        name: entry.1.clone(),
                                    };
                                } else {
                                    entry.0 = Some(id);
                                }
                            }

                            if let Some(func) = tc.function {
                                if let Some(name) = func.name {
                                    if entry.1.is_none() {
                                        entry.1 = Some(name.clone());
                                        yield StreamEvent::ToolCallStart {
                                            index: tc.index,
                                            id: entry.0.clone(),
                                            name: Some(name),
                                        };
                                    } else {
                                        entry.1 = Some(name);
                                    }
                                }

                                if let Some(arguments) = func.arguments {
                                    let buf = tool_args.entry(tc.index).or_default();
                                    buf.push_str(&arguments);
                                    if !arguments.is_empty() {
                                        yield StreamEvent::ToolCallArgumentsDelta {
                                            index: tc.index,
                                            arguments_fragment: arguments,
                                        };
                                    }
                                }
                            }
                        }
                    }

                    // Finish conditions
                    if let Some(reason) = choice.finish_reason {
                        match reason {
                            FinishReason::ToolCalls => {
                                // Finalize tool calls by parsing accumulated argument buffers.
                                for (index, buf) in tool_args.iter() {
                                    let (id_opt, name_opt) = tool_seen
                                        .get(index)
                                        .cloned()
                                        .unwrap_or((None, None));

                                    let name = name_opt.unwrap_or_else(|| "tool".to_string());
                                    let args_json: serde_json::Value = serde_json::from_str(buf)
                                        .map_err(|e| ArtificialError::Invalid(format!("invalid tool arguments JSON: {e}")))?;

                                    let intent = GenericFunctionCallIntent {
                                        id: id_opt.unwrap_or_else(|| format!("toolcall-{index}")),
                                        function: GenericFunctionCall { name, arguments: args_json },
                                    };

                                    yield StreamEvent::ToolCallComplete { index: *index, intent };
                                }

                                yield StreamEvent::MessageEnd;
                                return;
                            }
                            FinishReason::Stop | FinishReason::Length | FinishReason::ContentFilter => {
                                yield StreamEvent::MessageEnd;
                                return;
                            }
                        }
                    }
                }
            }
        })
    }
}
