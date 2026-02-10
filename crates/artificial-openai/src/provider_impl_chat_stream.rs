//! Streaming over chat/completions has been removed.
//!
//! This crate has fully migrated to the OpenAI Responses API (/v1/responses).
//!
//! The streaming implementations now live in `provider_impl_responses.rs` and
//! cover both:
//! - text-only streaming via `StreamingChatProvider::chat_complete_stream`
//! - rich, provider-agnostic streaming events (text + tool-calls) via
//!   `StreamingEventsProvider::chat_complete_events_stream`
//!
//! If you were relying on the legacy chat/completions streaming code that used
//! the `/v1/chat/completions` endpoint and incremental deltas, switch to the
//! Responses-based flows. The new implementation uses semantic streaming
//! events from the Responses API and maps them to the generic/core types.
//!
//! Feature parity:
//! - Incremental text output          -> `StreamEvent::TextDelta`
//! - End-of-message notification      -> `StreamEvent::MessageEnd`
//! - Tool-call streaming (arguments)  -> `StreamEvent::{ToolCallStart, ToolCallArgumentsDelta, ToolCallComplete}`
//! - Optional token usage at the end  -> `StreamEvent::Usage`
//!
//! See `provider_impl_responses.rs` for the concrete implementation.

#[allow(dead_code)]
fn __assert_adapter_traits_streaming(adapter: &crate::OpenAiAdapter)
where
    crate::OpenAiAdapter: artificial_core::provider::ChatCompletionProvider
        + artificial_core::provider::StreamingChatProvider
        + artificial_core::generic::StreamingEventsProvider,
{
    let _ = adapter;
}
