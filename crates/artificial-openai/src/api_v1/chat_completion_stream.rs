/*!
This file has been intentionally left empty.

The legacy streaming chat/completions types previously defined here have been removed.
The crate now uses the OpenAI Responses API (/v1/responses) exclusively.

For streaming:
- Use `StreamingChatProvider::chat_complete_stream` for incremental text (backed by Responses).
- Use `StreamingEventsProvider::chat_complete_events_stream` for structured streaming events
  (text deltas, tool-call arguments, etc.).

See `provider_impl_responses.rs` for the concrete implementations.
*/
