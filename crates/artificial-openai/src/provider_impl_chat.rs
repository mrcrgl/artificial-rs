#![allow(dead_code)]
//! Legacy Chat Completions shim
//!
//! This crate has migrated fully to the OpenAI Responses API (/v1/responses).
//! The actual provider implementations now live in `provider_impl_responses.rs`:
//! - `ChatCompletionProvider` (non-streaming text)
//! - `StreamingChatProvider` (text deltas)
//! - `StreamingEventsProvider` (structured streaming: text + tool-calls)
//!
//! This module remains as a tiny compatibility shim so that downstream code or
//! grep targets referring to `provider_impl_chat` can discover the new setup.
//! It contains a compile-time assertion ensuring the adapter implements the
//! intended traits, without pulling in any of the legacy chat/completions types.

/// Compile-time assertion that the adapter implements the Responses-based traits.
///
/// This function is never called; it only enforces trait bounds during compilation.
pub(crate) fn __assert_adapter_traits(adapter: &crate::OpenAiAdapter)
where
    crate::OpenAiAdapter: artificial_core::provider::ChatCompletionProvider
        + artificial_core::provider::StreamingChatProvider
        + artificial_core::generic::StreamingEventsProvider,
{
    let _ = adapter;
}
