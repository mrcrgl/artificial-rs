mod adapter;
mod model_map;

mod provider_impl_prompt;
mod provider_impl_responses;
pub use adapter::{OpenAiAdapter, OpenAiAdapterBuilder};
mod api_v1;
mod client;
pub mod error;
