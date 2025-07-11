mod adapter;
mod backend;
mod model_map;

pub use adapter::{OpenAiAdapter, OpenAiAdapterBuilder};
pub mod api_v1;
mod client;
pub mod error;
