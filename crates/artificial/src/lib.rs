//! # `artificial` – The umbrella crate
//!
//! This crate is a *one-stop import* that glues together the three
//! building-block crates in the workspace
//!
//! | Crate                    | What it provides                                                                 |
//! |--------------------------|----------------------------------------------------------------------------------|
//! | **`artificial-core`**    | Provider-agnostic traits (`Backend`, `PromptTemplate`), generic client, errors   |
//! | **`artificial-prompt`**  | Ergonomic helpers for building and chaining prompt fragments                     |
//! | **`artificial-types`**   | Reusable fragments, helper structs (`ThinkResult`, `CurrentDateFragment`, …)     |
//! | **`artificial-openai`**  | Thin HTTP client that implements `Backend` for the OpenAI *v1* API *(optional)*  |
//!
//! By default the crate only re-exports **core**, **prompt** and **types** so
//! downstream users can stay 100 % provider-agnostic.  Enabling the `openai`
//! Cargo feature additionally re-exports the adapter crate so a single
//! dependency line is enough to access the whole stack:
//!
//! ```toml
//! [dependencies]
//! artificial = { version = "0.1", features = ["openai"] }
//! ```
//!
//! ## Design philosophy
//!
//! * **Opt-in providers** – No unwanted dependencies: enabling `openai` pulls
//!   in `reqwest`, TLS, etc., otherwise your binary stays lean.
//! * **No procedural macros** – Everything is powered by ordinary traits and
//!   `impl`s so you can understand and extend the code without magic.
//! * **Type-safe JSON** – Responses are validated with
//!   [`schemars`](https://docs.rs/schemars) *before* they reach your code.
//!
//! ## Quick example
//!
//! ```rust,no_run
//! use artificial::{
//!     ArtificialClient,
//!     generic::{GenericMessage, GenericRole},
//!     model::{Model, OpenAiModel},
//!     template::{IntoPrompt, PromptTemplate},
//!     provider::PromptExecutionProvider,
//! };
//!
//! // Define the answer shape
//! #[derive(serde::Deserialize, schemars::JsonSchema)]
//! struct Hello { greeting: String }
//!
//! // Implement a tiny prompt
//! struct AskHello;
//! impl IntoPrompt for AskHello {
//!     type Message = GenericMessage;
//!     fn into_prompt(self) -> Vec<Self::Message> {
//!         vec![GenericMessage::new("Say hello!".into(), GenericRole::User)]
//!     }
//! }
//! impl PromptTemplate for AskHello {
//!     type Output = Hello;
//!     const MODEL: Model = Model::OpenAi(OpenAiModel::Gpt4oMini);
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let backend = artificial::openai::OpenAiAdapterBuilder::new_from_env().build()?;
//!     let client  = ArtificialClient::new(backend);
//!     let answer  = client.prompt_execute(AskHello).await?;
//!     println!("{}", answer.greeting);
//!     Ok(())
//! }
//! ```
//!
//! ## Crate contents
//!
//! The `pub use` statements below simply forward the public API of the
//! individual crates so users can write `artificial::ArtificialClient` instead
//! of juggling four separate dependencies.
//!
//! ---
//! _Happy prompting & may your JSON always validate!_
#![doc(html_root_url = "https://docs.rs/artificial/latest")]

pub use artificial_core::*;
pub use artificial_prompt as prompt;
pub use artificial_types as types;

#[cfg(feature = "openai")]
pub use artificial_openai as openai;
