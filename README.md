# Artificial ‒ typed, composable prompt engineering in Rust

Artificial is a **batteries-included toy framework** that demonstrates how to build
strongly-typed, provider-agnostic prompt pipelines in Rust.

The code base is intentionally small—less than 3 k lines spread over multiple
crates—yet it already shows how to:

* Compose reusable prompt *fragments* into high-level templates.
* Target multiple model providers via pluggable back-ends  
  (currently OpenAI, adding others is trivial).
* Describe the LLM’s **JSON response format in Rust** and have it checked
  at compile time with [`schemars`](https://docs.rs/schemars).
* Bubble up provider errors through a single, unified error type.
* Keep all heavy dependencies optional behind feature flags.

If you are curious how the new crop of “AI SDKs” work under the hood—or you
need a lean starting point for your own experiments—this repo is for you.

---

## Table of contents

1. [Crate layout](#crate-layout)  
2. [Installation](#installation)  
3. [Quick start](#quick-start)  
4. [Library tour](#library-tour)  
5. [Design goals](#design-goals)  
6. [FAQ](#faq)  
7. [License](#license)

---

## Crate layout

| Crate                        | Purpose                                                            |
|------------------------------|--------------------------------------------------------------------|
| **`artificial-core`**        | Provider-agnostic traits (`Backend`, `PromptTemplate`), client, error types |
| **`artificial-prompt`**      | String-building helpers (`PromptBuilder`, `PromptChain`)           |
| **`artificial-types`**       | Shared fragments (`CurrentDateFragment`, `StaticFragment`) and output helpers |
| **`artificial-openai`**      | Thin wrapper around *OpenAI /v1* with JSON-Schema function calling |
| **`artificial`**             | Glue crate that re-exports everything above for convenience        |

Each crate lives under `crates/*` and can be used independently, but most
people will depend on the umbrella crate:

```toml
[dependencies]
artificial = { path = "crates/artificial", features = ["openai"] }
```

---

## Installation

Artificial is published as a **workspace example**, so you typically work
against the Git repository directly:

```bash
git clone https://github.com/mrcrgl/artificial-rs.git
cd artificial-rs
cargo run -p artificial --example openai_hello_world
```

Requirements:

* **Rust 1.77** or newer (edition 2024)
* An OpenAI API key exported as `OPENAI_API_KEY`
* Internet access for the example back-end

---

## Quick start

Below is a minimal “Hello, JSON” example taken from
[`examples/openai_hello_world.rs`](crates/artificial/examples/openai_hello_world.rs):

```rust
use artificial::{
    ArtificialClient,
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    template::{IntoPrompt, PromptTemplate},
};
use artificial_openai::OpenAiAdapterBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct HelloResponse { greeting: String }

struct HelloPrompt;

impl IntoPrompt for HelloPrompt {
    type Message = GenericMessage;
    fn into_prompt(self) -> Vec<Self::Message> {
        vec![
            GenericMessage::new("You are R2-D2.".into(), GenericRole::System),
            GenericMessage::new("Say hello!".into(),  GenericRole::User),
        ]
    }
}

impl PromptTemplate for HelloPrompt {
    type Output = HelloResponse;
    const MODEL: Model = Model::OpenAi(OpenAiModel::Gpt4oMini);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let backend  = OpenAiAdapterBuilder::new_from_env().build()?;
    let client   = ArtificialClient::new(backend);
    let response = client.chat_complete(HelloPrompt).await?;
    println!("The droid says: {:?}", response);
    Ok(())
}
```

Run it:

```bash
cargo run -p artificial --example openai_hello_world
```

The program sends a request with an **inline JSON-Schema** and prints the
deserialised reply.

---

## Library tour

### Prompt fragments  
`artificial-types::fragments` contains small building blocks that turn into
`GenericMessage`s—think “Current date”, “Static system instruction”, “Last user
message”. You can combine them via `PromptChain`:

```rust
PromptChain::new()
    .with(CurrentDateFragment::new())
    .with(StaticFragment::new("You are a helpful bot.", GenericRole::System))
    .with(last_user_message)
    .build();
```

### Strongly-typed outputs  
Define a `struct` (must implement `JsonSchema` + `Deserialize`) and declare it
as `PromptTemplate::Output`. The OpenAI back-end automatically injects the
schema as `response_format = json_schema`.

### Provider back-ends  
A back-end only has to implement the single-method trait

```rust
trait Backend {
    type Message;
    fn chat_complete<P>(&self, prompt: P) -> Pin<Box<dyn Future<Output = Result<P::Output>> + Send>>
}
```

Because the `PromptTemplate` carries the desired model, the back-end can map it
to the provider’s naming scheme (`gpt-4o-mini`, `gpt-4o`, …).

---

## Design goals

* **Minimal surface area** – the framework should fit in one screenful of
  source code, making it easy to fork or copy-paste.
* **Compile-time checks everywhere** – wrong model name? missing `OPENAI_API_KEY`?
  Incompatible JSON schema? You find out early.
* **Pluggable transports** – swap out `artificial-openai` for an
  Ollama/Anthropic adapter without touching user code.
* **No macros** – generic traits and `impl`s keep the magic transparent.

---

## FAQ

**Why another AI SDK?**  
Because many existing SDKs are either *too* heavyweight or hide the inner
workings behind procedural macros. Artificial aims to be the smallest possible
blueprint you can still use in production.

**Does it support streaming completions?**  
Not yet. The `Backend` trait is purposely tiny; adding another method for
streaming is straightforward.

**Is the OpenAI back-end production ready?**  
It handles basic JSON-Schema function calling, retries and streaming are
missing. Treat it as a reference implementation.

**How do I add Anthropic or Ollama?**  

1. Create `artificial-anthropic` (or similar) crate.  
2. Wrap the provider’s HTTP API in a small client struct.  
3. Implement `Backend` for the adapter.  
4. Done—`ArtificialClient` works instantly with the new provider.

---

## License

Licensed under **MIT**. See [`LICENSE`](LICENSE) for the full text.