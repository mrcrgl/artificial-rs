#![allow(clippy::module_name_repetitions)]
//! Example: *Extracting memories from a multi-turn chat log (Star-Wars edition)*
//!
//! This showcases how to
//!
//! 1. Compose several prompt *fragments* (small, reusable pieces that turn
//!    into `GenericMessage`s) into a bigger [`PromptTemplate`].
//! 2. Keep everything strongly-typed – the response is parsed into
//!    [`ThinkResult<MemoryExtraction>`].
//!
//! The code is intentionally *stand-alone* so that you can run it from the crate
//! without touching any other files.

use artificial::{
    ArtificialClient,
    generic::{GenericMessage, GenericRole},
    model::{Model, OpenAiModel},
    template::{IntoPrompt, PromptTemplate},
};
use artificial_openai::OpenAiAdapterBuilder;
use artificial_prompt::{builder::PromptBuilder, chain::PromptChain};
use artificial_types::{
    fragments::{CurrentDateFragment, StaticFragment},
    outputs::result::ThinkResult,
};
use schemars::{
    JsonSchema, SchemaGenerator,
    schema::{InstanceType, Metadata, SchemaObject, SingleOrVec},
};
use serde::{Deserialize, Serialize};

/// ---------------------------------------------------------------------------
/// ❶ Domain stubs – nice and small so we can focus on the prompting logic
/// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Message {
    pub from: String,
    pub text: String,
}

/// ---------------------------------------------------------------------------
/// ❷ Prompt snippets (markdown in `examples/data/role/**.md`)
/// ---------------------------------------------------------------------------

/// “Base system” instructions that *any* prompt in this workspace usually adds
/// (e.g. stylistic guidelines, response hygiene, …).
const BASE_SYSTEM_ROLE: &str = include_str!("data/role/base_system.md");

/// Extra rules for the large language model on how to behave when *extracting
/// memories*.
const MEMORY_ARCHITECT_ROLE: &str = include_str!("data/role/memory_architect.md");

/// ---------------------------------------------------------------------------
/// ❸ A high-level prompt template: *CaptureMemory*
///
///    It is mostly a container that wires together the individual fragments
///    (system base, date, team profile, chat history, …).
/// ---------------------------------------------------------------------------

struct CaptureMemory<'a> {
    system_base_fragment: StaticFragment<'a>,
    memory_architect_role_fragment: StaticFragment<'a>,
    agent_fragment: AgentProfileFragment<'a>,
    team_fragment: TeamProfileFragment,
    history_fragment: MessageHistoryFragment,
}

impl<'a> CaptureMemory<'a> {
    /// Constructor with a slightly too many parameters – perfectly fine for an
    /// **example**.
    #[allow(clippy::too_many_arguments)]
    pub fn new(member: MemberProfile, history: Vec<Message>, team_profile: TeamProfile) -> Self {
        Self {
            system_base_fragment: BASE_SYSTEM_ROLE.into(),
            memory_architect_role_fragment: MEMORY_ARCHITECT_ROLE.into(),
            agent_fragment: AgentProfileFragment::new(member, &team_profile.team_name),
            team_fragment: TeamProfileFragment::new(team_profile),
            history_fragment: MessageHistoryFragment::new(history),
        }
    }
}

/// Insert the fragments into a [`PromptChain`] in a well-defined order.
impl<'a> IntoPrompt for CaptureMemory<'a> {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        // A final human instruction so the LLM *really* knows what we expect.
        let final_instruction = StaticFragment::new(
            "Extract any important memory worth remembering from this conversation.",
            GenericRole::User,
        );

        PromptChain::new()
            .with(self.system_base_fragment)
            .with(CurrentDateFragment::new()) // Helps the model reason about dates.
            .with(self.agent_fragment)
            .with(self.team_fragment)
            .with(self.memory_architect_role_fragment)
            .with(self.history_fragment)
            .with(final_instruction)
            .build()
    }
}

/// Wire template to a concrete model (here: GPT-4o Mini) **and** declare the
/// parsed output type.
impl<'a> PromptTemplate for CaptureMemory<'a> {
    type Output = ThinkResult<MemoryExtraction>;
    const MODEL: Model = Model::OpenAi(OpenAiModel::Gpt4oMini);
}

/// ---------------------------------------------------------------------------
/// ❹ The famous `main` function – spin up the backend, build the prompt, run it
/// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // -- Rebel Alliance team profiles ------------------------------------------------
    let member_r2d2 = MemberProfile {
        name: "R2-D2",
        biography: "Resourceful astromech droid and hobby scream-beeper.",
    };
    let member_luke = MemberProfile {
        name: "Luke Skywalker",
        biography: "Moisture-farmer-turned-Jedi. Good at bullseyeing womp rats.",
    };
    let member_chewie = MemberProfile {
        name: "Chewbacca",
        biography: "Walking carpet with a heart of gold. Fluent in Shyriiwook.",
    };

    let team_profile = TeamProfile {
        team_name: "Rebel Alliance",
        members: vec![member_r2d2.clone(), member_luke, member_chewie],
    };

    // -- Dummy chat history (radio chatter on a mission) ----------------------------
    let history = vec![
        Message {
            from: "Chewbacca".into(),
            text: "⏩ *static*  Rrrgh! (Translation: “Is everyone strapped in? I’m about \
                   to punch it.”)"
                .into(),
        },
        Message {
            from: "Luke Skywalker".into(),
            text: "Copy that, Chewie. R2, make sure the deflector shields are up. We \
                   don't want to become space confetti."
                .into(),
        },
        Message {
            from: "R2-D2".into(),
            text: "Beep-boop-bweep! (Translation: “Shields at 120%. I overclocked them. \
                   Please don't tell the warranty droid.”)"
                .into(),
        },
        Message {
            from: "Chewbacca".into(),
            text: "Raaawrr! (Translation: “Good, because I see three TIE fighters \
                   who think we're today's buffet special.”)"
                .into(),
        },
        Message {
            from: "Luke Skywalker".into(),
            text: "Stay on target, team. Remember: evasive roll first, philosophical \
                   quotes later."
                .into(),
        },
    ];

    // -- Build backend + client ------------------------------------------------------
    // Make sure you have `OPENAI_API_KEY` in your environment when running this.
    let backend = OpenAiAdapterBuilder::new_from_env().build()?;
    let client = ArtificialClient::new(backend);

    // -- Execute the prompt ----------------------------------------------------------
    let prompt = CaptureMemory::new(member_r2d2, history, team_profile);
    let result = client.chat_complete(prompt).await?;

    println!("🤖 LLM remembered:\n{:#?}", result);

    Ok(())
}

/// ===========================================================================
/// ❺ Reusable *fragment* implementations
/// ===========================================================================

/// ---- TeamProfileFragment --------------------------------------------------

pub struct TeamProfileFragment {
    team_spec: TeamProfile,
}

impl TeamProfileFragment {
    fn new(team_spec: TeamProfile) -> Self {
        Self { team_spec }
    }
}

impl IntoPrompt for TeamProfileFragment {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        let profile = serde_yaml::to_string(&self.team_spec)
            .unwrap_or_else(|_| "<serialization error>".into());

        let builder = PromptBuilder::new()
            .add_section_h2("Team Profile")
            .add_line("You are part of the following strike team:")
            .add_text_yaml(profile);

        vec![GenericMessage::new(builder.finalize(), GenericRole::System)]
    }
}

/// ---- AgentProfileFragment -------------------------------------------------

pub struct AgentProfileFragment<'a> {
    member: MemberProfile,
    team_name: &'a str,
}

impl<'a> AgentProfileFragment<'a> {
    fn new(member: MemberProfile, team_name: &'a str) -> Self {
        Self { member, team_name }
    }
}

impl IntoPrompt for AgentProfileFragment<'_> {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        let profile =
            serde_yaml::to_string(&self.member).unwrap_or_else(|_| "<serialization error>".into());

        let builder = PromptBuilder::new()
            .add_section_h2("Your Profile")
            .add_key_value("Name", self.member.name)
            .add_key_value("Biography", self.member.biography)
            .add_key_value("Affiliation", self.team_name)
            .add_text_yaml(profile);

        vec![GenericMessage::new(builder.finalize(), GenericRole::System)]
    }
}

/// ---- MessageHistoryFragment ----------------------------------------------

pub struct MessageHistoryFragment {
    history: Vec<Message>,
}

impl MessageHistoryFragment {
    fn new(history: Vec<Message>) -> Self {
        Self { history }
    }
}

impl IntoPrompt for MessageHistoryFragment {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        if self.history.is_empty() {
            return vec![];
        }

        let mut messages = Vec::with_capacity(self.history.len());

        for message in self.history {
            messages.extend(MessageFragment::new(&message.from, &message.text).into_prompt());
        }

        messages
    }
}

/// ---- MessageFragment ------------------------------------------------------

pub struct MessageFragment<'a> {
    name: &'a str,
    message: &'a str,
}

impl<'a> MessageFragment<'a> {
    pub fn new(name: &'a str, message: &'a str) -> Self {
        Self { name, message }
    }
}

impl IntoPrompt for MessageFragment<'_> {
    type Message = GenericMessage;

    fn into_prompt(self) -> Vec<Self::Message> {
        let builder = PromptBuilder::new()
            .add_section_h2(format!("Message from {}", self.name))
            .add_key_value("Body", "")
            .add_text_markdown(self.message)
            .add_blank_line();

        vec![GenericMessage::new(builder.finalize(), GenericRole::System)]
    }
}

/// ===========================================================================
/// ❻ Small data structs that we serialise into YAML inside the prompt
/// ===========================================================================

#[derive(Serialize)]
struct TeamProfile {
    team_name: &'static str,
    members: Vec<MemberProfile>,
}

#[derive(Serialize, Clone, Copy)]
struct MemberProfile {
    pub name: &'static str,
    pub biography: &'static str,
}

/// ===========================================================================
/// ❼ Output types – strictly validated with `schemars` to avoid accidents
/// ===========================================================================

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryExtraction {
    /// Summaries of memories that should be written to long-term store
    pub items: Vec<MemoryExtractionItem>,
}

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryExtractionItem {
    /// Short description of the remembered fact
    pub summary: String,
    /// Origin (agent name, message id, …). Optional but very helpful.
    #[schemars(required)]
    pub origin: Option<String>,
    /// Relevance score between 0 and 1
    pub relevance_score: f32,
    /// Category of memory
    pub classification: MemoryClassification,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryClassification {
    /// Task-specific insight or observation
    #[default]
    Reflective,
    /// Long-lived instruction the agent should obey
    Directive,
    /// General principle or strategy
    Strategic,
}

impl JsonSchema for MemoryClassification {
    fn schema_name() -> String {
        "MemoryClassification".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> schemars::schema::Schema {
        schemars::schema::Schema::Object(SchemaObject {
            metadata: Some(Box::new(Metadata {
                description: Some(
                    "Classification of the memory information. \
                     Possible values: reflective, directive, strategic."
                        .into(),
                ),
                ..Default::default()
            })),
            instance_type: Some(SingleOrVec::Single(Box::new(InstanceType::String))),
            enum_values: Some(vec![
                serde_json::Value::String("reflective".into()),
                serde_json::Value::String("directive".into()),
                serde_json::Value::String("strategic".into()),
            ]),
            ..Default::default()
        })
    }
}
