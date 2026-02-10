use artificial_core::generic::{GenericMessage, GenericRole};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

use super::tools::ToolCall;

/// Minimal role set kept for internal conversion and Responses API shaping.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    System,
    Assistant,
    Function,
    Tool,
}

/// Minimal text content wrapper used by internal message conversions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    Text(String),
}

impl Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match *self {
            Content::Text(ref text) => {
                if text.is_empty() {
                    serializer.serialize_none()
                } else {
                    serializer.serialize_str(text)
                }
            }
        }
    }
}

impl<'de> Deserialize<'de> for Content {
    fn deserialize<D>(deserializer: D) -> Result<Content, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ContentVisitor;

        impl<'de> Visitor<'de> for ContentVisitor {
            type Value = Content;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a valid content type")
            }

            fn visit_str<E>(self, value: &str) -> Result<Content, E>
            where
                E: de::Error,
            {
                Ok(Content::Text(value.to_string()))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Content::Text(String::new()))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Content::Text(String::new()))
            }
        }

        deserializer.deserialize_any(ContentVisitor)
    }
}

/// Provider-specific message shape used only for internal conversion to Responses API
/// "messages" JSON. Avoids exposing deprecated chat/completions wire types.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionMessage {
    pub role: MessageRole,
    pub content: Option<Content>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

impl From<GenericRole> for MessageRole {
    fn from(value: GenericRole) -> Self {
        match value {
            GenericRole::System => MessageRole::System,
            GenericRole::Assistant => MessageRole::Assistant,
            GenericRole::User => MessageRole::User,
            GenericRole::Tool => MessageRole::Tool,
        }
    }
}

impl From<MessageRole> for GenericRole {
    fn from(val: MessageRole) -> Self {
        match val {
            MessageRole::User => GenericRole::User,
            MessageRole::System => GenericRole::System,
            MessageRole::Assistant => GenericRole::Assistant,
            MessageRole::Function => GenericRole::Tool,
            MessageRole::Tool => GenericRole::Tool,
        }
    }
}

impl From<GenericMessage> for ChatCompletionMessage {
    fn from(value: GenericMessage) -> Self {
        Self {
            role: value.role.into(),
            content: value.content.map(Content::Text),
            name: value.name,
            tool_calls: value
                .tool_calls
                .map(|v| v.into_iter().map(Into::into).collect()),
            tool_call_id: value.tool_call_id,
        }
    }
}
