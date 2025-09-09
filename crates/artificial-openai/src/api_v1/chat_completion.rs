use artificial_core::error::ArtificialError;
use artificial_core::generic::{GenericFunctionSpec, GenericMessage, GenericRole};
use artificial_core::provider::ChatCompleteParameters;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use std::fmt;

use crate::impl_builder_methods;
use crate::model_map::map_model;

use super::common;
use super::tools::ToolCall;

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

impl ChatCompletionRequest {
    pub fn new(model: String, messages: Vec<ChatCompletionMessage>) -> Self {
        Self {
            model,
            messages,
            temperature: None,
            top_p: None,
            n: None,
            response_format: None,
            stream: None,
            tools: None,
            tool_choice: None,
        }
    }
}

impl_builder_methods!(
    ChatCompletionRequest,
    temperature: f64,
    top_p: f64,
    n: i64,
    response_format: serde_json::Value,
    stream: bool,
    tools: Vec<ToolSpec>,
    tool_choice: ToolChoice
);

impl<M> TryFrom<ChatCompleteParameters<M>> for ChatCompletionRequest
where
    M: Into<ChatCompletionMessage>,
{
    type Error = ArtificialError;

    fn try_from(value: ChatCompleteParameters<M>) -> Result<Self, Self::Error> {
        Ok(Self {
            model: map_model(&value.model)
                .ok_or(ArtificialError::InvalidRequest(format!(
                    "backend does not support selected model: {:?}",
                    value.model
                )))?
                .into(),
            messages: value.messages.into_iter().map(Into::into).collect(),
            tools: value
                .tools
                .and_then(|tools| Some(tools.into_iter().map(Into::into).collect())),
            temperature: value.temperature,
            top_p: None,
            n: None,
            response_format: value.response_format,
            stream: None,
            tool_choice: None,
        })
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ToolSpec {
    pub function: ToolFunctionSpec,
    pub r#type: ToolType,
}

impl From<GenericFunctionSpec> for ToolSpec {
    fn from(value: GenericFunctionSpec) -> Self {
        ToolSpec {
            function: ToolFunctionSpec {
                name: value.name,
                description: value.description,
                parameters: value.parameters,
                strict: Some(true),
            },
            r#type: ToolType::Function,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ToolFunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Copy, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
}

#[derive(Debug, Deserialize, Serialize, Copy, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    System,
    Assistant,
    Function,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    Text(String),
}

impl serde::Serialize for Content {
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

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    Text,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionMessage {
    pub role: MessageRole,
    pub content: Option<Content>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionMessageForResponse {
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Into<GenericMessage> for ChatCompletionMessageForResponse {
    fn into(self) -> GenericMessage {
        GenericMessage {
            content: self.content,
            role: self.role.into(),
            tool_calls: self
                .tool_calls
                .map(|calls| calls.into_iter().map(Into::into).collect()),
            name: self.name,
            tool_call_id: self.tool_call_id,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: i64,
    pub message: ChatCompletionMessageForResponse,
    pub finish_reason: Option<FinishReason>,
    pub finish_details: Option<FinishDetails>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: Option<String>,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: common::Usage,
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub struct FinishDetails {
    pub r#type: FinishReason,
    pub stop: String,
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

impl Into<GenericRole> for MessageRole {
    fn into(self) -> GenericRole {
        match self {
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
            content: value.content.map(|v| Content::Text(v)),
            name: value.name,
            tool_calls: value
                .tool_calls
                .and_then(|v| Some(v.into_iter().map(Into::into).collect())),
            tool_call_id: value.tool_call_id,
        }
    }
}
