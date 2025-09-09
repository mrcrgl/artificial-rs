use artificial_core::generic::{GenericFunctionCall, GenericFunctionCallIntent};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolCallFunction,
    pub r#type: ToolType,
}

impl Into<GenericFunctionCallIntent> for ToolCall {
    fn into(self) -> GenericFunctionCallIntent {
        GenericFunctionCallIntent {
            id: self.id,
            function: self.function.into(),
        }
    }
}

impl From<GenericFunctionCallIntent> for ToolCall {
    fn from(value: GenericFunctionCallIntent) -> Self {
        Self {
            id: value.id,
            function: value.function.into(),
            r#type: ToolType::Function,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl Into<GenericFunctionCall> for ToolCallFunction {
    fn into(self) -> GenericFunctionCall {
        GenericFunctionCall {
            name: self.name,
            arguments: self.arguments,
        }
    }
}

impl From<GenericFunctionCall> for ToolCallFunction {
    fn from(value: GenericFunctionCall) -> Self {
        Self {
            name: value.name,
            arguments: value.arguments,
        }
    }
}
