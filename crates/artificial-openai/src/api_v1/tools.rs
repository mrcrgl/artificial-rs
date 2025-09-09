use artificial_core::generic::{GenericFunctionCall, GenericFunctionCallIntent};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolCallFunction,
    pub r#type: ToolType,
}

impl From<ToolCall> for GenericFunctionCallIntent {
    fn from(val: ToolCall) -> Self {
        GenericFunctionCallIntent {
            id: val.id,
            function: val.function.into(),
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

impl From<ToolCallFunction> for GenericFunctionCall {
    fn from(val: ToolCallFunction) -> Self {
        GenericFunctionCall {
            name: val.name,
            arguments: val.arguments,
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
