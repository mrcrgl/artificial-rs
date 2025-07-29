use artificial_core::generic::{GenericFunctionCall, GenericFunctionCallIntent};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolCallFunction,
}

impl Into<GenericFunctionCallIntent> for ToolCall {
    fn into(self) -> GenericFunctionCallIntent {
        GenericFunctionCallIntent {
            id: self.id,
            function: self.function.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
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
