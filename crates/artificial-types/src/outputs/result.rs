use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, JsonSchema, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ThinkResult<T> {
    /// Status of the request.
    pub status: ThinkStatus,
    /// Reason about the status conclusion.
    pub reasoning: String,
    /// Confidence rating from 0.0 to 1.0.
    pub confidence: f32,
    /// Result data of the operation.
    #[schemars(required)]
    pub data: Option<T>,
}

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkStatus {
    Succeed,
    Error,
}
