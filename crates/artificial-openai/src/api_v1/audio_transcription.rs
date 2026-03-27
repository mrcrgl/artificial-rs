use std::collections::HashMap;

use artificial_core::provider::{TranscriptionResult, TranscriptionSegment};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AudioTranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub segments: Option<Vec<AudioTranscriptionSegment>>,
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AudioTranscriptionSegment {
    #[serde(default)]
    pub start: Option<f64>,
    #[serde(default)]
    pub end: Option<f64>,
    pub text: String,
}

impl From<AudioTranscriptionResponse> for TranscriptionResult {
    fn from(value: AudioTranscriptionResponse) -> Self {
        Self {
            text: value.text,
            language: value.language,
            duration_seconds: value.duration,
            segments: value.segments.map(|segments| {
                segments
                    .into_iter()
                    .map(|segment| TranscriptionSegment {
                        start_seconds: segment.start,
                        end_seconds: segment.end,
                        text: segment.text,
                    })
                    .collect()
            }),
            metadata: if value.metadata.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(
                    value.metadata.into_iter().collect(),
                ))
            },
        }
    }
}
