use std::{future::Future, pin::Pin};

use crate::error::Result;

/// Provider-agnostic audio transcription request.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    pub audio: Vec<u8>,
    pub mime_type: String,
    pub filename: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub model: Option<String>,
}

impl TranscriptionRequest {
    pub fn new(audio: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self {
            audio,
            mime_type: mime_type.into(),
            filename: None,
            language: None,
            prompt: None,
            model: None,
        }
    }

    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub duration_seconds: Option<f64>,
    pub segments: Option<Vec<TranscriptionSegment>>,
    pub metadata: Option<serde_json::Value>,
}

/// Provider capability for converting audio to text.
pub trait TranscriptionProvider: Send + Sync {
    fn transcribe<'s>(
        &'s self,
        request: TranscriptionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<TranscriptionResult>> + Send + 's>>;
}
