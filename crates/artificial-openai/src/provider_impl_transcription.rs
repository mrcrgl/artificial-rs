use std::{future::Future, pin::Pin, sync::Arc};

use artificial_core::{
    error::Result,
    provider::{TranscriptionProvider, TranscriptionRequest, TranscriptionResult},
};

use crate::OpenAiAdapter;

impl TranscriptionProvider for OpenAiAdapter {
    fn transcribe<'s>(
        &'s self,
        request: TranscriptionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<TranscriptionResult>> + Send + 's>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move { Ok(client.audio_transcription(request).await?) })
    }
}
