use std::pin::Pin;

use crate::OpenAiAdapter;
use crate::api_v1::ChatCompletionRequest;
use artificial_core::error::{ArtificialError, Result};
use artificial_core::provider::{ChatCompleteParameters, StreamingChatProvider};
use futures_core::stream::Stream;

impl StreamingChatProvider for OpenAiAdapter {
    type Delta<'s>
        = Pin<Box<dyn Stream<Item = Result<String>> + Send + 's>>
    where
        Self: 's;

    fn chat_complete_stream<'p, M>(&self, params: ChatCompleteParameters<M>) -> Self::Delta<'p>
    where
        M: Into<Self::Message> + Send + Sync + 'p,
    {
        let client = self.client.clone();

        Box::pin(async_stream::try_stream! {
        use futures_util::StreamExt;

        let request: ChatCompletionRequest = params.try_into()?;


            let stream = client.chat_completion_stream(request);
            futures_util::pin_mut!(stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(ArtificialError::from)?;
                for choice in chunk.choices {
                    if let Some(text) = choice.delta.content {
                        yield text;
                    }
                }
            }
        })
    }
}
