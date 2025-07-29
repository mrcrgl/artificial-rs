use std::sync::Arc;

use artificial_core::{
    error::ArtificialError,
    generic::{GenericChatCompletionResponse, GenericUsageReport},
    provider::{ChatCompleteParameters, ChatCompletionProvider},
};

use crate::{
    OpenAiAdapter,
    api_v1::{ChatCompletionMessage, ChatCompletionRequest, FinishReason},
    error::OpenAiError,
    model_map::map_model,
};

impl ChatCompletionProvider for OpenAiAdapter {
    type Message = ChatCompletionMessage;

    fn chat_complete<'p, M>(
        &self,
        params: ChatCompleteParameters<M>,
    ) -> std::pin::Pin<
        Box<
            dyn Future<
                    Output = artificial_core::error::Result<
                        artificial_core::generic::GenericChatCompletionResponse<
                            artificial_core::generic::GenericChatResponseMessage,
                        >,
                    >,
                > + Send
                + 'p,
        >,
    >
    where
        M: Into<Self::Message> + Send + Sync + 'p,
    {
        let client = Arc::clone(&self.client);

        Box::pin(async move {
            let model = params.model();
            let model = map_model(&model).ok_or(ArtificialError::InvalidRequest(format!(
                "backend does not support selected model: {:?}",
                model
            )))?;
            let messages = params.into_messages().into_iter().map(Into::into).collect();

            let request = ChatCompletionRequest::new(model.into(), messages);

            let mut response = client.chat_completion(request).await?;

            let usage_report = GenericUsageReport {
                prompt_tokens: response.usage.prompt_tokens as i64,
                completion_tokens: response.usage.completion_tokens as i64,
                total_tokens: response.usage.total_tokens as i64,
            };

            let Some(first_choice) = response.choices.pop() else {
                return Err(OpenAiError::Format("response has no choices".into()).into());
            };

            match &first_choice.finish_reason {
                Some(FinishReason::ToolCalls) => {
                    todo!()
                }
                None | Some(FinishReason::Stop) => {
                    let response = GenericChatCompletionResponse {
                        content: first_choice.message.into(),
                        usage: Some(usage_report),
                    };
                    Ok(response)
                }
                Some(other) => Err(OpenAiError::Format(format!(
                    "unhandled finish reason on API: {other:?}"
                ))
                .into()),
            }
        })
    }
}
