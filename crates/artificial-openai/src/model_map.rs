use std::borrow::Cow;

use artificial_core::model::{Model, OpenAiModel};

pub const GPT4_O_MINI: &str = "gpt-4o-mini";
pub const GPT4_O: &str = "gpt-4o";

pub(crate) fn map_model(model: Model) -> Option<Cow<'static, str>> {
    if let Model::Custom(custom) = model {
        return Some(custom);
    }

    let Model::OpenAi(openai_model) = model else {
        return None;
    };

    match openai_model {
        OpenAiModel::Gpt4o => Some(GPT4_O.into()),
        OpenAiModel::Gpt4oMini => Some(GPT4_O_MINI.into()),
    }
}
