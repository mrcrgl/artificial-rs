use std::borrow::Cow;

use artificial_core::model::{Model, OpenAiModel};

const GPT4_O_MINI: &str = "gpt-4o-mini";
const GPT4_O: &str = "gpt-4o";
const O3: &str = "o3";
const O3_MINI: &str = "o3-mini";
const O4_MINI: &str = "o4-mini";

pub(crate) fn map_model(model: &Model) -> Option<Cow<'static, str>> {
    if let Model::Custom(custom) = model {
        return Some(custom.clone());
    }

    let Model::OpenAi(openai_model) = model else {
        return None;
    };

    match openai_model {
        OpenAiModel::Gpt4o => Some(GPT4_O.into()),
        OpenAiModel::Gpt4oMini => Some(GPT4_O_MINI.into()),
        OpenAiModel::O3 => Some(O3.into()),
        OpenAiModel::O3Mini => Some(O3_MINI.into()),
        OpenAiModel::O4Mini => Some(O4_MINI.into()),
    }
}
