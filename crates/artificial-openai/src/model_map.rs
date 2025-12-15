use artificial_core::model::{Model, OpenAiModel};

const GPT5: &str = "gpt-5";
const GPT5_1: &str = "gpt-5.1";
const GPT5_2: &str = "gpt-5.2";
const GPT5_NANO: &str = "gpt-5-nano";
const GPT5_MINI: &str = "gpt-5-mini";
const GPT4_O_MINI: &str = "gpt-4o-mini";
const GPT4_O: &str = "gpt-4o";
const O3: &str = "o3";
const O3_MINI: &str = "o3-mini";
const O4_MINI: &str = "o4-mini";

pub(crate) fn map_model(model: &Model) -> Option<&'static str> {
    if let Model::Custom(custom) = *model {
        return Some(custom);
    }

    let Model::OpenAi(openai_model) = model else {
        return None;
    };

    match openai_model {
        OpenAiModel::Gpt4o => Some(GPT4_O),
        OpenAiModel::Gpt4oMini => Some(GPT4_O_MINI),
        OpenAiModel::O3 => Some(O3),
        OpenAiModel::O3Mini => Some(O3_MINI),
        OpenAiModel::O4Mini => Some(O4_MINI),
        OpenAiModel::Gpt5 => Some(GPT5),
        OpenAiModel::Gpt5Nano => Some(GPT5_NANO),
        OpenAiModel::Gpt5Mini => Some(GPT5_MINI),
        OpenAiModel::Gpt5_1 => Some(GPT5_1),
        OpenAiModel::Gpt5_2 => Some(GPT5_2),
    }
}
