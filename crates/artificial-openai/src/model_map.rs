use artificial_core::model::{Model, OpenAiModel};

const GPT5: &str = "gpt-5";
const GPT5_1: &str = "gpt-5.1";
const GPT5_2: &str = "gpt-5.2";
const GPT5_3: &str = "gpt-5.3";
const GPT5_4: &str = "gpt-5.4";
const GPT5_NANO: &str = "gpt-5-nano";
const GPT5_MINI: &str = "gpt-5-mini";
const GPT5_PRO: &str = "gpt-5-pro";
const GPT5_2_PRO: &str = "gpt-5.2-pro";
const GPT5_4_PRO: &str = "gpt-5.4-pro";
const GPT5_CODEX: &str = "gpt-5-codex";
const GPT5_1_CODEX: &str = "gpt-5.1-codex";
const GPT5_1_CODEX_MINI: &str = "gpt-5.1-codex-mini";
const GPT5_1_CODEX_MAX: &str = "gpt-5.1-codex-max";
const GPT5_2_CODEX: &str = "gpt-5.2-codex";
const GPT5_3_CODEX: &str = "gpt-5.3-codex";
const GPT4_1: &str = "gpt-4.1";
const GPT4_1_MINI: &str = "gpt-4.1-mini";
const GPT4_1_NANO: &str = "gpt-4.1-nano";
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
        OpenAiModel::Gpt5Pro => Some(GPT5_PRO),
        OpenAiModel::Gpt5_1 => Some(GPT5_1),
        OpenAiModel::Gpt5_1Codex => Some(GPT5_1_CODEX),
        OpenAiModel::Gpt5_1CodexMini => Some(GPT5_1_CODEX_MINI),
        OpenAiModel::Gpt5_1CodexMax => Some(GPT5_1_CODEX_MAX),
        OpenAiModel::Gpt5_2 => Some(GPT5_2),
        OpenAiModel::Gpt5_2Pro => Some(GPT5_2_PRO),
        OpenAiModel::Gpt5_2Codex => Some(GPT5_2_CODEX),
        OpenAiModel::Gpt5_3 => Some(GPT5_3),
        OpenAiModel::Gpt5_3Codex => Some(GPT5_3_CODEX),
        OpenAiModel::Gpt5_4 => Some(GPT5_4),
        OpenAiModel::Gpt5_4Pro => Some(GPT5_4_PRO),
        OpenAiModel::Gpt5Codex => Some(GPT5_CODEX),
        OpenAiModel::Gpt4_1 => Some(GPT4_1),
        OpenAiModel::Gpt4_1Mini => Some(GPT4_1_MINI),
        OpenAiModel::Gpt4_1Nano => Some(GPT4_1_NANO),
    }
}
