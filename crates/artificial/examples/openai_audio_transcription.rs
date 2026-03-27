use std::path::PathBuf;

use artificial::{
    ArtificialClient,
    openai::OpenAiAdapterBuilder,
    provider::{TranscriptionProvider, TranscriptionRequest},
};

/// Basic audio transcription example.
///
/// Usage:
///   OPENAI_API_KEY=... cargo run -p artificial --example openai_audio_transcription -- path/to/audio.wav
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let audio_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("missing audio path arg"))?;

    let audio = std::fs::read(&audio_path)?;
    let mime_type = guess_mime_type(&audio_path);
    let filename = audio_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio.wav")
        .to_string();

    let backend = OpenAiAdapterBuilder::new_from_env().build()?;
    let client = ArtificialClient::new(backend);

    let request = TranscriptionRequest::new(audio, mime_type)
        .with_filename(filename)
        .with_model("gpt-4o-mini-transcribe");

    let result = client.transcribe(request).await?;

    println!("Transcription:\n{}", result.text);
    if let Some(language) = result.language {
        println!("Language: {language}");
    }
    if let Some(duration) = result.duration_seconds {
        println!("Duration: {duration:.2}s");
    }

    Ok(())
}

fn guess_mime_type(path: &PathBuf) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("mp3") => "audio/mpeg",
        Some("m4a") => "audio/mp4",
        Some("ogg") => "audio/ogg",
        Some("webm") => "audio/webm",
        Some("wav") => "audio/wav",
        _ => "application/octet-stream",
    }
}
