use reqwest::Client;
use serde_json::json;
use std::error::Error;

static ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";

pub struct QwenTTS {
    api_key: String,
    text: String,
    voice: String,
    model: String,
}

impl QwenTTS {
    pub fn new(api_key: String, text: String, voice: String, model: String) -> Self {
        Self {
            api_key,
            text,
            voice,
            model,
        }
    }

    pub async fn send(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        if self.api_key.is_empty() {
            return Err("Missing Qwen API key".into());
        }
        let client = Client::new();
        let res = client
            .post(ENDPOINT)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": self.model,
                "input": {
                    "text": self.text,
                    "voice": self.voice
                }
            }))
            .send()
            .await?;

        let res = res.error_for_status()?;
        let data = res.json::<serde_json::Value>().await?;

        // Check if the response is successful
        if let Some(output) = data.get("output") {
            if let Some(audio) = output.get("audio") {
                if let Some(audio_url) = audio.get("url") {
                    if let Some(url) = audio_url.as_str() {
                        let audio_response = client.get(url).send().await?;
                        let audio_bytes = audio_response.bytes().await?;
                        return Ok(audio_bytes.to_vec());
                    }
                }
            }
        }

        Err(format!("Qwen API error: {:?}", data).into())
    }
}

pub async fn request(
    api_key: &str,
    text: &str,
    voice: &str,
    model: &str,
) -> Result<Vec<u8>, String> {
    let tts = QwenTTS::new(
        api_key.to_string(),
        text.to_string(),
        voice.to_string(),
        model.to_string(),
    );
    tts.send().await.map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_request() {
        let result = request("hello world", "hello world", "", "").await;
        match result {
            Ok(res) => {
                println!("{:?}", res);
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
}
