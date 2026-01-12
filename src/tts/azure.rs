use reqwest::Client;
use std::error::Error;

fn build_ssml(
    text: &str,
    speaker: &str,
    language: &str,
    style: &str,
    role: &str,
    rate: &str,
    pitch: &str,
) -> String {
    format!(
        r#"
        <speak version="1.0"
                xmlns="http://www.w3.org/2001/10/synthesis"
                xmlns:mstts="http://www.w3.org/2001/mstts"
                xmlns:emo="http://www.w3.org/2009/10/emotionml"
                xml:lang="{}"
            >
            <voice name="{}">
                <mstts:express-as style="{}" role="{}">
                    <prosody rate="{}%" pitch="{}%">
                        {}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>
    "#,
        language, speaker, style, role, rate, pitch, text
    )
}

async fn send(
    endpoint: &str,
    subscription_key: &str,
    xml: String,
) -> Result<Vec<u8>, Box<dyn Error>> {
    if endpoint.is_empty() {
        return Err("Missing Azure endpoint".into());
    }
    if subscription_key.is_empty() {
        return Err("Azure Subscription key is empty".into());
    }
    let client = Client::new();
    let res = client
        .post(endpoint)
        .header("Ocp-Apim-Subscription-Key", subscription_key)
        .header("Content-Type", "application/ssml+xml")
        .header(
            "X-Microsoft-OutputFormat",
            "audio-24khz-160kbitrate-mono-mp3",
        )
        .header("User-Agent", "curl")
        .body(xml)
        .send()
        .await?;

    if res.status().is_success() {
        let audio = res.bytes().await?;
        Ok(audio.to_vec())
    } else {
        Err(format!("Azure TTS failed with status code: {}", res.status()).into())
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn request(
    endpoint: &str,
    subscription_key: &str,
    text: &str,
    speaker: &str,
    language: &str,
    style: &str,
    role: &str,
    rate: &str,
    pitch: &str,
    raw_ssml: bool,
) -> Result<Vec<u8>, String> {
    let xml = if raw_ssml {
        text.to_string()
    } else {
        build_ssml(text, speaker, language, style, role, rate, pitch)
    };
    send(endpoint, subscription_key, xml)
        .await
        .map_err(|e| format!("Azure TTS failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_request() {
        let result = request(
            "",
            "",
            "hello world",
            "zh-CN-YunxiNeural-Male",
            "",
            "",
            "",
            "",
            "",
            false,
        )
        .await;
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
