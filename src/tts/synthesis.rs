//文本转语音合成服务
//使用各种TTS提供程序提供语音合成功能
use crate::common::config::{AudioCodec, AudioData, AudioFormat, SynthesisConfig};
use crate::common::error::{AgentError, Result, TtsError};
use crate::tts::kokoro_tts::KokoroTts;
use crate::tts::kokoro_tts_worker::KokoroTtsWorker;
use crate::tts::kokoro_voice::KokoroVoice;
use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use tokio::time::timeout;

// 静态的TTS实例初始化(懒加载)
#[allow(dead_code)]
static TTS_INSTANCE: LazyLock<Mutex<Option<Arc<KokoroTts>>>> = LazyLock::new(|| Mutex::new(None));
// 获取一个TTS实例的引用
#[allow(dead_code)]
fn get_tts() -> std::result::Result<Arc<KokoroTts>, TtsError> {
    let mut tts_guard = TTS_INSTANCE
        .lock()
        .map_err(|_| TtsError::init("Failed to acquire TTS instance lock".to_string()))?;

    if tts_guard.is_none() {
        // 新建tts实例如果未实例化
        let tts = KokoroTts::new();
        *tts_guard = Some(Arc::new(tts));
    }

    tts_guard
        .clone()
        .ok_or_else(|| TtsError::init("TTS initialization failed".to_string()))
}

/// 语音合成结果
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SynthesisResult {
    /// 生成的音频数据
    pub audio: AudioData,
    /// 处理持续时间（毫秒）
    pub processing_duration_ms: u64,
    /// 用于合成的语音(人声)
    pub voice: String,
    /// 合成的文本
    pub text: String,
    /// 结果的音频格式
    pub format: AudioFormat,
}

/// 语音信息
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct VoiceInfo {
    /// 语音标识id
    pub id: String,
    /// 人可读的语音名称，例如 普通话(晓晓)
    pub name: String,
    /// 语言代码（例如“en-US”、“es-es”）
    pub language: String,
    /// 声音的性别
    pub gender: VoiceGender,
    /// 语音质量/类型
    pub quality: VoiceQuality,
    /// 此语音是否支持SSML
    pub supports_ssml: bool,
    /// 此语音的采样率
    pub sample_rate: u32,
}

/// 语音性别枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum VoiceGender {
    Male,    //男
    Female,  //女
    Neutral, //中性
    Unknown, //未知
}

/// 语音质量枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum VoiceQuality {
    Standard, //标准
    Premium,  //高级
    Neural,   //中性
    Custom,   //定制
}

/// TTS服务提供商
#[derive(Debug, Clone)]
pub enum SynthesisProvider {
    /// OpenAI TTS API
    OpenAI,
    /// Azure Cognitive Services Speech
    Azure,
    /// 谷歌 云文本转语音
    Google,
    ///亚马逊 Polly
    Amazon,
    /// Kokoro TTS
    Kokoro,
}

impl SynthesisProvider {
    /// 从字符串中获取提供者
    pub fn from_string(provider: &str) -> Option<Self> {
        match provider.to_lowercase().as_str() {
            "openai" => Some(SynthesisProvider::OpenAI),
            "azure" => Some(SynthesisProvider::Azure),
            "google" => Some(SynthesisProvider::Google),
            "amazon" | "polly" => Some(SynthesisProvider::Amazon),
            "kokoro" => Some(SynthesisProvider::Kokoro),
            _ => None,
        }
    }
}

/// 文本到语音合成服务
#[derive(Debug)]
pub struct SynthesisService {
    /// 服务配置
    config: SynthesisConfig,
    /// API请求的HTTP客户端
    client: Client,
    /// 服务提供商
    provider: SynthesisProvider,
}

impl SynthesisService {
    /// 创建新的合成服务
    pub fn new(config: SynthesisConfig) -> Result<Self> {
        let provider = SynthesisProvider::from_string(&config.provider).ok_or_else(|| {
            AgentError::invalid_input(format!(
                "Unsupported synthesis provider: {}",
                config.provider
            ))
        })?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Failed to create HTTP client: {}", e),
                )
            })?;

        Ok(Self {
            config,
            client,
            provider,
        })
    }
    /// 将文本合成为语音
    #[allow(dead_code)]
    pub async fn synthesize(&self, text: &str) -> Result<SynthesisResult> {
        let start_time = Instant::now();
        info!(
            "Starting speech synthesis with provider: {:?}",
            self.provider
        );

        // 验证输入文本
        self.validate_text(text)?;

        // 根据提供者进行合成
        let result = match self.provider {
            SynthesisProvider::OpenAI => self.synthesize_with_openai(text).await,
            SynthesisProvider::Azure => self.synthesize_with_azure(text).await,
            SynthesisProvider::Google => self.synthesize_with_google(text).await,
            SynthesisProvider::Amazon => self.synthesize_with_amazon(text).await,
            SynthesisProvider::Kokoro => self.synthesize_with_kokoro(text).await,
        };

        match result {
            Ok(mut synthesis) => {
                synthesis.processing_duration_ms = start_time.elapsed().as_millis() as u64;
                info!(
                    "Speech synthesis completed in {}ms",
                    synthesis.processing_duration_ms
                );
                Ok(synthesis)
            }
            Err(e) => {
                error!("Speech synthesis failed: {}", e);
                Err(e)
            }
        }
    }

    /// 获取当前提供商的可用语音
    #[allow(dead_code)]
    pub async fn get_voices(&self) -> Result<Vec<VoiceInfo>> {
        match self.provider {
            SynthesisProvider::OpenAI => self.get_openai_voices().await,
            SynthesisProvider::Azure => self.get_azure_voices().await,
            SynthesisProvider::Google => self.get_google_voices().await,
            SynthesisProvider::Amazon => self.get_amazon_voices().await,
            SynthesisProvider::Kokoro => self.get_kokoro_voices().await,
        }
    }

    /// 验证用于合成的输入文本
    fn validate_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Err(AgentError::invalid_input(
                "Text cannot be empty".to_string(),
            ));
        }

        if text.len() > 4000 {
            return Err(AgentError::invalid_input(
                "Text too long for synthesis (max 4000 characters)".to_string(),
            ));
        }

        // 检查是否存在潜在的问题字符
        if text.chars().any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t') {
            warn!("Text contains control characters that may affect synthesis");
        }

        Ok(())
    }

    /// 使用OpenAI TTS API进行合成
    async fn synthesize_with_openai(&self, text: &str) -> Result<SynthesisResult> {
        let api_key = self.config.api_key.as_ref().ok_or_else(|| {
            AgentError::authentication("OpenAI API key not configured".to_string())
        })?;

        // 准备请求body
        let request_body = serde_json::json!({
            "model": "tts-1",
            "input": text,
            "voice": self.config.voice,
            "response_format": match self.config.output_format {
                AudioFormat::Mp3 => "mp3",
                AudioFormat::Wav => "wav",
                AudioFormat::Flac => "flac",
                AudioFormat::Aac => "aac",
                _ => "mp3",
            },
            "speed": self.config.speed,
        });

        // 发出API请求
        let endpoint = self
            .config
            .endpoint
            .as_deref()
            .unwrap_or("https://api.openai.com/v1/audio/speech");

        let response = timeout(
            Duration::from_secs(self.config.timeout),
            self.client
                .post(endpoint)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send(),
        )
        .await
        .map_err(|_| AgentError::tool("synthesis".to_string(), "Request timeout".to_string()))?
        .map_err(|e| AgentError::tool("synthesis".to_string(), format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("API error: {}", error_text),
            ));
        }

        // 获取音频字节数
        let audio_bytes = response.bytes().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to read audio data: {}", e),
            )
        })?;

        // 解码音频数据
        let audio = AudioCodec::decode_bytes(&audio_bytes, self.config.output_format)?;

        Ok(SynthesisResult {
            audio,
            processing_duration_ms: 0, // 将由调用者设置
            voice: self.config.voice.clone(),
            text: text.to_string(),
            format: self.config.output_format,
        })
    }

    /// 获取OpenAI语音列表
    async fn get_openai_voices(&self) -> Result<Vec<VoiceInfo>> {
        // OpenAI TTS voices (as of 2024)
        Ok(vec![
            VoiceInfo {
                id: "alloy".to_string(),
                name: "Alloy".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Neutral,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "echo".to_string(),
                name: "Echo".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "fable".to_string(),
                name: "Fable".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "onyx".to_string(),
                name: "Onyx".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "nova".to_string(),
                name: "Nova".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "shimmer".to_string(),
                name: "Shimmer".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
        ])
    }

    /// 使用Azure认知服务进行合成
    async fn synthesize_with_azure(&self, text: &str) -> Result<SynthesisResult> {
        debug!("Starting Azure Cognitive Services synthesis");

        let subscription_key = std::env::var("AZURE_SPEECH_KEY").map_err(|_| {
            AgentError::tool(
                "synthesis".to_string(),
                "AZURE_SPEECH_KEY environment variable not set".to_string(),
            )
        })?;

        let region = std::env::var("AZURE_SPEECH_REGION").unwrap_or_else(|_| "eastus".to_string());

        // 获取访问令牌
        let token = self.get_azure_access_token(&subscription_key, &region).await?;

        // 为Azure TTS创建SSML
        let ssml = self.create_azure_ssml(text, &self.config.voice).await?;

        // 创建合成请求
        let endpoint = format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/v1",
            region
        );

        let response = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/ssml+xml")
            .header(
                "X-Microsoft-OutputFormat",
                "audio-16khz-32kbitrate-mono-mp3",
            )
            .header("User-Agent", "rust-agent/1.0")
            .body(ssml)
            .send()
            .await
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Azure API request failed: {}", e),
                )
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("Azure API error: {}", error_text),
            ));
        }

        let audio_bytes = response.bytes().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to read Azure response: {}", e),
            )
        })?;

        info!(
            "Azure synthesis completed: {} bytes generated",
            audio_bytes.len()
        );

        // 将音频字节解码为AudioData
        let audio = AudioCodec::decode_bytes(&audio_bytes, AudioFormat::Mp3)?;

        Ok(SynthesisResult {
            audio,
            processing_duration_ms: 0, // Will be set by caller
            voice: self.config.voice.clone(),
            text: text.to_string(),
            format: AudioFormat::Mp3,
        })
    }

    /// 获取Azure语音列表
    async fn get_azure_voices(&self) -> Result<Vec<VoiceInfo>> {
        debug!("Getting Azure voices list");

        let subscription_key = std::env::var("AZURE_SPEECH_KEY").map_err(|_| {
            AgentError::tool(
                "synthesis".to_string(),
                "AZURE_SPEECH_KEY environment variable not set".to_string(),
            )
        })?;

        let region = std::env::var("AZURE_SPEECH_REGION").unwrap_or_else(|_| "eastus".to_string());

        // 获取访问令牌
        let token = self.get_azure_access_token(&subscription_key, &region).await?;

        // 获取语音列表
        let endpoint = format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/voices/list",
            region
        );

        let response = self
            .client
            .get(&endpoint)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Azure API request failed: {}", e),
                )
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("Azure API error: {}", error_text),
            ));
        }

        let voices_json: serde_json::Value = response.json().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to parse Azure response: {}", e),
            )
        })?;

        let mut voices = Vec::new();

        if let Some(voices_array) = voices_json.as_array() {
            for voice in voices_array {
                if let (Some(name), Some(display_name), Some(locale)) = (
                    voice.get("ShortName").and_then(|v| v.as_str()),
                    voice.get("DisplayName").and_then(|v| v.as_str()),
                    voice.get("Locale").and_then(|v| v.as_str()),
                ) {
                    let gender_str =
                        voice.get("Gender").and_then(|v| v.as_str()).unwrap_or("Unknown");

                    let gender = match gender_str.to_lowercase().as_str() {
                        "male" => VoiceGender::Male,
                        "female" => VoiceGender::Female,
                        "neutral" => VoiceGender::Neutral,
                        _ => VoiceGender::Unknown,
                    };

                    voices.push(VoiceInfo {
                        id: name.to_string(),
                        name: display_name.to_string(),
                        language: locale.to_string(),
                        gender,
                        quality: VoiceQuality::Neural, // Azure语音通常是中性的
                        supports_ssml: true,           // Azure支持SSML
                        sample_rate: 16000,            // Azure的默认采样率
                    });
                }
            }
        }

        info!("Retrieved {} Azure voices", voices.len());
        Ok(voices)
    }

    /// 使用Google Cloud文本转语音进行合成(暂未实现)
    async fn synthesize_with_google(&self, _text: &str) -> Result<SynthesisResult> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Google Cloud Text-to-Speech not implemented yet".to_string(),
        ))
    }

    /// 获取谷歌语音列表
    async fn get_google_voices(&self) -> Result<Vec<VoiceInfo>> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Google Cloud Text-to-Speech not implemented yet".to_string(),
        ))
    }

    /// 使用Amazon Polly进行合成(暂未实现)
    async fn synthesize_with_amazon(&self, _text: &str) -> Result<SynthesisResult> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Amazon Polly not implemented yet".to_string(),
        ))
    }

    /// 获取亚马逊语音列表
    async fn get_amazon_voices(&self) -> Result<Vec<VoiceInfo>> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Amazon Polly not implemented yet".to_string(),
        ))
    }

    /// 使用Kokoro TTS引擎进行合成
    async fn synthesize_with_kokoro(&self, text: &str) -> Result<SynthesisResult> {
        let tts = KokoroTtsWorker::new();
        let audio_result = tts.speak(
            text,
            KokoroVoice::from_string(&self.config.voice).unwrap_or(KokoroVoice::Man1),
            1.0,
        );
        match audio_result {
            Ok(audio) => {
                let audio_data = AudioData {
                    samples: audio.samples,
                    sample_rate: audio.sample_rate,
                    channels: 0,
                    duration: audio.duration as f64,
                    format: AudioFormat::Wav,
                };
                Ok(SynthesisResult {
                    audio: audio_data,
                    processing_duration_ms: 0, // Will be set by caller
                    voice: self.config.voice.clone(),
                    text: text.to_string(),
                    format: self.config.output_format,
                })
            }
            Err(e) => Err(AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to synthesize audio: {}", e),
            )),
        }
    }

    /// 获取Kokoro语音列表
    async fn get_kokoro_voices(&self) -> Result<Vec<VoiceInfo>> {
        let mut voice_infos: Vec<VoiceInfo> = Vec::new();
        // 遍历KokoroVoice所有枚举值
        for voice in KokoroVoice::iter() {
            println!("{:?}: {}", voice, voice.get_name());
            let voice_info = VoiceInfo {
                id: (voice as i32).to_string(),
                name: voice.get_name(),
                language: "zh-CN".to_string(),
                gender: if voice.get_is_man() {
                    VoiceGender::Male
                } else {
                    VoiceGender::Female
                },
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            };
            voice_infos.push(voice_info);
        }
        Ok(voice_infos)
    }

    /// 检查是否有语音可用
    #[allow(dead_code)]
    pub async fn is_voice_available(&self, voice_id: &str) -> Result<bool> {
        let voices = self.get_voices().await?;
        Ok(voices.iter().any(|v| v.id == voice_id))
    }

    /// 通过ID获取语音信息
    #[allow(dead_code)]
    pub async fn get_voice_info(&self, voice_id: &str) -> Result<Option<VoiceInfo>> {
        let voices = self.get_voices().await?;
        Ok(voices.into_iter().find(|v| v.id == voice_id))
    }

    /// 按语言获取声音列表
    #[allow(dead_code)]
    pub async fn get_voices_by_language(&self, language: &str) -> Result<Vec<VoiceInfo>> {
        let voices = self.get_voices().await?;
        Ok(voices.into_iter().filter(|v| v.language.starts_with(language)).collect())
    }

    /// 按性别获取声音列表
    #[allow(dead_code)]
    pub async fn get_voices_by_gender(&self, gender: VoiceGender) -> Result<Vec<VoiceInfo>> {
        let voices = self.get_voices().await?;
        Ok(voices
            .into_iter()
            .filter(|v| std::mem::discriminant(&v.gender) == std::mem::discriminant(&gender))
            .collect())
    }

    // ========================================
    // Azure语音服务相关方法
    // ========================================

    /// 获取语音服务的Azure访问令牌
    async fn get_azure_access_token(&self, subscription_key: &str, region: &str) -> Result<String> {
        let token_endpoint = format!(
            "https://{}.api.cognitive.microsoft.com/sts/v1.0/issuetoken",
            region
        );

        let response = self
            .client
            .post(&token_endpoint)
            .header("Ocp-Apim-Subscription-Key", subscription_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Failed to get Azure token: {}", e),
                )
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("Azure token request failed: {}", error_text),
            ));
        }

        let token = response.text().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to read Azure token: {}", e),
            )
        })?;

        Ok(token)
    }

    /// 为Azure TTS创建SSML
    async fn create_azure_ssml(&self, text: &str, voice: &str) -> Result<String> {
        // Create SSML with proper voice and language settings
        let ssml = format!(
            r#"<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{}">
                    <prosody rate="{:.1}">
                        {}
                    </prosody>
                </voice>
            </speak>"#,
            voice,
            self.config.speed,
            self.escape_ssml_text(text)
        );

        Ok(ssml)
    }

    /// SSML的转义文本
    fn escape_ssml_text(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// 基于文本和语音速率估计音频持续时间
    #[allow(dead_code)]
    pub fn estimate_audio_duration(&self, text: &str, speed: f32) -> f64 {
        // 平均语速约为每分钟150个单词
        let base_wpm = 150.0f64;
        let adjusted_wpm = base_wpm * speed as f64;
        let word_count = text.split_whitespace().count() as f64;
        let duration_minutes = word_count / adjusted_wpm;
        duration_minutes * 60.0 // 转换为秒
    }
}

/// 使用默认配置创建合成服务(默认是用openai)
#[allow(dead_code)]
pub fn create_default_synthesis_service() -> Result<SynthesisService> {
    let config = SynthesisConfig::default();
    SynthesisService::new(config)
}

/// 使用OpenAI TTS创建合成服务
#[allow(dead_code)]
pub fn create_openai_synthesis_service(api_key: String) -> Result<SynthesisService> {
    let config = SynthesisConfig {
        provider: "openai".to_string(),
        api_key: Some(api_key),
        voice: "alloy".to_string(),
        ..Default::default()
    };
    SynthesisService::new(config)
}

/// 使用kokoro TTS创建合成服务
#[allow(dead_code)]
pub fn create_kokoro_synthesis_service(voice_id: u16, speed: f32) -> Result<SynthesisService> {
    let config = SynthesisConfig {
        provider: "kokoro".to_string(),
        voice: voice_id.to_string(),
        speed,
        ..Default::default()
    };
    SynthesisService::new(config)
}

/// 估计语音持续时间的工具函数
#[allow(dead_code)]
pub fn estimate_speech_duration(text: &str, words_per_minute: f32) -> Duration {
    let word_count = text.split_whitespace().count() as f32;
    let minutes = word_count / words_per_minute;
    Duration::from_secs_f32(minutes * 60.0)
}

/// 将长文本分割成块的函数
#[allow(dead_code)]
pub fn split_text_for_synthesis(text: &str, max_chunk_size: usize) -> Vec<String> {
    if text.len() <= max_chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for sentence in text.split('.') {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        let sentence_with_period = format!("{}.", sentence);

        if current_chunk.len() + sentence_with_period.len() > max_chunk_size {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }

            // 如果单句太长，按单词拆分
            if sentence_with_period.len() > max_chunk_size {
                let words: Vec<&str> = sentence_with_period.split_whitespace().collect();
                let mut word_chunk = String::new();

                for word in words {
                    if word_chunk.len() + word.len() + 1 > max_chunk_size && !word_chunk.is_empty()
                    {
                        chunks.push(word_chunk.trim().to_string());
                        word_chunk.clear();
                    }

                    if !word_chunk.is_empty() {
                        word_chunk.push(' ');
                    }
                    word_chunk.push_str(word);
                }

                if !word_chunk.is_empty() {
                    current_chunk = word_chunk;
                }
            } else {
                current_chunk = sentence_with_period;
            }
        } else {
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&sentence_with_period);
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}
