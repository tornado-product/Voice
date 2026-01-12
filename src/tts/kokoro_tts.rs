use crate::tts::kokoro_voice::KokoroVoice;
use anyhow::anyhow;
use log::{info, warn};
use sherpa_rs::tts::{TtsAudio, VitsTts, VitsTtsConfig};
use std::path::{Path, PathBuf};

/// TTS 配置
#[derive(Debug, Clone)]
pub struct KokoroTtsConfig {
    pub model_path: PathBuf,
    pub tokens_path: PathBuf,
    pub dict_dir: PathBuf,
    pub lexicon_path: PathBuf,
    pub length_scale: f32,
}

impl Default for KokoroTtsConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./sherpa-onnx-vits-zh-ll/model.onnx"),
            tokens_path: PathBuf::from("./sherpa-onnx-vits-zh-ll/tokens.txt"),
            dict_dir: PathBuf::from("./sherpa-onnx-vits-zh-ll/dict"),
            lexicon_path: PathBuf::from("./sherpa-onnx-vits-zh-ll/lexicon.txt"),
            length_scale: 1.0,
        }
    }
}

/// 内部 TTS 引擎（非线程安全）
pub struct KokoroTts {
    tts: Option<VitsTts>,
    config: KokoroTtsConfig,
}

impl KokoroTts {
    /// 创建新的 TTS 引擎实例（不自动加载模型）
    pub fn new() -> Self {
        Self {
            tts: None,
            config: KokoroTtsConfig::default(),
        }
    }

    /// 使用默认配置创建并自动加载模型
    #[allow(dead_code)]
    pub fn new_with_default_model() -> anyhow::Result<Self> {
        let mut instance = Self::new();
        instance.load_default_model()?;
        Ok(instance)
    }

    /// 加载默认模型
    pub fn load_default_model(&mut self) -> anyhow::Result<bool> {
        let config = KokoroTtsConfig::default();
        self.load_model_with_config(config)
    }

    /// 动态加载模型（简化版：只需要模型文件路径）
    /// 其他路径根据模型路径自动推导
    pub fn load_model(&mut self, model_path: PathBuf) -> anyhow::Result<bool> {
        // 验证模型文件存在
        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {}", model_path.display()));
        }

        // 推导其他文件路径（假设标准目录结构）
        let model_dir = model_path.parent().ok_or_else(|| anyhow!("Invalid model path"))?;

        let config = KokoroTtsConfig {
            model_path: model_path.clone(),
            tokens_path: model_dir.join("tokens.txt"),
            dict_dir: model_dir.join("dict"),
            lexicon_path: model_dir.join("lexicon.txt"),
            length_scale: 1.0,
        };

        self.load_model_with_config(config)
    }

    /// 使用完整配置加载模型（高级用法）
    pub fn load_model_with_config(&mut self, config: KokoroTtsConfig) -> anyhow::Result<bool> {
        info!("Loading TTS model from: {}", config.model_path.display());

        // 验证所有必需文件
        self.validate_model_files(&config)?;

        // 创建 VITS 配置
        let vits_config = VitsTtsConfig {
            model: config.model_path.to_string_lossy().to_string(),
            tokens: config.tokens_path.to_string_lossy().to_string(),
            dict_dir: config.dict_dir.to_string_lossy().to_string(),
            lexicon: config.lexicon_path.to_string_lossy().to_string(),
            length_scale: config.length_scale,
            ..Default::default()
        };
        self.tts = Some(VitsTts::new(vits_config));
        self.config = config.clone();
        info!("TTS model loaded successfully");
        Ok(true)
    }

    /// 验证模型文件完整性
    fn validate_model_files(&self, config: &KokoroTtsConfig) -> anyhow::Result<()> {
        let required_files = vec![
            (&config.model_path, "Model file"),
            (&config.tokens_path, "Tokens file"),
            (&config.lexicon_path, "Lexicon file"),
        ];

        for (path, name) in required_files {
            if !path.exists() {
                return Err(anyhow!("{} not found: {}", name, path.display()));
            }
        }

        // 验证字典目录
        if !config.dict_dir.exists() {
            warn!(
                "Dictionary directory not found: {}",
                config.dict_dir.display()
            );
            // 字典目录可选，不强制要求
        }

        Ok(())
    }

    /// 检查模型是否已加载
    pub fn is_loaded(&self) -> bool {
        self.tts.is_some()
    }

    /// 获取当前加载的模型路径
    pub fn get_model_path(&self) -> Option<&Path> {
        if self.is_loaded() {
            Some(&self.config.model_path)
        } else {
            None
        }
    }

    /// 生成音频数据
    pub fn speak(
        &mut self,
        text: &str,
        voice: KokoroVoice,
        speed: f32,
    ) -> anyhow::Result<TtsAudio> {
        let tts = self
            .tts
            .as_mut()
            .ok_or_else(|| anyhow!("TTS model not loaded. Call load_model() first"))?;

        tts.create(text, voice.into(), speed)
            .map_err(|e| anyhow!("TTS generation failed: {}", e))
    }

    /// 生成音频并保存到文件
    pub fn speak_to_file(
        &mut self,
        text: &str,
        voice: KokoroVoice,
        speed: f32,
        save_dir: Option<&str>,
    ) -> anyhow::Result<String> {
        let audio = self.speak(text, voice, speed)?;

        // 生成唯一文件名
        use uuid::Uuid;
        let filename = format!("{}.wav", Uuid::new_v4());
        let save_dir = save_dir.unwrap_or("./audio");
        let save_path = Path::new(save_dir);
        let full_path = save_path.join(filename);

        // 确保目录存在
        if !save_path.exists() {
            std::fs::create_dir_all(save_path).map_err(|e| {
                anyhow!("Failed to create directory {}: {}", save_path.display(), e)
            })?;
        }

        // 写入音频文件
        sherpa_rs::write_audio_file(
            &full_path.to_string_lossy(),
            &audio.samples,
            audio.sample_rate,
        )
        .map_err(|e| anyhow!("Failed to write audio file: {}", e))?;

        info!("Audio saved to: {}", full_path.display());
        Ok(full_path.to_string_lossy().to_string())
    }

    /// 卸载当前模型（释放内存）
    pub fn unload_model(&mut self) {
        if self.tts.is_some() {
            info!("Unloading TTS model");
            self.tts = None;
        }
    }
}

impl Default for KokoroTts {
    fn default() -> Self {
        Self::new()
    }
}
