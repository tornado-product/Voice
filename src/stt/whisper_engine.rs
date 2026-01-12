use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::thread::available_parallelism;
use thiserror::Error;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Calculate optimal thread count: 75% of available CPUs, minimum 1
/// Leaves headroom for UI responsiveness and system tasks
fn optimal_thread_count() -> i32 {
    let cpus = available_parallelism().map(|p| p.get()).unwrap_or(4); // Fallback to 4 if detection fails

    let threads = ((cpus as f32) * 0.75).ceil() as i32;
    threads.max(1) // At least 1 thread
}

#[derive(Error, Debug)]
pub enum WhisperError {
    #[error("Model not loaded")]
    NotLoaded,
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Transcription failed: {0}")]
    TranscriptionError(String),
    #[error("Invalid audio data")]
    InvalidAudio,
}

/// Résultat du chargement du modèle
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadResult {
    /// Chargement réussi
    pub success: bool,
    /// GPU utilisé pour le modèle
    pub using_gpu: bool,
    /// Backend utilisé (nom)
    pub backend: String,
    /// Fallback CPU utilisé après échec GPU
    pub fallback_used: bool,
}

/// 带时间的语音转文本片段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperSegment {
    /// 分段文本
    pub text: String,
    /// 开始时间（秒）
    pub start: f64,
    /// 结束时间（秒）
    pub end: f64,
}
/// whisper的转换结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperResult {
    /// 所有文本
    pub all_text: String,
    /// WhisperSegment列表
    pub segments: Vec<WhisperSegment>,
}
impl Default for WhisperResult {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperResult {
    pub fn new() -> Self {
        Self {
            all_text: "".to_string(),
            segments: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub model_path: PathBuf,
    pub language: Option<String>,
    pub translate: bool,
    pub n_threads: i32,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        let threads = optimal_thread_count();
        info!("Whisper using {} threads (75% of available CPUs)", threads);
        Self {
            model_path: PathBuf::new(),
            language: None, // Auto-detect
            translate: false,
            n_threads: threads,
        }
    }
}

/// Whisper transcription engine using whisper-rs native bindings
pub struct WhisperEngine {
    context: Option<WhisperContext>,
    config: WhisperConfig,
    /// Track if GPU is being used for transcription
    using_gpu: bool,
    /// Track if fallback to CPU was used
    fallback_used: bool,
}

impl WhisperEngine {
    pub fn new() -> Self {
        Self {
            context: None,
            config: WhisperConfig::default(),
            using_gpu: false,
            fallback_used: false,
        }
    }

    /// Check if GPU is being used
    pub fn is_using_gpu(&self) -> bool {
        self.using_gpu
    }

    /// Check if fallback to CPU was used
    pub fn was_fallback_used(&self) -> bool {
        self.fallback_used
    }

    /// Get the current backend name
    pub fn get_backend_name(&self) -> String {
        if self.using_gpu {
            "GPU".to_string()
        } else {
            "CPU".to_string()
        }
    }

    /// Load a model from the given path (legacy method, uses GPU if available)
    pub fn load_model(&mut self, model_path: PathBuf) -> Result<(), WhisperError> {
        self.load_model_with_options(model_path, false).map(|_| ())
    }

    /// Load a model with explicit CPU/GPU control
    /// Returns ModelLoadResult with details about the loading
    pub fn load_model_with_options(
        &mut self,
        model_path: PathBuf,
        force_cpu: bool,
    ) -> Result<ModelLoadResult, WhisperError> {
        if !model_path.exists() {
            return Err(WhisperError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        // 简化 GPU 检测逻辑，避免在测试环境中崩溃
        let should_use_gpu = !force_cpu && self.is_gpu_available_safe();

        info!(
            "Loading Whisper model: {} (force_cpu={}, should_use_gpu={})",
            model_path.display(),
            force_cpu,
            should_use_gpu
        );

        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| WhisperError::LoadError("Invalid model path".to_string()))?;

        // First attempt: with GPU if available and not forced CPU
        if should_use_gpu {
            info!("Attempting to load model with GPU...");

            let mut params = WhisperContextParameters::default();
            params.use_gpu(true);

            match WhisperContext::new_with_params(model_path_str, params) {
                Ok(ctx) => {
                    self.context = Some(ctx);
                    self.config.model_path = model_path;
                    self.using_gpu = true;
                    self.fallback_used = false;

                    info!("Whisper model loaded successfully with GPU acceleration");

                    return Ok(ModelLoadResult {
                        success: true,
                        using_gpu: true,
                        backend: "GPU".to_string(),
                        fallback_used: false,
                    });
                }
                Err(gpu_error) => {
                    warn!(
                        "GPU loading failed: {}. Retrying with CPU fallback...",
                        gpu_error
                    );

                    // Fall through to CPU attempt
                }
            }
        }

        // CPU attempt (either forced or as fallback)
        info!("Loading model with CPU...");

        let mut cpu_params = WhisperContextParameters::default();
        cpu_params.use_gpu(false);

        let ctx = WhisperContext::new_with_params(model_path_str, cpu_params)
            .map_err(|e| WhisperError::LoadError(format!("CPU loading failed: {}", e)))?;

        self.context = Some(ctx);
        self.config.model_path = model_path;
        self.using_gpu = false;
        self.fallback_used = should_use_gpu; // True if we tried GPU first and failed

        if self.fallback_used {
            info!("Whisper model loaded with CPU (fallback from GPU failure)");
        } else {
            info!("Whisper model loaded with CPU (as requested)");
        }

        Ok(ModelLoadResult {
            success: true,
            using_gpu: false,
            backend: "CPU".to_string(),
            fallback_used: self.fallback_used,
        })
    }

    /// 安全的 GPU 可用性检查，避免在测试环境中崩溃
    fn is_gpu_available_safe(&self) -> bool {
        // 在测试环境中，总是返回 false 以避免 GPU 相关的崩溃
        #[cfg(test)]
        {
            false
        }

        // 在非测试环境中，进行简单的 GPU 检测
        #[cfg(not(test))]
        {
            // 只在 macOS 上启用 Metal，其他平台默认使用 CPU
            #[cfg(target_os = "macos")]
            {
                return true; // macOS 通常支持 Metal
            }

            #[cfg(not(target_os = "macos"))]
            {
                false // 其他平台默认使用 CPU 以避免问题
            }
        }
    }

    /// Set the language for transcription (None for auto-detect)
    pub fn set_language(&mut self, language: Option<String>) {
        self.config.language = language;
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self) -> bool {
        self.context.is_some()
    }

    /// Transcribe audio samples (i16 PCM, 16kHz mono)
    #[allow(dead_code)]
    pub fn transcribe(&self, samples: &[i16]) -> Result<String, WhisperError> {
        let result = self.transcribe_to_segment(samples);
        match result {
            Ok(res) => Ok(res.all_text),
            Err(e) => Err(e),
        }
    }

    pub fn transcribe_to_segment(&self, samples: &[i16]) -> Result<WhisperResult, WhisperError> {
        let ctx = self.context.as_ref().ok_or(WhisperError::NotLoaded)?;

        if samples.is_empty() {
            return Err(WhisperError::InvalidAudio);
        }

        // 验证音频数据的合理性
        if samples.len() < 1600 {
            // 至少0.1秒的音频
            warn!("Audio data is very short: {} samples", samples.len());
        }

        // Convert i16 samples to f32 (whisper-rs expects f32)
        // 使用更安全的转换方法
        let samples_f32: Vec<f32> = samples
            .iter()
            .map(|&s| {
                // 更精确的转换，避免除零和溢出
                if s == i16::MIN {
                    -1.0f32
                } else {
                    s as f32 / i16::MAX as f32
                }
            })
            .collect();

        // 验证转换后的数据
        if samples_f32.iter().all(|&x| x.abs() < 1e-6) {
            warn!("Audio appears to be silent or very quiet");
        }

        info!(
            "Transcribing {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f32 / 16000.0
        );

        // Create transcription parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Set language
        if let Some(ref lang) = self.config.language {
            params.set_language(Some(lang));
        } else {
            params.set_language(None); // Auto-detect
        }

        params.set_translate(self.config.translate);
        params.set_n_threads(self.config.n_threads);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Create a new state for this transcription
        let mut state = ctx.create_state().map_err(|e| {
            WhisperError::TranscriptionError(format!("Failed to create state: {}", e))
        })?;

        // Run transcription with error handling
        state.full(params, &samples_f32).map_err(|e| {
            WhisperError::TranscriptionError(format!("Transcription failed: {}", e))
        })?;
        let mut whisper_result: WhisperResult = WhisperResult::new();
        // Get the transcription result
        let num_segments = state.full_n_segments();

        let mut segments: Vec<WhisperSegment> = Vec::new();
        if num_segments == 0 {
            warn!("No segments found in transcription");
            return Ok(whisper_result);
        }
        let mut all_text: String = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                let seg_text = segment.to_string();
                if !seg_text.trim().is_empty() {
                    all_text.push_str(&seg_text);
                    all_text.push(' ');
                }
                let whisper_segment: WhisperSegment = WhisperSegment {
                    text: seg_text,
                    start: segment.start_timestamp() as f64,
                    end: segment.end_timestamp() as f64,
                };
                segments.push(whisper_segment);
            }
        }
        whisper_result.all_text = all_text;
        whisper_result.segments = segments;
        Ok(whisper_result)
    }
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = WhisperConfig::default();
        assert!(config.language.is_none());
        assert!(!config.translate);
        // n_threads is now dynamic (75% of CPUs), just ensure it's at least 1
        assert!(config.n_threads >= 1);
    }

    #[test]
    fn test_engine_not_loaded() {
        let engine = WhisperEngine::new();
        assert!(!engine.is_loaded());

        let result = engine.transcribe(&[0i16; 1000]);
        assert!(matches!(result, Err(WhisperError::NotLoaded)));
    }
}
