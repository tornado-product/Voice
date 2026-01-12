use crate::tts::kokoro_tts::{KokoroTts, KokoroTtsConfig};
use crate::tts::kokoro_voice::KokoroVoice;
use anyhow::anyhow;
use sherpa_rs::tts::TtsAudio;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// 线程安全的 TTS Worker
#[derive(Clone)]
pub struct KokoroTtsWorker {
    engine: Arc<Mutex<KokoroTts>>,
}

impl KokoroTtsWorker {
    /// 创建新的 TTS Worker（不自动加载模型）
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(KokoroTts::new())),
        }
    }

    /// 创建并加载默认模型
    #[allow(dead_code)]
    pub fn new_with_default_model() -> anyhow::Result<Self> {
        let worker = Self::new();
        worker.load_default_model()?;
        Ok(worker)
    }

    /// 加载默认模型（线程安全）
    pub fn load_default_model(&self) -> anyhow::Result<bool> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Failed to acquire KokoroTts lock: {}", e))?
            .load_default_model()
    }

    /// 动态加载模型（线程安全）
    #[allow(dead_code)]
    pub fn load_model(&self, model_path: PathBuf) -> anyhow::Result<bool> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Failed to acquire KokoroTts lock: {}", e))?
            .load_model(model_path)
    }

    /// 使用完整配置加载模型（线程安全）
    #[allow(dead_code)]
    pub fn load_model_with_config(&self, config: KokoroTtsConfig) -> anyhow::Result<bool> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Failed to acquire KokoroTts lock: {}", e))?
            .load_model_with_config(config)
    }

    /// 检查模型是否已加载（线程安全）
    #[allow(dead_code)]
    pub fn is_loaded(&self) -> bool {
        self.engine.lock().map(|engine| engine.is_loaded()).unwrap_or(false)
    }

    /// 获取当前模型路径（线程安全）
    #[allow(dead_code)]
    pub fn get_model_path(&self) -> Option<PathBuf> {
        self.engine
            .lock()
            .ok()
            .and_then(|engine| engine.get_model_path().map(|p| p.to_path_buf()))
    }

    /// 生成音频数据（线程安全）
    pub fn speak(&self, text: &str, voice: KokoroVoice, speed: f32) -> anyhow::Result<TtsAudio> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Failed to acquire KokoroTts lock: {}", e))?
            .speak(text, voice, speed)
    }

    /// 生成音频文件并返回路径（线程安全）
    #[allow(dead_code)]
    pub fn speak_to_file(
        &self,
        text: &str,
        voice: KokoroVoice,
        speed: f32,
        save_dir: Option<&str>,
    ) -> anyhow::Result<String> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Failed to acquire KokoroTts lock: {}", e))?
            .speak_to_file(text, voice, speed, save_dir)
    }

    /// 卸载模型（线程安全）
    #[allow(dead_code)]
    pub fn unload_model(&self) -> anyhow::Result<()> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Failed to acquire lock: {}", e))?
            .unload_model();
        Ok(())
    }
}

impl Default for KokoroTtsWorker {
    fn default() -> Self {
        Self::new()
    }
}

// ========== 使用示例 ==========

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_default_model() {
        let mut tts = KokoroTts::new();
        assert!(!tts.is_loaded());

        // 加载默认模型（可能失败如果文件不存在）
        let result = tts.load_default_model();
        if result.is_ok() {
            assert!(tts.is_loaded());
        }
    }

    #[test]
    fn test_load_custom_model() {
        let mut tts = KokoroTts::new();
        let model_path = PathBuf::from("./custom-model/model.onnx");

        // 如果文件不存在，应该返回错误
        let result = tts.load_model(model_path);
        if result.is_err() {
            assert!(!tts.is_loaded());
        }
    }

    #[test]
    fn test_worker_thread_safety() {
        let worker = KokoroTtsWorker::new();

        // 跨线程共享
        let worker_clone = worker.clone();
        let handle = std::thread::spawn(move || worker_clone.is_loaded());

        let loaded = handle.join().unwrap();
        assert!(!loaded); // 初始未加载
    }
}

// ========== 实际使用示例 ==========

#[cfg(test)]
mod usage_examples {
    use super::*;
    #[test]
    fn example_1_lazy_loading() {
        // 延迟加载：先创建，需要时再加载
        let tts = KokoroTtsWorker::new();

        // ... 应用启动后 ...

        // 用户触发 TTS 功能时才加载模型
        if let Err(e) = tts.load_default_model() {
            eprintln!("Failed to load model: {}", e);
            return;
        }

        // 现在可以使用了
        let _audio = tts.speak("Hello", KokoroVoice::default(), 1.0).unwrap();
    }
    #[test]
    fn example_2_eager_loading() {
        // 立即加载：启动时就加载好
        let tts = match KokoroTtsWorker::new_with_default_model() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to initialize TTS: {}", e);
                return;
            }
        };

        // 直接使用
        let _audio = tts.speak("Ready to go", KokoroVoice::default(), 1.0).unwrap();
    }
    #[test]
    #[ignore] // 需要模型文件，仅在本地有模型时运行
    fn example_3_switch_models() {
        let tts = KokoroTtsWorker::new();

        // 检查模型文件是否存在
        let zh_model = PathBuf::from("./models/zh/model.onnx");
        let en_model = PathBuf::from("./models/en/model.onnx");

        if !zh_model.exists() || !en_model.exists() {
            println!("Model files not found, skipping test");
            return;
        }

        // 加载中文模型
        tts.load_model(zh_model).unwrap();
        let _ = tts.speak("你好", KokoroVoice::default(), 1.0);

        // 切换到英文模型
        tts.load_model(en_model).unwrap();
        let _ = tts.speak("Hello", KokoroVoice::default(), 1.0);
    }
    #[test]
    #[ignore] // 需要模型文件，仅在本地有模型时运行
    fn example_4_custom_config() {
        let tts = KokoroTtsWorker::new();

        // 使用自定义配置
        let config = KokoroTtsConfig {
            model_path: PathBuf::from("./my-model/model.onnx"),
            tokens_path: PathBuf::from("./my-model/tokens.txt"),
            dict_dir: PathBuf::from("./my-model/dict"),
            lexicon_path: PathBuf::from("./my-model/lexicon.txt"),
            length_scale: 0.9, // 自定义语速缩放
        };

        // 检查模型文件是否存在
        if !config.model_path.exists() {
            println!("Model file not found, skipping test");
            return;
        }

        tts.load_model_with_config(config).unwrap();
    }
}
