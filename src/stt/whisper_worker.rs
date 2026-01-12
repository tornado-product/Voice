use crate::stt::whisper_engine::{WhisperEngine, WhisperError, WhisperResult};
use crate::stt::ModelLoadResult;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

/// Thread-safe wrapper for WhisperEngine
pub struct WhisperWorker {
    engine: Arc<Mutex<WhisperEngine>>,
}

impl WhisperWorker {
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(WhisperEngine::new())),
        }
    }

    /// Load a model (thread-safe)
    #[allow(dead_code)]
    pub fn load_model(&self, model_path: PathBuf) -> Result<(), WhisperError> {
        self.engine.lock().load_model(model_path)
    }

    /// Load a model with explicit CPU/GPU control (thread-safe)
    pub fn load_model_with_options(
        &self,
        model_path: PathBuf,
        force_cpu: bool,
    ) -> Result<ModelLoadResult, WhisperError> {
        self.engine.lock().load_model_with_options(model_path, force_cpu)
    }

    /// Set language (thread-safe)
    pub fn set_language(&self, language: Option<String>) {
        self.engine.lock().set_language(language);
    }

    /// Check if model is loaded (thread-safe)
    pub fn is_loaded(&self) -> bool {
        self.engine.lock().is_loaded()
    }

    /// Check if GPU is being used (thread-safe)
    pub fn is_using_gpu(&self) -> bool {
        self.engine.lock().is_using_gpu()
    }

    /// Check if fallback was used (thread-safe)
    pub fn was_fallback_used(&self) -> bool {
        self.engine.lock().was_fallback_used()
    }

    /// Get current backend name (thread-safe)
    pub fn get_backend_name(&self) -> String {
        self.engine.lock().get_backend_name()
    }

    /// Transcribe samples (thread-safe)
    #[allow(dead_code)]
    pub fn transcribe(&self, samples: &[i16]) -> Result<String, WhisperError> {
        self.engine.lock().transcribe(samples)
    }
    pub fn transcribe_to_segment(&self, samples: &[i16]) -> Result<WhisperResult, WhisperError> {
        self.engine.lock().transcribe_to_segment(samples)
    }
}

impl Default for WhisperWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for WhisperWorker {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(dead_code)]
    fn example_single_instance() {
        // 创建单个实例
        let _worker = WhisperWorker::new();

        // 使用
        //let audio = worker.transcribe().unwrap();
    }
    #[allow(dead_code)]
    fn example_shared_instance() {
        // 创建共享实例
        let _worker = WhisperWorker::new();

        // 跨线程共享（通过 clone）
        let _worker_clone = _worker.clone();
        std::thread::spawn(move || {
            //let _ = tts_clone.speak_data("Thread 1", KokoroVoice::default(), 1.0);
        });

        // 原实例仍可使用
        //let _ = tts.speak_data("Main thread", KokoroVoice::default(), 1.0);
    }
}
