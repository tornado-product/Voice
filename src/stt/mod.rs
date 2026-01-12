pub mod transcription;
mod whisper_engine;
mod whisper_worker;

//本代码来源于https://github.com/AccessDevops/S2Tui
pub use whisper_engine::ModelLoadResult;
pub use whisper_worker::WhisperWorker;
