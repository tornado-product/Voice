// 独立的测试二进制文件，用于调试Whisper加载问题
use std::path::Path;
use whisper_rs::{WhisperContext, WhisperContextParameters};

fn main() {
    println!("=== Independent Whisper Test Binary ===");

    const MODEL_PATH: &str = "ggml-small.bin";

    if !Path::new(MODEL_PATH).exists() {
        println!("Model {} not found!", MODEL_PATH);
        return;
    }

    println!("Model file exists: {}", MODEL_PATH);
    println!("Current thread: {:?}", std::thread::current().id());
    println!("Thread name: {:?}", std::thread::current().name());

    // 尝试加载模型
    println!("Attempting to load Whisper model...");

    match WhisperContext::new_with_params(MODEL_PATH, WhisperContextParameters::default()) {
        Ok(_ctx) => {
            println!("✓ SUCCESS: Whisper model loaded successfully in independent binary!");
        }
        Err(e) => {
            println!("✗ ERROR: Failed to load Whisper model: {:?}", e);
        }
    }

    println!("Test completed.");
}
