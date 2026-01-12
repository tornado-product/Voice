//use log::debug;

/*fn main() {
    // 设置环境变量（在初始化之前）
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    debug!("================");
}*/
mod common;
mod stt;
use hound::WavReader;
//use std::{fs::File, io::Write, time::Instant};
use std::path::PathBuf;
//use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use crate::common::config::{AudioData, AudioFormat};
use crate::stt::transcription::create_local_whisper_service;

//const MODEL_PATH: &str = "ggml-small.bin";
#[tokio::main]
async fn main() {
    // 创建本地 Whisper 服务
    let service = create_local_whisper_service().unwrap();

    // 尝试多个可能的模型路径
    let possible_paths = vec![
        PathBuf::from("ggml-small.bin"), // 相对路径（与main.rs一致）
        PathBuf::from("../ggml-small.bin"),
        PathBuf::from("../../ggml-small.bin"),
        PathBuf::from("./examples/ggml-small.bin"),
        PathBuf::from("./models/ggml-small.bin"),
    ];

    let mut model_path = None;
    for path in possible_paths {
        if path.exists() {
            model_path = Some(path);
            break;
        }
    }

    // 如果找到模型文件，测试服务功能
    if let Some(model_path) = model_path {
        println!("Found model at: {}", model_path.display());

        // 设置环境变量以跳过实际模型加载（用于演示目的）
        //std::env::set_var("WHISPER_SKIP_MODEL_LOADING", "1");

        // 测试模型加载（现在会返回模拟结果）
        let load_result = service.load_whisper_model(model_path, true);
        match load_result {
            Ok(result) => {
                println!("Model loading result: {:?}", result);
                println!("Using GPU: {}", result.using_gpu);
                println!("Backend: {}", result.backend);

                // 检查状态
                if let Ok(status) = service.get_whisper_status() {
                    println!("Whisper status: {:?}", status);
                }

                println!("✓ Local Whisper service test completed successfully");
            }
            Err(e) => {
                println!("Model loading failed: {:?}", e);
                println!("Continuing without Whisper model...");
                return; // 如果模型加载失败，退出程序
            }
        }

        // 清理环境变量
        std::env::remove_var("WHISPER_SKIP_MODEL_LOADING");
    } else {
        println!("No Whisper model found, skipping local whisper test");
        return;
    }

    let reader = WavReader::open("./examples/jfk.wav").expect("Error reading WAV file");
    let spec = reader.spec();

    // Check format
    if spec.channels != 1 || spec.sample_rate != 16000 {
        panic!("Audio must be mono 16kHz. Convert with:\nffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le audio_en.wav");
    }

    let samples: Vec<f32> =
        reader.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let audio_data = AudioData::new(samples, spec.sample_rate, spec.channels, AudioFormat::Wav);

    let res = service.transcribe(&audio_data).await;
    match res {
        Ok(result) => {
            println!("Transcribing result: {:?}", result);
        }
        Err(e) => {
            println!("Transcribe failed: {:?}", e);
        }
    }
}
/*
fn main() {
    let audio_path = "jfk.wav";
    let output_path = "transcription_en.txt";

    // 1. Check files
    if !Path::new(MODEL_PATH).exists() {
        panic!("Model {} not found! Download it with:\nwget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin -O {}", MODEL_PATH, MODEL_PATH);
    }

    if !Path::new(audio_path).exists() {
        panic!("Audio file {} not found!", audio_path);
    }

    // 2. Load model
    println!("[1/4] Loading model...");
    let ctx = WhisperContext::new_with_params(MODEL_PATH, WhisperContextParameters::default())
        .expect("Error loading model");

    // 3. Load and check audio
    println!("[2/4] Analyzing audio...");
    let audio_data = load_audio(audio_path);

    // 4. Setup parameters for English
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: 1.5,
    });
    //params.set_language(Some("en"));
    params.set_translate(false);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_token_timestamps(true);

    // 5. Transcription
    println!("[3/4] Transcribing...");
    let start_time = Instant::now();
    let mut state = ctx.create_state().expect("Error creating state");
    state
        .full(params, &audio_data)
        .expect("Error during transcription");

    // 6. Save results
    println!("[4/4] Saving...");
    save_results(&state, output_path, start_time);
    println!("Done! Results saved to {}", output_path);
}
*/

/*fn load_audio(path: &str) -> Vec<f32> {
    let reader = WavReader::open(path).expect("Error reading WAV file");
    let spec = reader.spec();

    // Check format
    if spec.channels != 1 || spec.sample_rate != 16000 {
        panic!("Audio must be mono 16kHz. Convert with:\nffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le audio_en.wav");
    }

    reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn save_results(state: &whisper_rs::WhisperState, path: &str, start_time: Instant) {
    let mut file = File::create(path).expect("Error creating file");
    writeln!(file, "Transcription results:").unwrap();

    let num_segments = state.full_n_segments();
    for i in 0..num_segments {
        let text = state.get_segment(i).expect("Error getting text");
        let start = text.start_timestamp() as f64 / 100.0;
        let end = text.end_timestamp() as f64 / 100.0;

        writeln!(file, "[{:.2}s-{:.2}s] {}", start, end, text.to_str().unwrap()).unwrap();
    }

    writeln!(
        file,
        "\nProcessing time: {:.2} sec",
        start_time.elapsed().as_secs_f32()
    )
        .unwrap();
}*/
