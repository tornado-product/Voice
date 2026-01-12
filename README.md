# Voice

一个功能强大的 Rust 语音处理库，提供语音转文本（STT）和文本转语音（TTS）功能。

## 功能特性

### 语音转文本 (STT)
- ✅ **本地 Whisper 模型支持** - 使用 whisper-rs 实现本地语音识别
- ✅ **多提供商支持** - 支持 OpenAI Whisper API、Azure Speech、Google Cloud Speech-to-Text
- ✅ **GPU 加速** - 支持 CUDA、Metal、CoreML、Vulkan、ROCm 等多种 GPU 后端
- ✅ **多语言识别** - 支持自动语言检测和指定语言识别
- ✅ **时间戳支持** - 提供字级和段级时间戳信息

### 文本转语音 (TTS)
- ✅ **多提供商支持** - 支持以下 TTS 服务：
  - OpenAI TTS API
  - Azure Cognitive Services Speech
  - Google Cloud Text-to-Speech
  - Kokoro TTS (本地模型)
  - Microsoft Edge TTS
  - TikTok TTS
  - 通义千问 (Qwen) TTS
- ✅ **本地 TTS 模型** - 支持 Kokoro TTS 本地模型
- ✅ **多种语音选择** - 支持多种语音和参数配置
- ✅ **语音速度控制** - 可调节语音播放速度

### 音频处理
- ✅ **多格式支持** - 支持 WAV、MP3、FLAC、OGG、AAC、M4A 等格式
- ✅ **音频编解码** - 完整的音频编解码功能
- ✅ **音频重采样** - 支持音频采样率转换
- ✅ **声道处理** - 支持单声道/立体声转换

### 其他特性
- ✅ **多语言支持** - 内置中英文国际化支持
- ✅ **异步处理** - 基于 Tokio 的异步架构
- ✅ **错误处理** - 完善的错误处理机制
- ✅ **配置灵活** - 丰富的配置选项

## 系统要求

- Rust 1.70+ (Edition 2021)
- Windows / Linux / macOS

### GPU 支持（可选）

根据你的硬件平台，可以启用相应的 GPU 加速功能：

- **CUDA** - NVIDIA GPU (需要 CUDA 工具包)
- **Metal** - Apple Silicon / macOS
- **CoreML** - Apple 设备
- **Vulkan** - 跨平台 GPU 支持
- **ROCm** - AMD GPU
- **OpenBLAS** - CPU 优化

## 安装

### 添加依赖

在你的 `Cargo.toml` 中添加：

```toml
[dependencies]
voice = { path = "../Voice" }  # 本地路径
# 或
# voice = { git = "https://your-repo/voice.git" }
```

### 启用 GPU 支持（可选）

```toml
[dependencies]
voice = { path = "../Voice", features = ["cuda"] }  # CUDA 支持
# 或
voice = { path = "../Voice", features = ["metal"] }  # Metal 支持
# 或
voice = { path = "../Voice", features = ["vulkan"] }  # Vulkan 支持
```

## 快速开始

### 语音转文本示例

```rust
use voice::stt::transcription::{create_local_whisper_service, TranscriptionService};
use voice::common::config::{AudioCodec, AudioFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建本地 Whisper 服务
    let service = create_local_whisper_service()?;
    
    // 加载 Whisper 模型
    let model_path = std::path::PathBuf::from("ggml-small.bin");
    service.load_whisper_model(model_path, true)?;
    
    // 加载音频文件
    let audio_data = AudioCodec::decode_file("audio.wav")?;
    
    // 转录音频
    let result = service.transcribe(&audio_data).await?;
    println!("转录结果: {}", result.text);
    
    Ok(())
}
```

### 文本转语音示例

```rust
use voice::tts::synthesis::{create_kokoro_synthesis_service, SynthesisService};
use voice::common::config::SynthesisConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建 Kokoro TTS 服务
    let service = create_kokoro_synthesis_service(0, 1.0)?;
    
    // 合成语音
    let text = "你好，这是一个测试";
    let result = service.synthesize(text).await?;
    
    // 保存音频
    std::fs::write("output.wav", result.audio.samples)?;
    
    Ok(())
}
```

## 项目结构

```
Voice/
├── src/
│   ├── main.rs              # 主程序入口
│   ├── lib.rs               # 库入口
│   ├── common/              # 通用模块
│   │   ├── config.rs        # 配置结构
│   │   └── error.rs         # 错误处理
│   ├── stt/                 # 语音转文本模块
│   │   ├── transcription.rs # 转录服务
│   │   ├── whisper_engine.rs
│   │   └── whisper_worker.rs
│   └── tts/                 # 文本转语音模块
│       ├── synthesis.rs     # 合成服务
│       ├── azure.rs         # Azure TTS
│       ├── msedge.rs        # Edge TTS
│       ├── kokoro_tts.rs    # Kokoro TTS
│       ├── qwen.rs          # 通义千问 TTS
│       └── tiktok.rs        # TikTok TTS
├── lang/                    # 国际化文件
│   ├── zh-CN.yml
│   └── en-US.yml
├── Cargo.toml
└── README.md
```

## 配置

### Whisper 模型

下载 Whisper 模型文件（例如 `ggml-small.bin`）并放置在项目目录中。可以从以下位置下载：

- [Hugging Face - whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)

### TTS 配置

不同 TTS 提供商需要不同的配置：

- **OpenAI**: 需要 API Key
- **Azure**: 需要 API Key 和 Endpoint
- **Kokoro**: 需要本地模型文件
- **其他**: 根据各提供商文档配置

## 依赖项

主要依赖包括：

- `whisper-rs` - Whisper 模型绑定
- `sherpa-rs` - Sherpa-ONNX 语音处理
- `symphonia` - 音频编解码
- `tokio` - 异步运行时
- `reqwest` - HTTP 客户端
- `serde` - 序列化/反序列化

## 许可证

本项目为私有项目，不对外发布。

## 贡献

欢迎提交 Issue 和 Pull Request。

## CI/CD 工作流

项目使用 GitHub Actions 进行持续集成和持续部署。

### 持续集成 (CI)

每次推送到主分支或创建 Pull Request 时，会自动运行以下检查：

- ✅ **代码格式检查** - 使用 `cargo fmt` 确保代码格式一致
- ✅ **静态分析** - 使用 `cargo clippy` 进行代码质量检查
- ✅ **多平台构建** - 在 Linux、Windows、macOS 上构建项目
- ✅ **单元测试** - 运行所有测试用例
- ✅ **文档生成** - 生成 API 文档
- ✅ **安全检查** - 使用 `cargo audit` 检查依赖漏洞
- ✅ **依赖更新检查** - 使用 `cargo outdated` 检查过时依赖

工作流配置文件位于 `.github/workflows/ci.yml`

### 持续部署 (CD)

当创建版本标签（格式：`v*.*.*`）时，会自动：

- 📦 **构建发布版本** - 在多个平台上构建 Release 版本
- 📝 **创建 GitHub Release** - 自动创建发布说明
- 📥 **生成发布包** - 为每个平台生成可下载的发布包
- 📦 **发布到 crates.io** - 自动发布到 Rust 包仓库（如果配置了 `CARGO_REGISTRY_TOKEN`）

工作流配置文件位于 `.github/workflows/release.yml`

### 手动发布到 crates.io

使用 `publish.yml` 工作流可以手动发布到 crates.io：

1. 进入 GitHub Actions 页面
2. 选择 "Publish to crates.io" 工作流
3. 输入版本号并运行

**前置条件**：
- 在 GitHub Secrets 中配置 `CARGO_REGISTRY_TOKEN`
- 确保 `Cargo.toml` 中 `publish != false`

工作流配置文件位于 `.github/workflows/publish.yml`

### 本地运行检查

在提交代码前，建议在本地运行以下命令：

```bash
# 格式化代码（自动修复格式问题）
cargo fmt --all

# 检查代码格式（不修改文件，仅检查）
cargo fmt --all -- --check

# 运行 Clippy
cargo clippy --all-targets --all-features -- -D warnings

# 运行测试
cargo test

# 构建项目
cargo build --release
```

**提示**：
- 使用 `cargo fmt --all` 可以自动格式化所有代码
- 使用 `cargo fmt --all -- --check` 可以检查格式是否符合要求（CI 使用此命令）
- 项目包含 `rustfmt.toml` 配置文件，用于统一代码格式化规则

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)

## 注意事项

### 编译问题

如果遇到编译问题，特别是与 whisper.cpp 相关的中文字符编码问题，请参考 `build.rs` 中的注释说明。

### 模型文件

- Whisper 模型文件较大，请确保有足够的磁盘空间
- 首次使用需要下载相应的模型文件
- GPU 加速需要相应的驱动和工具包支持

## 联系方式

如有问题或建议，请通过 Issue 反馈。
