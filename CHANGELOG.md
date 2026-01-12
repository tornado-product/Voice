# 更新日志

本文档记录 Voice 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [1.0.1] - 2024-XX-XX

### 新增
- 初始版本发布
- 语音转文本（STT）功能
  - 本地 Whisper 模型支持
  - OpenAI Whisper API 集成
  - Azure Speech Services 支持
  - Google Cloud Speech-to-Text 支持
  - GPU 加速支持（CUDA、Metal、CoreML、Vulkan、ROCm）
  - 多语言识别和自动语言检测
  - 字级和段级时间戳支持

- 文本转语音（TTS）功能
  - OpenAI TTS API 集成
  - Azure Cognitive Services Speech 集成
  - Google Cloud Text-to-Speech 集成
  - Kokoro TTS 本地模型支持
  - Microsoft Edge TTS 支持
  - TikTok TTS 支持
  - 通义千问（Qwen）TTS 支持
  - 多种语音选择和参数配置
  - 语音速度控制

- 音频处理功能
  - 多格式音频编解码（WAV、MP3、FLAC、OGG、AAC、M4A）
  - 音频重采样功能
  - 单声道/立体声转换
  - 音频质量配置

- 国际化支持
  - 中文（简体）支持
  - 英文支持
  - 基于 rust-i18n 的国际化框架

- 错误处理
  - 完善的错误类型定义
  - 详细的错误信息
  - 错误链支持

- 配置系统
  - 灵活的配置结构
  - 默认配置支持
  - 多提供商配置

### 技术特性
- 基于 Tokio 的异步架构
- 线程安全的服务实现
- 模块化设计
- 完善的文档注释

### 依赖项
- whisper-rs (来自 Gitee)
- sherpa-rs 0.6.8
- symphonia 0.5
- tokio 1.0
- reqwest 0.11
- serde 1.0
- 其他相关依赖

### 已知问题
- whisper.cpp 编译时可能存在中文字符编码问题（参考 build.rs 注释）
- 某些 GPU 后端可能需要额外的系统配置

### 文档
- 初始 README 文档
- 代码注释和文档

---

## [未发布]

### 计划中
- [ ] 流式语音识别支持
- [ ] 更多 TTS 提供商集成
- [ ] 音频效果处理（降噪、标准化等）
- [ ] WebSocket 支持
- [ ] 更完善的错误恢复机制
- [ ] 性能优化
- [ ] 单元测试和集成测试
- [ ] 更多语言支持
- [ ] 示例代码和教程

---

## 版本说明

- **主版本号**：不兼容的 API 修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

## 链接

- [README.md](README.md) - 项目说明文档
- [Cargo.toml](Cargo.toml) - 项目配置和依赖
