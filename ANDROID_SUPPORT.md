# Android 平台支持分析

本文档分析 Voice 项目编译为 Android 库的可行性和所需步骤。

## 可行性评估

### ✅ 可以编译，但需要一些修改

项目**可以**编译为 Android 库，但需要解决以下问题：

## 主要障碍和解决方案

### 1. **tokio-tungstenite 的 TLS 后端** ⚠️

**问题**：
- 当前使用 `native-tls` 特性，在 Android 上可能不稳定
- Android 系统库的 TLS 支持可能不完整

**解决方案**：
```toml
# 修改 Cargo.toml
tokio-tungstenite = { version = "0.23.1", features = ["rustls-tls"], default-features = false }
```

### 2. **whisper-rs 的 C++ 依赖** ⚠️

**问题**：
- `whisper-rs` 依赖 `whisper.cpp`，需要 C++ 编译
- 需要 Android NDK 和 CMake

**解决方案**：
- 安装 Android NDK
- 配置交叉编译工具链
- 可能需要修改 `whisper-rs-sys` 的构建脚本

### 3. **平台特定依赖** ✅

**当前状态**：
- Windows、Linux、macOS 依赖已正确配置为平台特定
- 没有 Android 特定依赖冲突

**需要添加**：
```toml
# Android dependencies (如果需要)
[target.'cfg(target_os = "android")'.dependencies]
# 可以添加 Android 特定的依赖
```

### 4. **其他依赖检查** ✅

大部分依赖支持 Android：
- ✅ `serde`, `serde_json` - 纯 Rust，支持 Android
- ✅ `reqwest` (使用 rustls-tls) - 支持 Android
- ✅ `tokio` - 支持 Android
- ✅ `sherpa-rs` - 需要检查，可能支持
- ⚠️ `symphonia` - 需要检查音频编解码在 Android 上的支持
- ✅ `hound`, `minimp3` - 纯 Rust，应该支持
- ✅ 其他依赖 - 大部分支持

## 编译步骤

### 1. 安装 Android 目标

```bash
# 添加 Android 目标（根据需要的架构选择）
rustup target add aarch64-linux-android    # ARM 64-bit
rustup target add armv7-linux-androideabi # ARM 32-bit
rustup target add i686-linux-android      # x86 32-bit
rustup target add x86_64-linux-android    # x86 64-bit
```

### 2. 安装 Android NDK

下载并安装 Android NDK (推荐 r25c 或更新版本)

### 3. 配置交叉编译

创建或编辑 `~/.cargo/config.toml`：

```toml
[target.aarch64-linux-android]
ar = "/path/to/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
linker = "/path/to/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang"

[target.armv7-linux-androideabi]
ar = "/path/to/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
linker = "/path/to/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi21-clang"
```

### 4. 修改 Cargo.toml

添加 Android 库类型配置：

```toml
[lib]
name = "voice"
path = "src/lib.rs"
crate-type = ["cdylib", "rlib"]  # cdylib 用于生成 .so 文件
```

### 5. 条件编译

可能需要为 Android 平台添加条件编译：

```rust
#[cfg(target_os = "android")]
// Android 特定代码
```

### 6. 编译

```bash
cargo build --target aarch64-linux-android --release
```

## 需要修改的文件

### 1. Cargo.toml

```toml
# 修改 tokio-tungstenite
tokio-tungstenite = { version = "0.23.1", features = ["rustls-tls"], default-features = false }

# 添加 Android 库类型
[lib]
crate-type = ["cdylib", "rlib"]

# 可选：添加 Android 特定依赖
[target.'cfg(target_os = "android")'.dependencies]
# 如果需要 Android 特定的依赖
```

### 2. 可能需要条件编译的代码

检查以下模块是否需要 Android 特定处理：
- `src/stt/whisper_engine.rs` - 可能需要 Android 特定的 GPU 检测
- `src/tts/` - 某些 TTS 提供商可能在 Android 上不可用
- 网络请求相关代码 - 确保使用 rustls

## 潜在问题

### 1. **whisper-rs 构建问题**

`whisper-rs-sys` 的构建脚本可能需要修改以支持 Android：
- 需要正确配置 CMake
- 可能需要禁用某些特性（如 CUDA、Metal）
- OpenBLAS 在 Android 上可能需要特殊配置

### 2. **运行时依赖**

生成的 `.so` 文件可能需要以下运行时依赖：
- C++ 标准库
- OpenBLAS（如果启用）
- 其他系统库

### 3. **JNI 接口**

如果需要从 Java/Kotlin 调用，需要：
- 创建 JNI 绑定层
- 使用 `jni` crate
- 处理字符串转换和内存管理

## 建议的实施方案

### 阶段 1：基础支持
1. 修改 `tokio-tungstenite` 使用 `rustls-tls`
2. 添加 Android 目标到 CI
3. 测试基础编译

### 阶段 2：功能适配
1. 条件编译平台特定功能
2. 测试核心功能（STT/TTS）
3. 处理 Android 特定的错误

### 阶段 3：优化
1. 添加 JNI 绑定（如果需要）
2. 优化性能
3. 减少依赖大小

## 结论

**项目可以编译为 Android 库**，但需要：
1. ✅ 修改 TLS 后端（简单）
2. ⚠️ 配置 Android NDK 和交叉编译（中等难度）
3. ⚠️ 处理 whisper-rs 的 C++ 依赖（可能较复杂）
4. ⚠️ 测试和调试（需要时间）

**推荐**：先从简单的功能开始测试，逐步添加复杂功能。

## 相关资源

- [Rust on Android](https://mozilla.github.io/firefox-browser-architecture/experiments/2017-09-21-rust-on-android.html)
- [cargo-ndk](https://github.com/bbqsrc/cargo-ndk) - 简化 Android 交叉编译的工具
- [Android NDK 文档](https://developer.android.com/ndk)
